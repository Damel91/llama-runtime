package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
)

type message struct {
	Protocol       int    `json:"protocol"`
	Status         string `json:"status"`
	ProxyURL       string `json:"proxy_url,omitempty"`
	BackendVersion string `json:"backend_version,omitempty"`
	Code           string `json:"code,omitempty"`
}

func emit(w io.Writer, m message) error { return json.NewEncoder(w).Encode(m) }

func errorCode(err error) string {
	if err == nil {
		return ""
	}
	switch err.Error() {
	case "state_in_use":
		return "state_in_use"
	case "state_profile_mismatch":
		return "state_mismatch"
	case "invalid_state", "foreign_state":
		return "state_invalid"
	case "proxy_start_failed":
		return "proxy_start_failed"
	default:
		return "tailnet_start_failed"
	}
}

func scrubEnvironment() {
	for _, item := range os.Environ() {
		name, _, _ := strings.Cut(item, "=")
		upper := strings.ToUpper(name)
		if strings.HasPrefix(upper, "TS") || upper == "HTTP_PROXY" || upper == "HTTPS_PROXY" || upper == "ALL_PROXY" || upper == "NO_PROXY" {
			_ = os.Unsetenv(name)
		}
	}
}

// run owns state/backend/proxy in that order. The enclosing process supervisor
// also enforces startup/shutdown bounds: tsnet.Start can block before Up begins
// observing its context, and Close must NEVER race that initialization.
func run(ctx context.Context, input io.Reader, output io.Writer, factory func(config) backend) int {
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 4096), maxControlLine+1)
	if !scanner.Scan() {
		_ = emit(output, message{Protocol: 1, Status: "error", Code: "invalid_config"})
		return 2
	}
	c, err := parseConfig(scanner.Bytes())
	if err != nil {
		_ = emit(output, message{Protocol: 1, Status: "error", Code: "invalid_config"})
		return 2
	}
	lease, err := acquireState(c)
	if err != nil {
		_ = emit(output, message{Protocol: 1, Status: "error", Code: errorCode(err)})
		return 2
	}
	b := factory(c)
	lifetime, cancel := context.WithCancel(ctx)
	defer cancel()
	control := make(chan string, 1)
	go func() {
		code := ""
		if scanner.Scan() {
			var command struct {
				Protocol int    `json:"protocol"`
				Command  string `json:"command"`
			}
			if decodeStrict(scanner.Bytes(), &command) != nil || command.Protocol != 1 || command.Command != "shutdown" {
				code = "invalid_command"
			}
		} else if scanner.Err() != nil {
			code = "invalid_command"
		}
		control <- code
		cancel()
	}()
	startup, startCancel := context.WithTimeout(lifetime, secondsDuration(c.StartupTimeoutSec))
	upDone := make(chan error, 1)
	go func() { upDone <- b.Up(startup) }()
	var upErr error
	select {
	case upErr = <-upDone:
		startCancel()
	case <-startup.Done():
		startCancel()
		code := "tailnet_start_timeout"
		if lifetime.Err() != nil {
			code = "startup_cancelled"
		}
		_ = emit(output, message{Protocol: 1, Status: "error", Code: code})
		// Do not race Close with synchronous Start. Hold the state lease until
		// Up returns; process exit/outer kill releases the OS lock if it never does.
		go func() { <-upDone; _ = b.Close(); _ = lease.Close() }()
		return 2
	}
	cleanup := func(p *proxy) bool {
		cancel()
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer lease.Close()
			shutdown, stop := context.WithTimeout(context.Background(), secondsDuration(c.ShutdownTimeoutSec))
			defer stop()
			if p != nil {
				_ = p.Close(shutdown)
			}
			_ = b.Close()
		}()
		shutdown, stop := context.WithTimeout(context.Background(), secondsDuration(c.ShutdownTimeoutSec))
		defer stop()
		select {
		case <-done:
			return true
		case <-shutdown.Done():
			return false
		}
	}
	if upErr != nil || lifetime.Err() != nil {
		code := errorCode(upErr)
		if lifetime.Err() != nil {
			code = "startup_cancelled"
		}
		_ = emit(output, message{Protocol: 1, Status: "error", Code: code})
		cleanup(nil)
		return 2
	}
	p, err := newProxy(lifetime, c, b)
	if err != nil {
		_ = emit(output, message{Protocol: 1, Status: "error", Code: errorCode(err)})
		cleanup(nil)
		return 2
	}
	served := make(chan error, 1)
	go func() { served <- p.Serve() }()
	if emit(output, message{Protocol: 1, Status: "ready", ProxyURL: p.URL(), BackendVersion: backendVersion}) != nil {
		cleanup(p)
		return 2
	}
	code := ""
	select {
	case code = <-control:
	case <-ctx.Done():
	case serveErr := <-served:
		if serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
			code = "proxy_failed"
		}
	}
	if !cleanup(p) {
		code = "shutdown_timeout"
	}
	if code != "" {
		_ = emit(output, message{Protocol: 1, Status: "error", Code: code})
		return 2
	}
	_ = emit(output, message{Protocol: 1, Status: "closed"})
	return 0
}

func main() {
	log.SetOutput(io.Discard)
	if len(os.Args) == 2 && os.Args[1] == "--version" {
		_ = emit(os.Stdout, message{Protocol: 1, Status: "version", BackendVersion: backendVersion})
		return
	}
	if len(os.Args) != 1 {
		_ = emit(os.Stdout, message{Protocol: 1, Status: "error", Code: "invalid_config"})
		os.Exit(2)
	}
	// The Python launcher also scrubs BEFORE exec, since upstream env knobs can
	// be captured by package initialization before main is entered.
	scrubEnvironment()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	os.Exit(run(ctx, os.Stdin, os.Stdout, newTSBackend))
}
