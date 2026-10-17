package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"tailscale.com/ipn/ipnstate"
	"tailscale.com/types/key"
)

func testConfig(t *testing.T) config {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return config{Protocol: 1, StateDir: filepath.Join(root, "state"), Hostname: "runtime-test", ControlURL: officialControlURL, ProxyToken: strings.Repeat("a", 32), StartupTimeoutSec: 1, DialTimeoutSec: 1, ShutdownTimeoutSec: 1}
}

func encoded(c config) []byte { data, _ := json.Marshal(c); return data }

func TestConfigStrictAndSanitized(t *testing.T) {
	c := testConfig(t)
	for _, tc := range []struct {
		name string
		edit func(map[string]any)
	}{
		{"unknown", func(m map[string]any) { m["secret_unknown"] = "secret" }},
		{"case_alias", func(m map[string]any) { delete(m, "protocol"); m["Protocol"] = 1 }},
		{"aliased_duplicate", func(m map[string]any) { m["Protocol"] = 1 }},
		{"null", func(m map[string]any) { m["ephemeral"] = nil }},
		{"zero", func(m map[string]any) { m["dial_timeout_sec"] = 0 }},
		{"overflow", func(m map[string]any) { m["dial_timeout_sec"] = 86401 }},
		{"fractional_protocol", func(m map[string]any) { m["protocol"] = 1.5 }},
		{"http_control", func(m map[string]any) { m["control_url"] = "http://secret.example" }},
		{"userinfo", func(m map[string]any) { m["control_url"] = "https://secret:password@example" }},
		{"invalid_port", func(m map[string]any) { m["control_url"] = "https://example:65536" }},
		{"relative_state", func(m map[string]any) { m["state_dir"] = "relative" }},
		{"token", func(m map[string]any) { m["proxy_token"] = "secret" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var m map[string]any
			_ = json.Unmarshal(encoded(c), &m)
			tc.edit(m)
			data, _ := json.Marshal(m)
			_, err := parseConfig(data)
			if err == nil || err.Error() != "invalid_config" {
				t.Fatalf("unsanitized result: %v", err)
			}
		})
	}
	for _, data := range [][]byte{[]byte(`{"protocol":1,"protocol":1}`), append(encoded(c), []byte(` {}`)...), bytes.Repeat([]byte("a"), maxControlLine+1)} {
		if _, err := parseConfig(data); err == nil {
			t.Fatal("accepted ambiguous input")
		}
	}
	c.ControlURL = ""
	c.Hostname = "Runtime-Test"
	parsed, err := parseConfig(encoded(c))
	if err != nil || parsed.ControlURL != officialControlURL || parsed.Hostname != "runtime-test" {
		t.Fatalf("defaults: %+v %v", parsed, err)
	}
}

func TestStateLockIdentityAndForeignState(t *testing.T) {
	c := testConfig(t)
	lease, err := acquireState(c)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := acquireState(c); err == nil || err.Error() != "state_in_use" {
		t.Fatalf("second identity lock: %v", err)
	}
	lease.Close()
	lease, err = acquireState(c)
	if err != nil {
		t.Fatal(err)
	}
	lease.Close()
	changed := c
	changed.Hostname = "other"
	if _, err := acquireState(changed); err == nil || err.Error() != "state_profile_mismatch" {
		t.Fatalf("profile mismatch: %v", err)
	}
	foreign := testConfig(t)
	if err := os.MkdirAll(foreign.StateDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(foreign.StateDir, "tailscaled.state"), []byte("secret"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := acquireState(foreign); err == nil || err.Error() != "foreign_state" {
		t.Fatalf("foreign state: %v", err)
	}
}

func TestStateRejectsSymlink(t *testing.T) {
	c := testConfig(t)
	if err := os.Mkdir(c.StateDir, 0700); err != nil {
		t.Fatal(err)
	}
	link := c.StateDir + "-link"
	if err := os.Symlink(c.StateDir, link); err != nil {
		t.Skip("symlink privilege unavailable")
	}
	c.StateDir = link
	if _, err := acquireState(c); err == nil {
		t.Fatal("symlink state accepted")
	}
}

func peerStatus(peers ...*ipnstate.PeerStatus) *ipnstate.Status {
	s := &ipnstate.Status{Peer: map[key.NodePublic]*ipnstate.PeerStatus{}}
	for _, peer := range peers {
		s.Peer[key.NewNode().Public()] = peer
	}
	return s
}

func TestVerifiedPeerResolution(t *testing.T) {
	ip := netip.MustParseAddr("100.101.102.103")
	s := peerStatus(&ipnstate.PeerStatus{DNSName: "workstation.private.example.", TailscaleIPs: []netip.Addr{ip}})
	for _, name := range []string{"workstation.private.example:8080", "WORKSTATION.PRIVATE.EXAMPLE.:8080", "100.101.102.103:8080"} {
		got, err := resolvePeer(s, name)
		if err != nil || got != netip.AddrPortFrom(ip, 8080) {
			t.Fatalf("%s: %v %v", name, got, err)
		}
	}
	for _, name := range []string{"workstation:8080", "example.com:8080", "1.1.1.1:8080", "127.0.0.1:8080", "100.101.102.103:0", "100.101.102.103:99999", "bad"} {
		if _, err := resolvePeer(s, name); err == nil {
			t.Fatalf("unknown peer admitted %s", name)
		}
	}
	ambiguous := peerStatus(&ipnstate.PeerStatus{DNSName: "same.example.", TailscaleIPs: []netip.Addr{ip}}, &ipnstate.PeerStatus{DNSName: "same.example.", TailscaleIPs: []netip.Addr{netip.MustParseAddr("100.101.102.104")}})
	if _, err := resolvePeer(ambiguous, "same.example:80"); err == nil || err.Error() != "ambiguous_peer" {
		t.Fatalf("ambiguous: %v", err)
	}
	for _, bad := range []string{"127.0.0.1", "::1", "::", "224.0.0.1", "fe80::1", "::ffff:100.1.1.1"} {
		if validPeerIP(netip.MustParseAddr(bad)) {
			t.Fatalf("invalid peer IP accepted %s", bad)
		}
	}
}

func TestBackendOfflineConfigurationAndMissingShimFailClosed(t *testing.T) {
	c := testConfig(t)
	c.ControlURL, c.Hostname, c.AuthKey, c.Ephemeral = "https://headscale.example", "isolated-node", "secret-not-printed", true
	b := newTSBackend(c).(*tsBackend)
	if b.server.ControlURL != c.ControlURL || b.server.Hostname != c.Hostname || b.server.AuthKey != c.AuthKey || !b.server.Ephemeral || b.server.Dir != c.StateDir {
		t.Fatal("explicit VPN identity configuration not preserved")
	}
	if _, err := b.Dial(context.Background(), "model.private.example:8080"); err == nil || err.Error() != "backend_unavailable" {
		t.Fatalf("uninitialized direct shim admitted a dial: %v", err)
	}
	if _, err := os.Stat(c.StateDir); !os.IsNotExist(err) {
		t.Fatal("offline construction created state or started backend")
	}
}

func TestDirectDialUsesFreshVerifiedNumericPeerAndNeverFallback(t *testing.T) {
	ip := netip.MustParseAddr("100.101.102.103")
	status := peerStatus(&ipnstate.PeerStatus{DNSName: "model.private.example.", TailscaleIPs: []netip.Addr{ip}})
	statuses, dials := 0, 0
	b := &tsBackend{
		status: func(context.Context) (*ipnstate.Status, error) { statuses++; return status, nil },
		direct: func(_ context.Context, target netip.AddrPort) (net.Conn, error) {
			dials++
			if target != netip.AddrPortFrom(ip, 8080) {
				t.Fatalf("non-numeric/unverified destination %v", target)
			}
			// A removed peer between verification and dialing must fail INSIDE
			// this callback; it cannot select a second OS dial or DNS resolver.
			status = peerStatus()
			return nil, errors.New("removed peer secret diagnostic")
		},
	}
	if _, err := b.Dial(context.Background(), "model.private.example:8080"); err == nil || err.Error() != "peer_dial_failed" {
		t.Fatalf("direct failure: %v", err)
	}
	if _, err := b.Dial(context.Background(), "model.private.example:8080"); err == nil || err.Error() != "unknown_peer" {
		t.Fatalf("stale peer admitted: %v", err)
	}
	if statuses != 2 || dials != 1 {
		t.Fatalf("status/dial counts %d/%d", statuses, dials)
	}
}

type fakeBackend struct {
	up    func(context.Context) error
	dial  func(context.Context, string) (net.Conn, error)
	close func() error
}

func (b *fakeBackend) Up(ctx context.Context) error {
	if b.up != nil {
		return b.up(ctx)
	}
	return nil
}
func (b *fakeBackend) Dial(ctx context.Context, address string) (net.Conn, error) {
	if b.dial == nil {
		return nil, errors.New("unknown peer with secrets")
	}
	return b.dial(ctx, address)
}
func (b *fakeBackend) Close() error {
	if b.close != nil {
		return b.close()
	}
	return nil
}

func startTestProxy(t *testing.T, b backend) (*proxy, *http.Client) {
	t.Helper()
	p, err := newProxy(context.Background(), testConfig(t), b)
	if err != nil {
		t.Fatal(err)
	}
	go p.Serve()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if err := p.Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
	})
	u, _ := url.Parse(p.URL())
	u.User = url.UserPassword("runtime", p.config.ProxyToken)
	transport := &http.Transport{Proxy: http.ProxyURL(u), TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}
	t.Cleanup(transport.CloseIdleConnections)
	return p, &http.Client{Transport: transport, Timeout: 2 * time.Second}
}

func loopbackDial(target string, calls *atomic.Int32) func(context.Context, string) (net.Conn, error) {
	return func(ctx context.Context, address string) (net.Conn, error) {
		calls.Add(1)
		if address != "model.private.example:8080" {
			return nil, errors.New("secret unknown peer")
		}
		return (&net.Dialer{}).DialContext(ctx, "tcp", target)
	}
}

func TestHTTPForwardPreservesHostStripsAuthAndNeverFollowsRedirect(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Host != "model.private.example:8080" || r.Header.Get("Proxy-Authorization") != "" {
			t.Errorf("origin headers: %s %v", r.Host, r.Header)
		}
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "http://internet.example/private", 302)
			return
		}
		data, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Write(data)
	}))
	defer upstream.Close()
	_, client := startTestProxy(t, &fakeBackend{dial: loopbackDial(strings.TrimPrefix(upstream.URL, "http://"), &calls)})
	response, err := client.Post("http://model.private.example:8080/embed", "application/json", strings.NewReader(`{"input":"text"}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if string(body) != `{"input":"text"}` {
		t.Fatalf("body: %s", body)
	}
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err = client.Get("http://model.private.example:8080/redirect")
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != 302 || calls.Load() != 2 {
		t.Fatalf("redirect/replay: status%d calls%d", response.StatusCode, calls.Load())
	}
}

func TestProxyRejectsMissingAuthenticationBeforeDial(t *testing.T) {
	var calls atomic.Int32
	p, _ := startTestProxy(t, &fakeBackend{dial: loopbackDial("127.0.0.1:1", &calls)})
	u, _ := url.Parse(p.URL())
	transport := &http.Transport{Proxy: http.ProxyURL(u)}
	defer transport.CloseIdleConnections()
	response, err := (&http.Client{Transport: transport}).Get("http://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != 407 || calls.Load() != 0 {
		t.Fatalf("auth %d %d", response.StatusCode, calls.Load())
	}
}

func TestHTTPSConnectPreservesTLSAndOriginHeaders(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil || r.TLS.ServerName != "model.private.example" || r.Header.Get("Proxy-Authorization") != "" {
			t.Errorf("TLS/headers: %+v %v", r.TLS, r.Header)
		}
		io.WriteString(w, "encrypted")
	}))
	defer server.Close()
	_, client := startTestProxy(t, &fakeBackend{dial: loopbackDial(strings.TrimPrefix(server.URL, "https://"), &calls)})
	response, err := client.Get("https://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil || string(body) != "encrypted" || calls.Load() != 1 {
		t.Fatalf("TLS response %q %v calls%d", body, err, calls.Load())
	}
}

func TestCONNECTPreservesBufferedBytesAndDrainsOnClose(t *testing.T) {
	upstream, peer := net.Pipe()
	finished := make(chan struct{})
	go func() { defer close(finished); defer peer.Close(); _, _ = io.Copy(peer, peer) }()
	p, _ := startTestProxy(t, &fakeBackend{dial: func(context.Context, string) (net.Conn, error) { return upstream, nil }})
	conn, err := net.Dial("tcp", p.listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(2 * time.Second))
	auth := base64.StdEncoding.EncodeToString([]byte("runtime:" + p.config.ProxyToken))
	fmt.Fprintf(conn, "CONNECT model.private.example:8080 HTTP/1.1\r\nHost: model.private.example:8080\r\nProxy-Authorization: Basic %s\r\n\r\npipelined-bytes", auth)
	reader := bufio.NewReader(conn)
	response, err := http.ReadResponse(reader, &http.Request{Method: "CONNECT"})
	if err != nil || response.StatusCode != 200 {
		t.Fatalf("CONNECT %v %v", response, err)
	}
	data := make([]byte, len("pipelined-bytes"))
	if _, err := io.ReadFull(reader, data); err != nil || string(data) != "pipelined-bytes" {
		t.Fatalf("buffered bytes %q %v", data, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := p.Close(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("tunnel upstream not drained")
	}
	p.mu.Lock()
	n := len(p.connections)
	p.mu.Unlock()
	if n != 0 {
		t.Fatalf("tracked connections remain: %d", n)
	}
}

func TestSSEFlushAndQuietBodyHaveNoHelperIdleTimeout(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		time.Sleep(100 * time.Millisecond)
		io.WriteString(w, "data: second\n\n")
	}))
	defer server.Close()
	_, client := startTestProxy(t, &fakeBackend{dial: loopbackDial(strings.TrimPrefix(server.URL, "http://"), &calls)})
	start := time.Now()
	response, err := client.Get("http://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	if err != nil || line != "data: first\n" || time.Since(start) > 80*time.Millisecond {
		t.Fatalf("SSE first chunk did not flush: %q %v", line, err)
	}
	remaining, err := io.ReadAll(reader)
	response.Body.Close()
	if err != nil || !strings.Contains(string(remaining), "second") {
		t.Fatalf("quiet SSE: %q %v", remaining, err)
	}
}

func TestTruncatedHTTPBodyAbortsDownstreamAndDoesNotReplay(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		_, _ = http.ReadRequest(bufio.NewReader(conn))
		io.WriteString(conn, "HTTP/1.1 200 OK\r\nContent-Length: 30\r\n\r\nshort")
	}()
	var calls atomic.Int32
	_, client := startTestProxy(t, &fakeBackend{dial: loopbackDial(listener.Addr().String(), &calls)})
	response, err := client.Get("http://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.ReadAll(response.Body)
	response.Body.Close()
	if err == nil || calls.Load() != 1 {
		t.Fatalf("truncation masked/replayed: %v calls%d", err, calls.Load())
	}
}

func TestTruncatedChunkedBodyCannotBecomeCleanEOF(t *testing.T) {
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		_, _ = http.ReadRequest(bufio.NewReader(conn))
		io.WriteString(conn, "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n")
	}()
	var calls atomic.Int32
	_, client := startTestProxy(t, &fakeBackend{dial: loopbackDial(listener.Addr().String(), &calls)})
	response, err := client.Get("http://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err == nil || string(body) != "{}" || calls.Load() != 1 {
		t.Fatalf("missing upstream chunk terminator masked or replayed: %q %v calls%d", body, err, calls.Load())
	}
}

type synchronizedBuffer struct {
	mu     sync.Mutex
	data   bytes.Buffer
	notify chan struct{}
}

func (b *synchronizedBuffer) Write(data []byte) (int, error) {
	b.mu.Lock()
	n, e := b.data.Write(data)
	b.mu.Unlock()
	select {
	case b.notify <- struct{}{}:
	default:
	}
	return n, e
}
func (b *synchronizedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.data.String()
}
func waitOutput(t *testing.T, b *synchronizedBuffer, needle string) {
	t.Helper()
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for !strings.Contains(b.String(), needle) {
		select {
		case <-b.notify:
		case <-deadline.C:
			t.Fatalf("missing %q: %s", needle, b.String())
		}
	}
}

func TestRunReadyShutdownAndSanitization(t *testing.T) {
	c := testConfig(t)
	input, writer := io.Pipe()
	defer input.Close()
	defer writer.Close()
	output := &synchronizedBuffer{notify: make(chan struct{}, 1)}
	closed := make(chan struct{})
	result := make(chan int, 1)
	go func() {
		result <- run(context.Background(), input, output, func(config) backend { return &fakeBackend{close: func() error { close(closed); return nil }} })
	}()
	fmt.Fprintf(writer, "%s\n", encoded(c))
	waitOutput(t, output, `"status":"ready"`)
	if strings.Contains(output.String(), c.ProxyToken) || strings.Contains(output.String(), c.StateDir) {
		t.Fatal("startup leaked secret/path")
	}
	io.WriteString(writer, "{\"protocol\":1,\"command\":\"shutdown\"}\n")
	if code := <-result; code != 0 {
		t.Fatalf("exit%d %s", code, output.String())
	}
	select {
	case <-closed:
	default:
		t.Fatal("backend not closed")
	}
	lease, err := acquireState(c)
	if err != nil {
		t.Fatalf("state not released: %v", err)
	}
	lease.Close()
}

func TestStartupTimeoutDoesNotRaceBackendClose(t *testing.T) {
	c := testConfig(t)
	c.StartupTimeoutSec = .03
	input, writer := io.Pipe()
	defer input.Close()
	defer writer.Close()
	output := &synchronizedBuffer{notify: make(chan struct{}, 1)}
	allowUp := make(chan struct{})
	closed := make(chan struct{})
	result := make(chan int, 1)
	go func() {
		result <- run(context.Background(), input, output, func(config) backend {
			return &fakeBackend{up: func(context.Context) error { <-allowUp; return errors.New("secret upstream error") }, close: func() error { close(closed); return nil }}
		})
	}()
	fmt.Fprintf(writer, "%s\n", encoded(c))
	if code := <-result; code != 2 {
		t.Fatal(code)
	}
	if !strings.Contains(output.String(), "tailnet_start_timeout") || strings.Contains(output.String(), "secret") {
		t.Fatalf("unsafe error: %s", output.String())
	}
	select {
	case <-closed:
		t.Fatal("Close raced Up")
	default:
	}
	if _, err := acquireState(c); err == nil {
		t.Fatal("state unlocked before backend stopped")
	}
	close(allowUp)
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("backend not cleaned after Up")
	}
	deadline := time.Now().Add(time.Second)
	for {
		lease, err := acquireState(c)
		if err == nil {
			lease.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("state never released")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestStartupFailureRedactsProviderError(t *testing.T) {
	c := testConfig(t)
	input, writer := io.Pipe()
	defer input.Close()
	defer writer.Close()
	var output bytes.Buffer
	result := make(chan int, 1)
	go func() {
		result <- run(context.Background(), input, &output, func(config) backend {
			return &fakeBackend{up: func(context.Context) error { return errors.New("authkey-secret login-url private-path") }}
		})
	}()
	fmt.Fprintf(writer, "%s\n", encoded(c))
	if code := <-result; code != 2 {
		t.Fatal(code)
	}
	if !strings.Contains(output.String(), "tailnet_start_failed") || strings.Contains(output.String(), "secret") {
		t.Fatalf("diagnostic leak %s", output.String())
	}
}

func TestProxyDialTimeoutIsFiniteAndFailureSanitized(t *testing.T) {
	b := &fakeBackend{dial: func(ctx context.Context, _ string) (net.Conn, error) {
		<-ctx.Done()
		return nil, errors.New("secret address and auth key")
	}}
	p, client := startTestProxy(t, b)
	p.config.DialTimeoutSec = .03
	start := time.Now()
	response, err := client.Get("http://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if response.StatusCode != 502 || string(body) != `{"error":"peer_dial_failed"}` || time.Since(start) > time.Second {
		t.Fatalf("dial deadline/diagnostic: %d %s", response.StatusCode, body)
	}
}

func TestProxyCloseInterruptsQuietHTTPAndDrainsConnections(t *testing.T) {
	entered, stopped := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		w.(http.Flusher).Flush()
		close(entered)
		<-r.Context().Done()
		close(stopped)
	}))
	defer server.Close()
	var calls atomic.Int32
	p, client := startTestProxy(t, &fakeBackend{dial: loopbackDial(strings.TrimPrefix(server.URL, "http://"), &calls)})
	response, err := client.Get("http://model.private.example:8080/")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := p.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := io.ReadAll(response.Body); err == nil {
		t.Fatal("quiet response completed successfully after shutdown")
	}
	select {
	case <-stopped:
	case <-time.After(time.Second):
		t.Fatal("upstream request not canceled")
	}
}

func TestRunEOFAndShutdownTimeout(t *testing.T) {
	for _, blocked := range []bool{false, true} {
		t.Run(fmt.Sprintf("blocked=%v", blocked), func(t *testing.T) {
			c := testConfig(t)
			c.ShutdownTimeoutSec = .03
			input, writer := io.Pipe()
			defer input.Close()
			defer writer.Close()
			output := &synchronizedBuffer{notify: make(chan struct{}, 1)}
			unblock, closed := make(chan struct{}), make(chan struct{})
			result := make(chan int, 1)
			go func() {
				result <- run(context.Background(), input, output, func(config) backend {
					return &fakeBackend{close: func() error {
						if blocked {
							<-unblock
						}
						close(closed)
						return nil
					}}
				})
			}()
			fmt.Fprintf(writer, "%s\n", encoded(c))
			waitOutput(t, output, `"status":"ready"`)
			writer.Close()
			code := <-result
			if blocked {
				if code != 2 || !strings.Contains(output.String(), "shutdown_timeout") {
					t.Fatalf("shutdown bound: %d %s", code, output.String())
				}
				if _, err := acquireState(c); err == nil {
					t.Fatal("state unlocked while backend Close still active")
				}
				close(unblock)
			} else if code != 0 || !strings.Contains(output.String(), `"status":"closed"`) {
				t.Fatalf("EOF: %d %s", code, output.String())
			}
			select {
			case <-closed:
			case <-time.After(time.Second):
				t.Fatal("cleanup did not finish")
			}
		})
	}
}

func TestRunRejectsCommandCaseAliasesAndAliasedDuplicates(t *testing.T) {
	for _, command := range []string{`{"protocol":1,"Command":"shutdown"}`, `{"protocol":1,"command":"shutdown","Command":"shutdown"}`} {
		t.Run(command, func(t *testing.T) {
			c := testConfig(t)
			input, writer := io.Pipe()
			defer input.Close()
			defer writer.Close()
			output := &synchronizedBuffer{notify: make(chan struct{}, 1)}
			result := make(chan int, 1)
			go func() {
				result <- run(context.Background(), input, output, func(config) backend { return &fakeBackend{} })
			}()
			fmt.Fprintf(writer, "%s\n", encoded(c))
			waitOutput(t, output, `"status":"ready"`)
			fmt.Fprintf(writer, "%s\n", command)
			if code := <-result; code != 2 || !strings.Contains(output.String(), `"code":"invalid_command"`) {
				t.Fatalf("ambiguous command accepted: %d %s", code, output.String())
			}
		})
	}
}
