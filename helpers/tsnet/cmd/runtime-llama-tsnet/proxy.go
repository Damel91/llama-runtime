package main

import (
	"context"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

type proxy struct {
	backend     backend
	config      config
	server      *http.Server
	listener    net.Listener
	transport   *http.Transport
	ctx         context.Context
	cancel      context.CancelFunc
	mu          sync.Mutex
	closed      bool
	connections map[net.Conn]struct{}
	handlers    sync.WaitGroup
}

type managedConn struct {
	net.Conn
	onClose func()
	once    sync.Once
}

func (c *managedConn) Close() error {
	var err error
	c.once.Do(func() { err = c.Conn.Close(); c.onClose() })
	return err
}

func newProxy(parent context.Context, c config, b backend) (*proxy, error) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return nil, errors.New("proxy_start_failed")
	}
	ctx, cancel := context.WithCancel(parent)
	p := &proxy{backend: b, config: c, listener: ln, ctx: ctx, cancel: cancel, connections: make(map[net.Conn]struct{})}
	p.transport = &http.Transport{
		Proxy: nil, DialContext: p.dial, DisableKeepAlives: true, DisableCompression: true,
		ForceAttemptHTTP2: false, MaxResponseHeaderBytes: 64 * 1024,
	}
	p.server = &http.Server{
		Handler: p, ReadHeaderTimeout: 5 * time.Second, MaxHeaderBytes: 64 * 1024,
		ErrorLog:    log.New(io.Discard, "", 0),
		BaseContext: func(net.Listener) context.Context { return ctx },
		ConnState: func(conn net.Conn, state http.ConnState) {
			p.mu.Lock()
			closeNow := state == http.StateNew && p.closed
			if state == http.StateNew && !closeNow {
				p.connections[conn] = struct{}{}
			}
			if state == http.StateClosed {
				delete(p.connections, conn)
			}
			p.mu.Unlock()
			if closeNow {
				conn.Close()
			}
		},
	}
	return p, nil
}

func (p *proxy) URL() string { return "http://" + p.listener.Addr().String() }

func (p *proxy) Serve() error { return p.server.Serve(p.listener) }

func (p *proxy) dial(ctx context.Context, network, address string) (net.Conn, error) {
	if network != "tcp" && network != "tcp4" && network != "tcp6" {
		return nil, errors.New("invalid_target")
	}
	dialCtx, cancel := context.WithTimeout(ctx, secondsDuration(p.config.DialTimeoutSec))
	defer cancel()
	conn, err := p.backend.Dial(dialCtx, address)
	if err != nil {
		return nil, errors.New("peer_dial_failed")
	}
	managed := &managedConn{Conn: conn}
	managed.onClose = func() { p.mu.Lock(); delete(p.connections, managed); p.mu.Unlock() }
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		conn.Close()
		return nil, errors.New("proxy_closed")
	}
	p.connections[managed] = struct{}{}
	p.mu.Unlock()
	return managed, nil
}

func proxyError(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	io.WriteString(w, `{"error":"`+code+`"}`)
}

func stripHopHeaders(h http.Header) {
	for _, line := range h.Values("Connection") {
		for _, name := range strings.Split(line, ",") {
			h.Del(strings.TrimSpace(name))
		}
	}
	for _, name := range []string{"Connection", "Proxy-Connection", "Proxy-Authorization", "Proxy-Authenticate", "Keep-Alive", "TE", "Trailer", "Transfer-Encoding", "Upgrade"} {
		h.Del(name)
	}
}

func (p *proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		proxyError(w, 503, "proxy_closed")
		return
	}
	p.handlers.Add(1)
	p.mu.Unlock()
	defer p.handlers.Done()
	wanted := "Basic " + base64.StdEncoding.EncodeToString([]byte("runtime:"+p.config.ProxyToken))
	if subtle.ConstantTimeCompare([]byte(r.Header.Get("Proxy-Authorization")), []byte(wanted)) != 1 {
		w.Header().Set("Proxy-Authenticate", `Basic realm="runtime"`)
		proxyError(w, 407, "proxy_auth_required")
		return
	}
	if r.Method == http.MethodConnect {
		p.connect(w, r)
		return
	}
	if r.URL.Scheme != "http" || r.URL.Host == "" || r.URL.User != nil {
		proxyError(w, 400, "invalid_target")
		return
	}
	request := r.Clone(r.Context())
	request.RequestURI = ""
	request.Header = r.Header.Clone()
	stripHopHeaders(request.Header)
	// A fresh upstream connection for every plain-HTTP request prevents Go's
	// automatic replay of eligible requests on reused broken connections.
	response, err := p.transport.RoundTrip(request)
	if err != nil {
		proxyError(w, 502, "peer_dial_failed")
		return
	}
	defer response.Body.Close()
	stripHopHeaders(response.Header)
	for name, values := range response.Header {
		for _, value := range values {
			w.Header().Add(name, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	controller := http.NewResponseController(w)
	if err := controller.Flush(); err != nil {
		return
	}
	if _, err = io.Copy(flushingWriter{writer: w, controller: controller}, response.Body); err != nil {
		// Do not turn a failed upstream stream into a clean chunked EOF.
		panic(http.ErrAbortHandler)
	}
}

type flushingWriter struct {
	writer     io.Writer
	controller *http.ResponseController
}

func (w flushingWriter) Write(data []byte) (int, error) {
	n, err := w.writer.Write(data)
	if err == nil {
		err = w.controller.Flush()
	}
	return n, err
}

func (p *proxy) connect(w http.ResponseWriter, r *http.Request) {
	if _, port, err := net.SplitHostPort(r.Host); err != nil || port == "" {
		proxyError(w, 400, "invalid_target")
		return
	}
	upstream, err := p.dial(r.Context(), "tcp", r.Host)
	if err != nil {
		proxyError(w, 502, "peer_dial_failed")
		return
	}
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		upstream.Close()
		proxyError(w, 500, "proxy_unavailable")
		return
	}
	client, buffered, err := hijacker.Hijack()
	if err != nil {
		upstream.Close()
		return
	}
	defer client.Close()
	defer upstream.Close()
	defer func() { p.mu.Lock(); delete(p.connections, client); p.mu.Unlock() }()
	// Hijack inherits read/header deadlines, which must not become an SSE idle
	// limit. Buffered bytes can contain a pipelined TLS handshake: preserve them.
	_ = client.SetDeadline(time.Time{})
	if _, err = buffered.WriteString("HTTP/1.1 200 Connection Established\r\n\r\n"); err != nil {
		return
	}
	if err = buffered.Flush(); err != nil {
		return
	}
	done := make(chan struct{}, 2)
	go func() { _, _ = io.Copy(upstream, buffered.Reader); done <- struct{}{} }()
	go func() { _, _ = io.Copy(client, upstream); done <- struct{}{} }()
	completed := 0
	select {
	case <-done:
		completed = 1
	case <-p.ctx.Done():
	}
	// HTTP-specific tunnel policy: a completed copy means the request transport
	// has ended. Close BOTH ends; generic TCP half-close semantics are not offered.
	client.Close()
	upstream.Close()
	for completed < 2 {
		<-done
		completed++
	}
}

func (p *proxy) Close(ctx context.Context) error {
	p.mu.Lock()
	p.closed = true
	connections := make([]net.Conn, 0, len(p.connections))
	for conn := range p.connections {
		connections = append(connections, conn)
	}
	p.mu.Unlock()
	p.cancel()
	p.listener.Close()
	for _, conn := range connections {
		conn.Close()
	}
	p.transport.CloseIdleConnections()
	err := p.server.Shutdown(ctx)
	done := make(chan struct{})
	go func() { p.handlers.Wait(); close(done) }()
	select {
	case <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}
