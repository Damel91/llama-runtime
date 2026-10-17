package main

import (
	"context"
	"errors"
	"net"
	"net/netip"
	"strconv"
	"strings"

	"tailscale.com/ipn/ipnstate"
	"tailscale.com/logtail"
	"tailscale.com/tsnet"
)

type backend interface {
	Up(context.Context) error
	Dial(context.Context, string) (net.Conn, error)
	Close() error
}

type tsBackend struct {
	server *tsnet.Server
	status func(context.Context) (*ipnstate.Status, error)
	direct func(context.Context, netip.AddrPort) (net.Conn, error)
}

func newTSBackend(c config) backend {
	logtail.Disable()
	discard := func(string, ...any) {}
	return &tsBackend{server: &tsnet.Server{
		Dir: c.StateDir, Hostname: c.Hostname, AuthKey: c.AuthKey,
		ControlURL: c.ControlURL, Ephemeral: c.Ephemeral,
		Logf: discard, UserLogf: discard,
	}}
}

func (b *tsBackend) Up(ctx context.Context) error {
	if _, err := b.server.Up(ctx); err != nil {
		return errors.New("join_failed")
	}
	lc, err := b.server.LocalClient()
	if err != nil {
		return errors.New("backend_unavailable")
	}
	// PINNED SHIM: Server.Sys is explicitly unstable upstream. In v1.104.1,
	// NetstackDialTCP directly calls gVisor DialContextTCPWithBind. UserDial
	// must not be used: it can resolve/dial unknown peers through the OS.
	sys := b.server.Sys()
	if sys == nil {
		return errors.New("backend_unavailable")
	}
	dialer, ok := sys.Dialer.GetOK()
	if !ok || dialer == nil || dialer.NetstackDialTCP == nil {
		return errors.New("backend_unavailable")
	}
	b.status, b.direct = lc.Status, dialer.NetstackDialTCP
	return nil
}

func parsePort(text string) (uint16, error) {
	port, err := strconv.ParseUint(text, 10, 16)
	if err != nil || port == 0 {
		return 0, errors.New("invalid_target")
	}
	return uint16(port), nil
}

func canonicalDNS(name string) string {
	return strings.ToLower(strings.TrimSuffix(name, "."))
}

func resolvePeer(status *ipnstate.Status, address string) (netip.AddrPort, error) {
	host, portText, err := net.SplitHostPort(address)
	if err != nil || status == nil {
		return netip.AddrPort{}, errors.New("unknown_peer")
	}
	port, err := parsePort(portText)
	if err != nil {
		return netip.AddrPort{}, err
	}
	requestedIP, ipErr := netip.ParseAddr(host)
	name := canonicalDNS(host)
	var matched *ipnstate.PeerStatus
	for _, peer := range status.Peer {
		if peer == nil {
			continue
		}
		matches := false
		if ipErr == nil {
			for _, ip := range peer.TailscaleIPs {
				if ip == requestedIP {
					matches = true
				}
			}
		} else if peer.DNSName != "" && name == canonicalDNS(peer.DNSName) {
			matches = true
		}
		if matches {
			if matched != nil {
				return netip.AddrPort{}, errors.New("ambiguous_peer")
			}
			matched = peer
		}
	}
	if matched == nil {
		return netip.AddrPort{}, errors.New("unknown_peer")
	}
	if ipErr == nil {
		if !validPeerIP(requestedIP) {
			return netip.AddrPort{}, errors.New("unknown_peer")
		}
		return netip.AddrPortFrom(requestedIP, port), nil
	}
	var selected netip.Addr
	for _, ip := range matched.TailscaleIPs {
		if validPeerIP(ip) && (!selected.IsValid() || ip.Less(selected)) {
			selected = ip
		}
	}
	if !selected.IsValid() {
		return netip.AddrPort{}, errors.New("unknown_peer")
	}
	return netip.AddrPortFrom(selected, port), nil
}

func validPeerIP(ip netip.Addr) bool {
	return ip.IsValid() && ip.Zone() == "" && !ip.Is4In6() &&
		!ip.IsUnspecified() && !ip.IsLoopback() && !ip.IsMulticast() && !ip.IsLinkLocalUnicast()
}

func (b *tsBackend) Dial(ctx context.Context, address string) (net.Conn, error) {
	if b.status == nil || b.direct == nil {
		return nil, errors.New("backend_unavailable")
	}
	status, err := b.status(ctx)
	if err != nil {
		return nil, errors.New("peer_status_unavailable")
	}
	target, err := resolvePeer(status, address)
	if err != nil {
		return nil, err
	}
	// Numeric destination and direct netstack callback: peer-map changes can
	// cause a VPN dial failure but can never select an OS dial/resolver fallback.
	conn, err := b.direct(ctx, target)
	if err != nil {
		return nil, errors.New("peer_dial_failed")
	}
	return conn, nil
}

func (b *tsBackend) Close() error { return b.server.Close() }
