package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/url"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"time"
)

const protocolVersion = 1
const backendVersion = "1.104.1"
const maxControlLine = 64 * 1024
const officialControlURL = "https://controlplane.tailscale.com"

var hostnamePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9-]{0,62}$`)
var tokenPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{32,256}$`)

type config struct {
	Protocol           int     `json:"protocol"`
	StateDir           string  `json:"state_dir"`
	Hostname           string  `json:"hostname"`
	AuthKey            string  `json:"auth_key,omitempty"`
	ControlURL         string  `json:"control_url,omitempty"`
	Ephemeral          bool    `json:"ephemeral"`
	ProxyToken         string  `json:"proxy_token"`
	StartupTimeoutSec  float64 `json:"startup_timeout_sec"`
	DialTimeoutSec     float64 `json:"dial_timeout_sec"`
	ShutdownTimeoutSec float64 `json:"shutdown_timeout_sec"`
}

func decodeStrict(line []byte, target any) error {
	if len(line) == 0 || len(line) > maxControlLine {
		return errors.New("invalid_config")
	}
	targetType := reflect.TypeOf(target)
	if targetType == nil || targetType.Kind() != reflect.Pointer || targetType.Elem().Kind() != reflect.Struct {
		return errors.New("invalid_config")
	}
	allowed := make(map[string]bool)
	for _, field := range reflect.VisibleFields(targetType.Elem()) {
		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		if field.PkgPath == "" && name != "" && name != "-" {
			allowed[name] = true
		}
	}
	// encoding/json normally accepts duplicate keys and null primitive fields.
	// The control protocol permits one unambiguous, non-null object only.
	shape := json.NewDecoder(bytes.NewReader(line))
	opening, err := shape.Token()
	if err != nil || opening != json.Delim('{') {
		return errors.New("invalid_config")
	}
	seen := make(map[string]bool)
	for shape.More() {
		key, err := shape.Token()
		name, ok := key.(string)
		if err != nil || !ok || !allowed[name] || seen[name] {
			return errors.New("invalid_config")
		}
		seen[name] = true
		var value json.RawMessage
		if shape.Decode(&value) != nil || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return errors.New("invalid_config")
		}
	}
	if closing, err := shape.Token(); err != nil || closing != json.Delim('}') {
		return errors.New("invalid_config")
	}
	if shape.Decode(new(any)) != io.EOF {
		return errors.New("invalid_config")
	}
	decoder := json.NewDecoder(bytes.NewReader(line))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return errors.New("invalid_config")
	}
	if decoder.Decode(new(any)) != io.EOF {
		return errors.New("invalid_config")
	}
	return nil
}

func finiteDuration(seconds float64) bool {
	nanos := seconds * float64(time.Second)
	return !math.IsNaN(seconds) && !math.IsInf(seconds, 0) && seconds <= 86400 && nanos >= 1 && nanos < float64(math.MaxInt64)
}

func secondsDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}

func parseConfig(line []byte) (config, error) {
	var c config
	if err := decodeStrict(line, &c); err != nil {
		return c, err
	}
	if c.Protocol != protocolVersion || !filepath.IsAbs(c.StateDir) || filepath.Clean(c.StateDir) != c.StateDir ||
		!hostnamePattern.MatchString(c.Hostname) || strings.HasSuffix(c.Hostname, "-") ||
		!tokenPattern.MatchString(c.ProxyToken) || !finiteDuration(c.StartupTimeoutSec) ||
		!finiteDuration(c.DialTimeoutSec) || !finiteDuration(c.ShutdownTimeoutSec) ||
		len(c.AuthKey) > 8192 || strings.ContainsAny(c.AuthKey, "\r\n\x00") {
		return c, errors.New("invalid_config")
	}
	if c.ControlURL == "" {
		c.ControlURL = officialControlURL
	}
	u, err := url.Parse(c.ControlURL)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" ||
		(u.Path != "" && u.Path != "/") || strings.ContainsAny(c.ControlURL, "\\ \r\n\t\x00") {
		return c, errors.New("invalid_config")
	}
	if u.Port() != "" {
		if _, err := parsePort(u.Port()); err != nil {
			return c, errors.New("invalid_config")
		}
	}
	c.ControlURL = strings.TrimSuffix(c.ControlURL, "/")
	c.Hostname = strings.ToLower(c.Hostname)
	return c, nil
}
