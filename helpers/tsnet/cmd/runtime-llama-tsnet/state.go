package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

const stateMarker = "runtime-helper-identity.json"
const stateLock = "runtime-helper.lock"

type identity struct {
	Protocol   int    `json:"protocol"`
	ControlURL string `json:"control_url"`
	Hostname   string `json:"hostname"`
	Ephemeral  bool   `json:"ephemeral"`
}

type stateLease struct {
	file *os.File
}

func rejectSymlinkComponents(path string) error {
	for current := path; ; current = filepath.Dir(current) {
		info, err := os.Lstat(current)
		if err == nil && (info.Mode()&os.ModeSymlink != 0 || !info.IsDir()) {
			return errors.New("invalid_state")
		}
		if err != nil && !os.IsNotExist(err) {
			return errors.New("invalid_state")
		}
		if parent := filepath.Dir(current); parent == current {
			return nil
		}
	}
}

func privateFile(path string, flag int) (*os.File, error) {
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() || checkPrivate(path, info, false) != nil {
			return nil, errors.New("invalid_state")
		}
	} else if !os.IsNotExist(err) {
		return nil, errors.New("invalid_state")
	}
	f, err := os.OpenFile(path, flag, 0600)
	if err != nil {
		return nil, errors.New("invalid_state")
	}
	info, statErr := f.Stat()
	pathInfo, pathErr := os.Lstat(path)
	if statErr != nil || pathErr != nil || !info.Mode().IsRegular() || !os.SameFile(info, pathInfo) || checkPrivate(path, info, false) != nil {
		f.Close()
		return nil, errors.New("invalid_state")
	}
	return f, nil
}

func acquireState(c config) (*stateLease, error) {
	if err := rejectSymlinkComponents(c.StateDir); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(c.StateDir, 0700); err != nil {
		return nil, errors.New("invalid_state")
	}
	info, err := os.Lstat(c.StateDir)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 || checkPrivate(c.StateDir, info, true) != nil {
		return nil, errors.New("invalid_state")
	}
	f, err := privateFile(filepath.Join(c.StateDir, stateLock), os.O_CREATE|os.O_RDWR)
	if err != nil {
		return nil, err
	}
	if err := lockFile(f); err != nil {
		f.Close()
		return nil, errors.New("state_in_use")
	}
	lease := &stateLease{file: f}
	ok := false
	defer func() {
		if !ok {
			lease.Close()
		}
	}()
	markerPath := filepath.Join(c.StateDir, stateMarker)
	expected := identity{protocolVersion, c.ControlURL, c.Hostname, c.Ephemeral}
	if _, err := os.Lstat(markerPath); os.IsNotExist(err) {
		entries, readErr := os.ReadDir(c.StateDir)
		if readErr != nil {
			return nil, errors.New("invalid_state")
		}
		for _, entry := range entries {
			if entry.Name() != stateLock {
				return nil, errors.New("foreign_state")
			}
		}
		marker, err := privateFile(markerPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY)
		if err != nil {
			return nil, err
		}
		writeErr := json.NewEncoder(marker).Encode(expected)
		syncErr := marker.Sync()
		closeErr := marker.Close()
		if writeErr != nil || syncErr != nil || closeErr != nil {
			return nil, errors.New("invalid_state")
		}
	} else if err != nil {
		return nil, errors.New("invalid_state")
	} else {
		marker, err := privateFile(markerPath, os.O_RDONLY)
		if err != nil {
			return nil, err
		}
		data := make([]byte, maxControlLine+1)
		n, readErr := marker.Read(data)
		marker.Close()
		var found identity
		if readErr != nil || n > maxControlLine || decodeStrict(data[:n], &found) != nil || found != expected {
			return nil, errors.New("state_profile_mismatch")
		}
	}
	ok = true
	return lease, nil
}

func (s *stateLease) Close() error {
	if s == nil || s.file == nil {
		return nil
	}
	f := s.file
	s.file = nil
	unlockFile(f)
	return f.Close()
}
