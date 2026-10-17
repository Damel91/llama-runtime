#!/bin/sh
set -eu

# Go downloads the pinned toolchain/dependencies when absent. Callers may set
# GOCACHE, GOMODCACHE and GOPATH to isolated locations; no system Go is changed.
# GOOS/GOARCH select a target. Ship TAILSCALE-LICENSE.txt with any binary.
if [ "$#" -ne 1 ]; then
  printf '%s\n' 'Usage: build.sh /absolute/path/to/runtime-llama-tsnet[.exe]' >&2
  exit 2
fi
case "$1" in
  /*|[A-Za-z]:[\\/]*) ;;
  *) printf '%s\n' 'Output path must be absolute.' >&2; exit 2 ;;
esac
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd -- "$script_dir"
exec go build -buildvcs=false -tags ts_omit_webclient -trimpath -o "$1" ./cmd/runtime-llama-tsnet
