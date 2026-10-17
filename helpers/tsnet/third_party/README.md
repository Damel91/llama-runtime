# Native helper licenses and notices

Ship this directory, the repository `NOTICE`, and `TAILSCALE-LICENSE.txt` with
every separately delivered native helper binary. The Python wheel has its own
license files; it does not contain the helper executable.

`MANIFEST.json` records the exact delivered binary hashes, target union, pinned
module versions, included package paths, and SHA-256 of every copied notice.
`licenses/` preserves original license, notice, author, copyright, and patent
files found at dependency module roots and compiled-package ancestors.
`ATTRIBUTIONS.txt` additionally preserves source copyright and SPDX lines.
The helper's Apache-2.0 license and Go runtime/standard-library notices are
included. Third-party files retain their upstream licensing terms.

The inventory covers the union of Darwin arm64, Linux amd64, and Windows amd64.
It conservatively includes package-graph notices for modules recorded in each
binary's Go build metadata. It does not claim every compiled package survives
linker elimination. Test-only dependencies are excluded.

All delivered helpers are built with `ts_omit_webclient`. The collector checks
both module metadata and the original cached web asset bytes to verify that
the web UI, its JavaScript dependencies, and its font assets are absent. Merely
setting `RunWebClient=false` does not remove those assets from an untagged binary.
Public builds also use `-trimpath -buildvcs=false` to omit local source paths and
repository revision metadata from the executable.

To regenerate after building all three targets, run the collector with the
matching Go toolchain and already populated offline dependency/cache directories:

```sh
python3 collect_licenses.py \
  --go /path/to/go1.27.1/bin/go \
  --gomodcache /path/to/offline/module-cache \
  --gocache /path/to/offline/build-cache \
  --gopath /path/to/offline/gopath \
  --artifact darwin/arm64=/path/to/runtime-llama-tsnet-darwin-arm64 \
  --artifact linux/amd64=/path/to/runtime-llama-tsnet-linux-amd64 \
  --artifact windows/amd64=/path/to/runtime-llama-tsnet-windows-amd64.exe
```

The collector disables module/proxy downloads, never executes the helper, and
never enrolls a node. Output has no timestamps or absolute cache/user paths.
Regeneration replaces only its generated `licenses/`, manifest and attribution
files, preserving this README.
