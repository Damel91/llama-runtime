#!/usr/bin/env python3
"""Collect offline notices for the delivered native helper target union.

Run after building every artifact with ``ts_omit_webclient``. The collector
reads build metadata, obtains the matching package graphs without network
access, and preserves license/notice files at module and package ancestors.
It never launches the helper or authenticates a VPN node. Generated output
contains upstream-relative paths, versions and hashes, never cache paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

_TAG = "ts_omit_webclient"
_FORBIDDEN = "github.com/tailscale/web-client-prebuilt"
_NOTICE_NAME = re.compile(
    r"(?:licen[cs]e|copying|notice|copyright|authors|patents)(?:$|[._-])", re.I
)
_SOURCE_FIELDS = ("GoFiles", "CgoFiles", "CFiles", "CXXFiles", "HFiles", "SFiles")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(go: str, arguments: list[str], cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        [go, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise ValueError("offline Go metadata/package inspection failed")
    return result.stdout


def json_objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    result = []
    offset = 0
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        value, offset = decoder.raw_decode(text, offset)
        if not isinstance(value, dict) or value.get("Error"):
            raise ValueError("incomplete offline package graph")
        result.append(value)
    return result


def binary_metadata(text: str) -> tuple[str, dict[str, str], dict[str, str]]:
    lines = text.splitlines()
    if not lines:
        raise ValueError("missing artifact build metadata")
    toolchain = lines[0].rsplit(": ", 1)[-1]
    modules, settings = {}, {}
    for line in lines[1:]:
        fields = line.split()
        if fields and fields[0] == "dep":
            modules[fields[1]] = fields[2]
        if fields and fields[0] == "build":
            key, _, value = fields[1].partition("=")
            if key in {"GOOS", "GOARCH", "CGO_ENABLED", "-tags", "-trimpath"}:
                settings[key] = value
    if _TAG not in settings.get("-tags", "").split(","):
        raise ValueError("artifact must be built with ts_omit_webclient")
    if _FORBIDDEN in modules:
        raise ValueError("artifact still links web UI assets")
    return toolchain, modules, settings


def copyright_lines(path: Path) -> set[str]:
    """Preserve source copyright/SPDX attribution, excluding executable code."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = set()
    for line in text.splitlines()[:100]:
        if re.match(r"\s*package\s+", line):
            break
        if re.search(r"copyright|SPDX-License-Identifier", line, re.I):
            lines.add(line.strip().lstrip("/*# ").rstrip("*/ "))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go", required=True, help="Go toolchain executable")
    parser.add_argument("--gomodcache", required=True, type=Path)
    parser.add_argument("--gocache", required=True, type=Path)
    parser.add_argument("--gopath", required=True, type=Path)
    parser.add_argument(
        "--artifact", action="append", required=True, metavar="GOOS/GOARCH=PATH"
    )
    args = parser.parse_args()
    module_root = Path(__file__).resolve().parent
    output = module_root / "third_party"
    env = dict(os.environ)
    env.update(
        GOPROXY="off",
        GOSUMDB="off",
        GOTOOLCHAIN="local",
        GOENV="off",
        GOFLAGS="",
        GOMODCACHE=str(args.gomodcache.resolve()),
        GOCACHE=str(args.gocache.resolve()),
        GOPATH=str(args.gopath.resolve()),
    )
    goroot = Path(run(args.go, ["env", "GOROOT"], module_root, env).strip())
    artifacts, records, copied, attribution = [], {}, {}, {}
    copied["licenses/runtime-llama-helper/LICENSE"] = (
        module_root.parents[1] / "LICENSE"
    ).read_bytes()
    targets = set()
    for specification in args.artifact:
        target, separator, filename = specification.partition("=")
        if not separator or not re.fullmatch(r"[a-z0-9]+/[a-z0-9]+", target):
            raise ValueError("artifact must use GOOS/GOARCH=PATH")
        if target in targets:
            raise ValueError("duplicate artifact target")
        targets.add(target)
        artifact = Path(filename)
        binary = artifact.read_bytes()
        toolchain, modules, settings = binary_metadata(
            run(args.go, ["version", "-m", str(artifact.resolve())], module_root, env)
        )
        goos, goarch = target.split("/")
        if settings.get("GOOS") != goos or settings.get("GOARCH") != goarch:
            raise ValueError("artifact target disagrees with binary metadata")
        # Verify exclusion by content as well as module metadata. Runtime-only
        # RunWebClient=false does not prevent embedding these upstream assets.
        for base in args.gomodcache.glob(f"{_FORBIDDEN}@*/build"):
            for asset in base.rglob("*"):
                if asset.is_file() and asset.read_bytes() in binary:
                    raise ValueError("artifact still contains embedded web assets")
        target_env = dict(env, GOOS=goos, GOARCH=goarch)
        target_env["CGO_ENABLED"] = settings.get("CGO_ENABLED", "0")
        packages = json_objects(
            run(
                args.go,
                [
                    "list",
                    "-mod=readonly",
                    "-deps",
                    "-json",
                    "-tags=" + _TAG,
                    "./cmd/runtime-llama-tsnet",
                ],
                module_root,
                target_env,
            )
        )
        seen_modules = set()
        for package in packages:
            module = package.get("Module")
            if module and module.get("Main"):
                continue
            if module:
                name, version = module["Path"], module["Version"]
                if name not in modules:
                    continue  # compiler/linker excluded this module from binary
                if modules[name] != version or module.get("Replace"):
                    raise ValueError("artifact and package module identities disagree")
                base = Path(module["Dir"])
                seen_modules.add(name)
            elif package.get("Standard"):
                name, version, base = "go-stdlib", toolchain, goroot
            else:
                raise ValueError("unidentified package in offline dependency graph")
            key = name + "@" + version
            entry = records.setdefault(
                key,
                {
                    "module": name,
                    "version": version,
                    "targets": set(),
                    "packages": set(),
                    "files": set(),
                },
            )
            entry["targets"].add(target)
            entry["packages"].add(package["ImportPath"])
            directory = Path(package["Dir"])
            current = directory
            while True:
                current.relative_to(base)
                for source in current.iterdir():
                    if (
                        source.is_file()
                        and source.suffix != ".go"
                        and _NOTICE_NAME.match(source.name)
                    ):
                        relative = source.relative_to(base).as_posix()
                        destination = "licenses/" + key + "/" + relative
                        data = source.read_bytes()
                        if destination in copied and copied[destination] != data:
                            raise ValueError("conflicting upstream notice content")
                        copied[destination] = data
                        entry["files"].add(destination)
                if current == base:
                    break
                current = current.parent
            headers = attribution.setdefault(key, set())
            for field in _SOURCE_FIELDS:
                for filename in package.get(field, []):
                    headers.update(copyright_lines(directory / filename))
        if seen_modules != set(modules):
            raise ValueError("artifact module not covered by offline package graph")
        artifacts.append(
            {
                "target": target,
                "filename": artifact.name,
                "sha256": digest(binary),
                "go_version": toolchain,
                "build_settings": settings,
                "web_assets_excluded": True,
            }
        )
    required = {"darwin/arm64", "linux/amd64", "windows/amd64"}
    if not required.issubset(targets):
        raise ValueError("collect the delivered Darwin/Linux/Windows target union")
    for key, entry in records.items():
        if not entry["files"]:
            raise ValueError("dependency has no preserved license/notice: " + key)
    copied["ATTRIBUTIONS.txt"] = (
        "Native helper source copyright/SPDX attribution inventory.\n"
        "Full license and notice texts are preserved under licenses/.\n\n"
        + "\n\n".join(
            key + "\n" + "\n".join(sorted(lines))
            for key, lines in sorted(attribution.items())
        )
        + "\n"
    ).encode("utf-8")
    manifest = {
        "schema": 1,
        "generator": "helpers/tsnet/collect_licenses.py",
        "method": "binary module metadata plus compiled-package ancestor notices",
        "build_tags": [_TAG],
        "project_license": "licenses/runtime-llama-helper/LICENSE",
        "go_mod_sha256": digest((module_root / "go.mod").read_bytes()),
        "go_sum_sha256": digest((module_root / "go.sum").read_bytes()),
        "artifacts": sorted(artifacts, key=lambda item: item["target"]),
        "modules": [
            dict(
                entry,
                targets=sorted(entry["targets"]),
                packages=sorted(entry["packages"]),
                files=sorted(entry["files"]),
            )
            for _, entry in sorted(records.items())
        ],
        "files": [
            {"path": name, "sha256": digest(data)}
            for name, data in sorted(copied.items())
        ],
    }
    output.mkdir(exist_ok=True)
    license_root = output / "licenses"
    if license_root.exists():
        shutil.rmtree(license_root)
    for name, data in sorted(copied.items()):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Collected {len(records)} modules, {len(copied)} notice files, "
        f"{len(artifacts)} artifacts; no web assets linked."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError):
        print(
            "Offline native notice collection failed.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
