"""Explicit opt-in CLI for long conversations and24 actual generation turns."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import qualify_long as long_probe
import qualify_real_turns as real_probe
from runtime_llama import LlamaCppClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="HTTP(S) llama-server root URL"
    )
    parser.add_argument(
        "--model", required=True, help="Exact alias of an already loaded model"
    )
    parser.add_argument(
        "--expected-build", required=True, help="Exact /props build_info required"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output directory; existing paths are refused",
    )
    parser.add_argument(
        "--acknowledge-live",
        required=True,
        action="store_true",
        help="Explicitly authorize synthetic inference; no server administration",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=int,
        choices=[8192, 32768, 65536],
        default=[8192, 32768, 65536],
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory to preserve earlier evidence")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Native catalog is read before model-specific routes to avoid intentional autoload.
    client = LlamaCppClient(args.base_url, timeout=10)
    try:
        catalog = client.request("GET", "/models", timeout=10)
        long_probe.write(output / "preflight-catalog.json", catalog)
        rows = (
            catalog.get("data", catalog.get("models", []))
            if isinstance(catalog, dict)
            else catalog
        )
        if not isinstance(rows, list):
            raise RuntimeError("native_catalog_did_not_return_model_list")
        matching = [
            row
            for row in rows
            if isinstance(row, dict)
            and (row.get("id") == args.model or row.get("name") == args.model)
        ]
        if len(matching) != 1:
            raise RuntimeError("selected_model_not_uniquely_present_in_catalog")
        status = matching[0].get("status")
        loaded = status.get("value") if isinstance(status, dict) else status
        if loaded != "loaded":
            raise RuntimeError(
                "selected_model_not_observed_loaded; load externally before qualifying"
            )
    finally:
        client.close()
    oracle = copy.deepcopy(long_probe.ORACLE)
    oracle.update(
        endpoint=args.base_url.rstrip("/"),
        model=args.model,
        expected_build=args.expected_build,
    )
    long_probe.write(output / "long-conversation-oracle.json", oracle)
    shutil.copy2(
        Path(__file__).parent / "long-real-turns-oracle.json",
        output / "long-real-turns-oracle.json",
    )
    long_probe.ROOT, long_probe.ORACLE = output, oracle
    long_probe.BASE, long_probe.MODEL = oracle["endpoint"], args.model
    real_probe.ROOT = output / "real-turns"
    real_probe.ORACLE_PATH = output / "long-real-turns-oracle.json"
    real_probe.ORACLE = json.loads(real_probe.ORACLE_PATH.read_text())
    original_argv = sys.argv
    try:
        sys.argv = [
            str(Path(__file__)),
            "--execute-live",
            "--targets",
            *map(str, args.targets),
        ]
        long_probe.main()
        long_summary = json.loads((output / "summary.json").read_text())
        if long_summary.get("stopped") or long_summary.get(
            "harness_or_preflight_error"
        ):
            print(
                "Long qualification stopped;24-turn inference was not started.",
                flush=True,
            )
            return 1
        sys.argv = [str(Path(__file__)), "--execute-live"]
        real_probe.main()
        real_summary = json.loads((output / "real-turns" / "summary.json").read_text())
        long_cases = long_summary.get("cases", [])
        long_passed = len(long_cases) == len(args.targets) * 2 and all(
            case.get("result", {}).get("library_status") == "PASS"
            and case.get("result", {}).get("model_status") == "PASS"
            for case in long_cases
        )
        real_passed = bool(
            real_summary.get("library_passed") and real_summary.get("model_passed")
        )
        long_probe.write(
            output / "campaign-summary.json",
            {
                "long_cases_passed": long_passed,
                "expected_long_cases": len(args.targets) * 2,
                "observed_long_cases": len(long_cases),
                "real_24_turns_passed": real_passed,
                "overall_passed": long_passed and real_passed,
            },
        )
        return 0 if long_passed and real_passed else 1
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
