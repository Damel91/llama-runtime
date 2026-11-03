"""Headless entry point for the optional local HTTP relay."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import sys
from threading import current_thread, main_thread

from runtime_llama import __version__


def default_config_path() -> Path:
    """Select external per-user storage without creating a file or directory."""
    if sys.platform == "win32":
        directory = os.environ.get("APPDATA")
        fallback = Path.home() / "AppData" / "Roaming"
    else:
        directory = os.environ.get("XDG_CONFIG_HOME")
        fallback = Path.home() / ".config"
    if directory and Path(directory).is_absolute():
        fallback = Path(directory)
    return fallback / "llama-runtime" / "relay.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llama-runtime-relay",
        description="Run the local inference relay and its browser configuration UI.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="use this configuration file instead of the per-user default",
    )
    parser.add_argument("--host", help="override the listening host for this run")
    parser.add_argument(
        "--port", type=int, help="override the listening port for this run"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def _load_web():
    import uvicorn

    from runtime_llama.relay.app import create_app

    return uvicorn, create_app


def _run_server(uvicorn, app, listener):
    class RelayServer(uvicorn.Server):
        def _drain_then_exit(self, sig, frame):
            try:
                # Uvicorn waits for active requests before lifespan shutdown.
                # Quiet SDK streams need cancellation before that wait begins.
                app.state.relay_cancel_all()
            finally:
                super().handle_exit(sig, frame)

        def handle_exit(self, sig, frame):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._drain_then_exit(sig, frame)
            else:
                # Signals may interrupt a callback already holding a request
                # lock. Perform cancellation in the loop, outside that stack.
                loop.call_soon_threadsafe(self._drain_then_exit, sig, None)

        @contextmanager
        def capture_signals(self):
            if current_thread() is not main_thread():
                yield
                return
            previous = {
                sig: signal.signal(sig, self.handle_exit)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                yield
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            # The relay owns these CLI signals. Do not replay SIGTERM after
            # successful drain: that would replace normal exit with a kill.

    server = RelayServer(
        uvicorn.Config(
            app,
            host=listener["host"],
            port=listener["port"],
            loop="asyncio",
            lifespan="on",
            ws="none",
            access_log=False,
            proxy_headers=False,
        )
    )
    server.run()


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        uvicorn, create_app = _load_web()
    except ImportError:
        print(
            "The relay requires optional dependencies. Install with: pip install 'runtime-llama[relay]'",
            file=sys.stderr,
        )
        return 2

    from runtime_llama.relay.config import (
        RelayConfigError,
        RelayConfigStore,
        validate_config,
    )
    from runtime_llama.relay.service import RelayService

    try:
        store = RelayConfigStore(options.config or default_config_path())
        listener = store.snapshot().config["listener"]
        if options.host is not None:
            listener["host"] = options.host
        if options.port is not None:
            listener["port"] = options.port
        listener = validate_config({"listener": listener})["listener"]
    except RelayConfigError as error:
        print(f"Invalid relay configuration ({error}).", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("Cannot read the relay configuration file.", file=sys.stderr)
        return 2

    app = create_app(service=RelayService(store, listener=listener))
    try:
        _run_server(uvicorn, app, listener)
    except KeyboardInterrupt:
        return 0
    return 0
