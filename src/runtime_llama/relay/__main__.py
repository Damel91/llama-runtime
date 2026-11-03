"""Support ``python -m runtime_llama.relay`` without importing web on help."""

from runtime_llama.relay.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
