# Contributor entry point

- Read the README and relevant technical documents before changing contracts.
- Keep the Python distribution `runtime-llama` and import `runtime_llama`
  compatible unless an intentional migration is requested.
- The server operator owns model loading, processes, builds and hardware.
  Keep runtime configuration and credentials outside the repository.
- Preserve copyright and upstream attribution in LICENSE, NOTICE and the
  vendored MIT license.
- For source or package changes run `python -m pytest -q`,
  `ruff check src tests probes`, `ruff format --check src tests probes` and
  `python -m build --no-isolation`. HTTP loopback tests require socket access.
- When runtime contents or packaging change, install-smoke the built wheel
  in a clean environment outside the source tree.
- Keep deterministic verification, live server qualification and application
  acceptance distinct. Live probes require explicit opt-in.
