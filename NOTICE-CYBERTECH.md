# The Cybertech Engine - notice

The Cybertech Engine is a rebranded build of **Ojas Engine**
(https://github.com/Ajna-inc/ojas-engine), Copyright (c) 2026 Ajna Inc,
used under the MIT licence (see LICENSE). Ajna Inc is the author of the engine.

## What was changed by The Cybertech
- The command-line program is named `cybertech-engine` (was `ojas`).
- Usage text, README and the `owned_by` field of `/v1/models` use the Cybertech name.
- Default binary paths in `scripts/` point to `cybertech-engine`.
- Added a model manager (`crates/ojas-cli/src/pull.rs`): `pull`, `list`, `rm`, and use of pulled
  model names in `run`, `chat`, `serve`, `bench`, `tokenize` and `info`. Downloads use the system `curl`.
- Added `START_CYBERTECH_ENGINE.bat`, `QUICKSTART.txt` and `.github/workflows/build-windows.yml`
  (builds the Windows .exe on GitHub).
- Updated the `systemone` integration test for the new binary name.

## What was NOT changed
- Engine code, kernels and crate names (`ojas-*`), and `OJAS_*` environment variables.
- `assets/ojas.png` is Ajna Inc's logo and is not used as Cybertech branding.
  Replace it with your own logo if you want one.

## Licences you must keep when you ship this
- `LICENSE` (MIT, Ajna Inc + The Cybertech)
- `third_party/` (ggml and PyTorch notices)
- Model files have their own licences (see `models/MANIFEST.md`). Some listed
  vision weights are AGPL-3.0 - do not bundle those in a commercial product.
