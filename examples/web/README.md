# rfgboost in the browser (WASM / Pyodide)

A self-contained page that trains `rfgboost` entirely in the browser — no server,
no install. The Rust/PyO3 extension is compiled to WebAssembly and run via
[Pyodide](https://pyodide.org/); categorical features are WOE-encoded by
`fastwoe-rs`, also compiled to WASM.

## Run it locally

The wheels are loaded via relative URLs, so open the page through a web server
(not `file://`):

```bash
cd examples/web
python -m http.server 8000
# open http://localhost:8000
```

Click **Train rfgboost**. Toggle the categorical checkbox to exercise the
`fastwoe-rs` WOE path.

## Deploy (GitHub Pages)

Serve `examples/web/` as a static site — everything it needs is in this folder:

- `index.html` — the demo
- `wheels/rfgboost-0.1.1-cp312-cp312-pyodide_2024_0_wasm32.whl`
- `wheels/fastwoe_rs-0.1.11-cp39-abi3-pyodide_2024_0_wasm32.whl`

## Version pinning (important)

The wheels are built for **CPython 3.12 / emscripten 3.1.58**, so the page pins
**Pyodide 0.27.7**. Two ABI notes:

- Pyodide 0.27's `micropip` does not yet accept the PEP 783 `pyemscripten` tag,
  so the bundled wheels use the equivalent `pyodide_2024_0_wasm32` tag (same
  bytes). On Pyodide 0.28+ the `pyemscripten` tag should work directly.
- `deps=False` is used because numpy / scikit-learn ship with Pyodide and
  `fastwoe-rs` is installed explicitly.

## Rebuilding the wheels

Built with the pinned toolchain (Rust `nightly-2025-02-01`, emscripten `3.1.58`,
`pyodide-build`) via `pyodide build`. See the repository notes for the full recipe.
