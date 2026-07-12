# PUFFIN — Agent Instruction Floor

PUFFIN is a desktop GUI for interactive 1D curve fitting: load a spectrum, build a
composite model from YAML-defined elements, tweak parameters with a live preview on
every change, exclude points, run a SciPy `curve_fit` in a background thread, and
export results. Strict MVVM on PySide6 + pyqtgraph.

## Quick Reference

- **Runtime:** Python 3.12 in `.venv/` (repo declares 3.10+). Not a package — there is
  no root `__init__.py`; imports are top-level (`from models import ...`, `from view...`).
- **Stack:** PySide6, pyqtgraph, numpy, scipy, pandas, PyYAML, matplotlib. Deps in
  `requirements.txt` (pinned lock: `requirements.lock.txt`). **PyYAML is required** —
  model elements load from YAML at runtime.
- **Entry / run:** `python main.py` from the repo root. Do **not** use `python -m PUFFIN.main`
  (no `PUFFIN` package exists — it raises ImportError). Launchers: `run-puffin.bat`
  (existing `.venv`), `run-puffin-dev.bat` (uv-managed venv from the lock file),
  `installer/WINDOWS-install-PUFFIN.bat` (end-user install).
- **Config:** `config/settings.json` at repo root, via `dataio/configuration.py`
  (`get_config()` singleton). Created on first run — runtime-generated local state,
  not source. Never hand-edit while the app runs.
- **Models:** defined as `models/model_elements/*.yaml` (voigt, gaussian, dho,
  triangle_peak, linear_background), loaded by `get_model_spec()` with hardcoded
  fallbacks. Saved custom models live in `models/custom_models/*.yaml`.

## Hard Rules (each names the silent failure it prevents)

1. **MVVM boundary is one-directional.** The View (`view/`) never touches `ModelState`
   or model data directly — everything crosses `viewmodel/fitter_vm.py`, the only
   bridge. `view/input_handler.py` converts pyqtgraph events into viewmodel calls.
   *Silent failure:* a direct View→Model access compiles and runs but breaks the
   signal-driven live preview in non-obvious ways (`plot_updated` / `parameters_updated`
   stop reflecting reality).
2. **Parameter linking: sync before you fit.** Link groups are integers 1–99 (0 or
   `None` = unlinked), stored on each `Parameter`. During a fit only the group
   *representative* is optimized; the other members must be value-synced to it first.
   *Silent failure:* a desynced group crashes `curve_fit` or silently fits stale
   values. When cloning components in composite models, preserve `link_group` **and**
   the parameter-name prefix — a naive deep-copy loses both. See `docs/IMPLEMENTATION_NOTES.md`.
3. **Poisson σ has a 1.0 floor.** Errors are `sqrt(max(counts, 1.0))`
   (`models/model_state.py`, `__init__` and `set_data`). Zero-count points get σ=1.0,
   not 0. This must stay consistent across export → reload. *Silent failure:* dropping
   the floor yields zero uncertainties that blow up χ² weighting.
   **This is the OPPOSITE of sibling repo ISAR's no-floor convention** (shared DHO
   domain — easy to cross-contaminate; do not port ISAR's error handling here).
4. **`archive/` and `Model Test.py` are not live code.** `archive/` (incl.
   `archive/Minis Testing/`) holds abandoned experiments — never import from it.
   `Model Test.py` references `models.dho_voigt_model`, which does not exist; it is
   inspiration, not a runnable module.

## Domain Concepts

- **Model spec registry.** `get_model_spec(name)` in `models/model_specs.py` resolves a
  name (case-insensitive, with an alias map) to a `BaseModelSpec`: YAML element first,
  then saved custom model, then hardcoded fallback, then an empty `BaseModelSpec`. It
  never raises on an unknown name — it degrades. A `Parameter` carries `value`, `type`,
  optional `min`/`max`/`fixed`/`link_group`, surfaced to the auto-built form via `to_spec()`.
- **Composite models.** `CompositeModelSpec` flattens component parameters into
  prefixed flat names (`_param_links` maps flat name → (component, local name)). Add a
  component by cloning its spec; the clone must keep prefixes and link metadata.
- **`ModelState.evaluate()` fallback chain.** Tries, in order: a passed callable →
  `state.model` (`.evaluate(x, params)` then callable) → `state.model_spec`. New models
  must expose `evaluate(x, params)` or be callable with that signature, or previews and
  fits silently return zeros.
- **Exclusion mask.** `ModelState.excluded` is a bool array (True = excluded), reset to
  match data length on load. Fitting uses `get_masked_data()`; edit exclusions through
  the viewmodel (`toggle_box_exclusion`), never in the View.
- **Input dir resolution:** `config.default_load_folder` → `FMO_ANALYSIS_INPUT_DIR`
  env var → `~/Documents` (`dataio/data_loader.py`). Preserve that order.

## Before You Touch X

| Task area | Read first |
|---|---|
| Parameter linking / fit representative logic | `docs/IMPLEMENTATION_NOTES.md`, `docs/PARAMETER_LINKING.md` |
| Plot events → viewmodel wiring | `IMPLEMENTATION_SUMMARY.md`, `INPUT_HANDLER_INTEGRATION.md` |
| Save / export (data, params, images, fits) | `docs/SAVE_FUNCTIONALITY.md` |
| Saving / loading custom composite models | `docs/SAVE_CUSTOM_MODEL.md`, `docs/SAVE_MODEL_UI_GUIDE.md` |

## Module Ownership

- `models/` — `model_state.py` (runtime data + active spec + exclusion), `model_specs.py`
  (`BaseModelSpec`/`CompositeModelSpec`, `Parameter`, `get_model_spec` registry),
  `model_elements/*.yaml` + `loader.py` (element definitions). Owns the math and spec;
  owns no UI and no threading.
- `viewmodel/fitter_vm.py` — the only View↔Model bridge; all business logic, fit
  orchestration, config save. Owns coordination; owns no Qt widgets.
- `view/` — pure UI: `main_window.py`, `docks/`, `dialogs/`, `view_box.py` (custom
  pyqtgraph viewbox emitting selection/exclusion signals), `input_handler.py`
  (centralizes plot events into signals). Owns no calculation or model access.
- `worker/fit_worker.py` — `QThread` running `curve_fit`; emits `(fit_result, y_fit)`.
  Never mutate widgets from inside it — cross threads only via Qt signals.
- `dataio/` — `configuration.py` (settings), `data_loader.py` (delimiter auto-detect,
  returns `(x, y, errors, file_info)`), `data_saver.py` / `data_exporter.py`,
  `fit_persistence.py`. Owns all file I/O.

## Anti-Patterns (project-specific)

- **Adding a model without an `evaluate(x, params)` / callable surface** — the fallback
  chain returns zeros and the preview looks blank with no error.
- **Registering a new model only in the hardcoded fallback** — prefer a
  `model_elements/*.yaml` element (see `_template.yaml`); the hardcoded specs exist for
  backward compatibility, not as the extension point.
- **Widening the top-level import surface** — do not add a root `__init__.py` or convert
  to a `PUFFIN.` package; it breaks every `from models import ...` and the launchers.

## Verification (cheapest first)

1. **Import check (no GUI):**
   `.venv\Scripts\python.exe -c "import models.model_state, models.model_specs, dataio, viewmodel.fitter_vm, worker.fit_worker"`
   — run from the repo root. Verified passing 2026-07-12.
2. **Tests — read this before trusting them.** `tests/` are **standalone scripts**, not
   a pytest suite, and **pytest is not installed in `.venv`** (`python -m pytest` fails).
   Run directly: `.venv\Scripts\python.exe tests\test_save_custom_model.py`. As of
   2026-07-12: `test_save_custom_model.py` passes only under UTF-8 stdio (set
   `PYTHONUTF8=1`; the default Windows cp1252 console crashes on its `✓`/`✗` output).
   `test_save_functionality.py` **fails on Windows** — it hardcodes `/tmp/` output paths.
   Treat green here as "the save round-trip logic works," not "the suite is CI-clean."
3. **Do not launch the GUI** as a verification step in an agent session.

---

*Last updated: 2026-07-12. Folds in the former `.github/copilot-instructions.md` and*
*`.github/agents/my-agent.md` (both deleted as antiquated).*
