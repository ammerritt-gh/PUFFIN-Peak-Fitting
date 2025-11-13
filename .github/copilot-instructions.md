description: Practical orientation for AI copilots working on BigFit.
---

# BigFit Copilot Notes
- **Entry point**: `BigFit/main.py` spins up the PySide6 app, instantiates `ModelState`, `FitterViewModel`, and `view.main_window.MainWindow`, then wires Qt signals (`plot_updated`, `log_message`, `parameters_updated`).
- **MVVM split**: `view/main_window.py` is strictly UI; `viewmodel/fitter_vm.py` owns coordination/business logic; `models/model_state.py` keeps the runtime data and model spec.
- **Plot stack**: `view/view_box.CustomViewBox` emits selection/exclusion signals; `view/input_handler.InputHandler` converts those into `FitterViewModel` calls; the view never touches model data directly.
- **Dynamic parameter UI**: `MainWindow._refresh_parameters()` expects `FitterViewModel.get_parameters()` to return `{name: spec_dict}` where spec keys include `value`, `type`, optional `min`/`max`/`choices`/`decimals`/`step`. Missing keys default to sensible widgets.
- **Parameter flow**: UI widgets populate a dict → `FitterViewModel.apply_parameters()` writes into `state.model` (a `SimpleNamespace`) and mirrors values back into `state.model_spec.params` to keep subsequent refreshes in sync.
- **Model specs**: Extend `models/model_specs.BaseModelSpec`; use `Parameter` helpers (`ptype`, `min`, `decimals`, `step`) so the UI can render meaningful controls. Register new specs in `get_model_spec()`.
- **Model state evaluation**: `ModelState.evaluate()` tries, in order, a passed callable → `state.model` → `state.model_spec`. Ensure new models expose either `evaluate(x, params)` or are callable with that signature so live previews and fits work.
- **Data expectations**: `model_state` stores `x_data`, `y_data`, and `errors` as NumPy arrays. `plot_updated` consumers assume matching lengths; trim arrays before emitting.
- **Fitting workflow**: `FitterViewModel.run_fit()` wraps `model_spec.evaluate` for SciPy `curve_fit`, spins up `worker/fit_worker.FitWorker` (QThread). Only update UI via its signals; never mutate widgets inside the worker thread.
- **Fit results**: `FitWorker` emits `(fit_result_dict, y_fit)`; `on_finished` copies fitted values into both `state.model_spec.params` and `state.model`, then emits `parameters_updated` so the form rebuilds with latest numbers.
- **Curve bookkeeping**: `MainWindow` tracks plotted items in `self.curves`; `InputHandler.detect_curve_at()` expects `viewmodel.curves` to map curve ids → `(x, y)` arrays. When adding new overlays, update both mappings to keep selection features alive.
- **Logging**: Prefer emitting `log_message` from viewmodel/worker; the view routes it to the docked QTextEdit. Fall back to `print` only inside guarded `except` blocks as seen in existing code.
- **Data loading**: `dataio/data_loader.py` prompts via QFileDialog, auto-detects delimiters, and returns `(energy, counts, errors, file_info)`. Errors are inferred if absent; respect these conventions when adding loaders.
- **Saving**: `dataio/data_saver.save_dataset()` writes tab-separated columns (`Energy`, `Counts`, optional `Fit`). Provide `y_fit` for full exports.
- **Configuration**: `dataio/configuration.get_config()` maintains a repo-local `config/settings.json`. Updating defaults must call `FitterViewModel.save_config()` so the singleton reload logic keeps everything consistent.
- **Default folders**: `resolve_default_input_dir()` prioritizes the config’s `default_load_folder`, then `FMO_ANALYSIS_INPUT_DIR`, then `~/Documents`. Preserve that order for predictable UX.
- **Hotkeys & input**: `InputHandler.handle_key()` logs space/± events and clears selection via `viewbox.clear_selection()`. New shortcuts should follow this pattern and talk to the viewmodel first.
- **Background exclusions**: ViewBox exclusion mode uses dashed orange rectangles (`excludeBoxDrawn`); extend exclusion logic in the viewmodel (`toggle_box_exclusion`) rather than in the view.
- **Extending models**: When adding composite models (e.g., DHO+Voigt), keep heavy math in `models/`; surface UI-tunable values through `Parameter` instances so the auto-form keeps working.
- **Dependencies**: Core runtime uses PySide6, pyqtgraph, numpy, scipy, pandas (CSV IO), matplotlib (only for separate experiments). Install via `pip install PySide6 pyqtgraph numpy scipy pandas matplotlib` before running the UI.
- **Running locally**: From repo root, start the GUI with `python -m BigFit.main`. Visual Studio (.pyproj) already points to `main.py` if you use that IDE.
- **Thread safety**: Only Qt signals cross threads. If you add long-running work, subclass `QThread` like `FitWorker` and emit results instead of touching widgets directly.
- **Error handling**: Loader and fit paths wrap exceptions and broadcast user-friendly messages (`RuntimeError`, log text). Follow that precedent to avoid hard crashes in the GUI.
- **Prototypes**: `Minis Testing/` contains archived PySide experiments; none are imported by the main app. Use them for reference but avoid introducing new dependencies there.
- **Legacy references**: `Model Test.py` points at `models.dho_voigt_model` which is absent. Treat it as inspiration, not an active dependency.
- **Coding style**: Stick to ASCII, rely on short inline comments sparingly, and surface user-facing feedback via Qt signals.
