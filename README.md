# PUFFIN

PUFFIN is a PySide6 desktop app for interactive 1D curve fitting (e.g., spectroscopy / lineshape analysis). It provides live plotting with PyQtGraph, dynamic parameter forms, SciPy-based fitting, and a clean MVVM split between view, viewmodel, and model state.

## Key Features
- MVVM architecture: `view/` is UI-only, `viewmodel/` owns coordination, `models/` holds specs and runtime state.
- Interactive plotting: mouse/keyboard handlers for clicks, drags, wheel, and shortcuts with logging to the dock panel.
- Dynamic parameter UI: widgets are auto-generated from model specs (min/max/choices/steps/decimals) and stay in sync with the underlying model state.
- Parameter linking: tie parameters into link groups for shared values; visually indicated in the parameters dock and honored during fitting.
- Model library: YAML-based elements (Gaussian, Voigt, DHO, linear background) and composite specs; extend by adding a `model_elements/*.yaml` and registering in `models/model_specs.py`.
- Fitting workflow: SciPy `curve_fit` wrapped in a worker thread; results push back into both the model spec and runtime model, refreshing the UI.
- Data IO: flexible loader auto-detects delimiters; saver writes tab-separated columns (`Energy`, `Counts`, optional `Fit`). Configuration stored in `config/settings.json`.

## Requirements
- Python 3.10+
- Dependencies: PySide6, pyqtgraph, numpy, scipy, pandas, matplotlib (see `requirements.txt`).

Install locally:
```bash
python -m pip install -r requirements.txt
```

## Run the App
From the repo root:
```bash
python -m PUFFIN.main
```
This launches the GUI with the default docks (plot, controls, parameters, log, elements, resolution).

## Typical Workflow
1. Load data via the File menu (CSV/TSV; columns `Energy`, `Counts`, optional `Errors`). Loader infers errors if absent.
2. Choose or build a model in the Elements/Parameters docks. Parameter widgets mirror the active model spec.
3. Adjust parameters and click **Update** to preview the curve. Use link groups to lock parameters together.
4. Run **Fit** to optimize with `curve_fit`; results propagate back into the UI and the runtime model.
5. Save the fitted dataset (data + optional fit column) from the File menu.

## Keyboard & Mouse Shortcuts
- `R`: reset/auto-range the plot view.
- `F`: run fit.
- `U`: update plot with current parameters.
- `Space`: clear selection (viewmodel-driven).
- `Ctrl + Mouse Wheel`: example parameter tweak hook (see `FitterViewModel.handle_wheel_scroll`).
- Click/move/wheel events are logged; the `InputHandler` emits signals that MainWindow forwards to the viewmodel.

## Configuration
- Defaults live in `config/settings.json`. The loader uses `default_load_folder`, falling back to `FMO_ANALYSIS_INPUT_DIR` then `~/Documents`.
- Call `FitterViewModel.save_config()` after changing defaults so subsequent runs pick them up.

## Project Structure (high level)
- `main.py`: entry point that wires `ModelState`, `FitterViewModel`, and `view/main_window.py`.
- `models/`: model specs, runtime state, metrics, YAML element definitions.
- `view/`: PySide6 UI, docks, input handler, custom view box.
- `viewmodel/`: coordination/business logic, logging helpers.
- `worker/`: threaded fitting worker.
- `dataio/`: configuration, loaders/savers, fit persistence.
- `examples/`: usage examples (e.g., input handler patterns).
- `docs/`: implementation notes and design guides.

## Extending
- Add a new model: create `model_elements/<name>.yaml`, register in `models/model_spec.get_model_spec()`, and expose parameters via `Parameter` helpers so the auto-form renders meaningful controls.
- Add interactions: extend `view/input_handler.py` signals or `viewmodel/fitter_vm.py` handlers; keep UI mutations out of worker threads.

## Troubleshooting
- Plot updates but data lengths mismatch: ensure `x_data`, `y_data`, and `errors` arrays are trimmed to equal length before emitting `plot_updated`.
- UI not reflecting parameter changes: confirm `FitterViewModel.apply_parameters()` is called so values mirror into both `state.model` and `state.model_spec.params`.
- File dialog defaults: verify `config/settings.json` is readable and follows the default folder resolution order.

## License
Provide or confirm a license before distribution; none is bundled in the repository.
