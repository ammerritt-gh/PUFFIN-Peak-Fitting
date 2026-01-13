# PUFFIN

PUFFIN is a desktop app for interactive 1D curve fitting (PySide6 + PyQtGraph + SciPy). Load a spectrum, pick/build a model, tweak parameters with live previews, exclude bad points, run a fit, and export results.

## Install

Requirements: Python 3.10+

```bash
python -m pip install -r requirements.txt
```

## Run

From the repo root:

```bash
python main.py
```

## How To Use

1. **Load data**
	- Open the **Controls** dock and click **Load Data**.
	- You can load multiple files; pick the active dataset from **Loaded Files**.

2. **Build or choose a model**
	- Use the **Elements** dock to assemble a model (peaks/background/components).
	- The **Parameters** dock updates automatically for the active model.

3. **Preview (live)**
	- Adjust parameter values in **Parameters**; the plot updates as you change values.
	- Click **Update Plot** in **Controls** if you want a manual refresh.

4. **Exclude bad points**
	- Click **Exclude** (Controls) to enable exclusion mode.
	- Click points or drag a box to toggle exclusions.
	- Click **Include All** to clear exclusions.

5. **Fit**
	- Click **Run Fit** (Controls) or press **F**.
	- Fit results are written back into the parameter panel.

6. **Save**
	- Click **Save Data** (Controls) to open the **Save Data** dock.
	- Export plot images, ASCII data, and parameter summaries (or **Save All**).

7. **Configure Instrument** (Optional)
	- Open the **Instrument** dock from the **Docks** menu.
	- Select an instrument configuration and click **Load**.
	- Adjust slits, collimators, crystals, and other instrument-specific parameters.
	- See [Instrument Configuration](docs/INSTRUMENT_CONFIGURATION.md) for details.

## Data Files

- Supported: `.dat`, `.txt`, `.csv`
- Format: at least **two numeric columns** per row (`x`, `y`); an optional **third numeric column** is used as `error`.
- Headers/notes are OK: the loader skips non-numeric lines until it finds data.
- Delimiters are auto-detected (comma/tab/whitespace). Lines starting with `#` are treated as comments.

## Shortcuts

- **F**: run fit
- **Space**: clear selection
- **D**: toggle exclude mode
- **0**: clear selected curve/component
- **1–9**: select a component (composite models)
- **Q / E**: cycle selected component

## Tips

- Use the **Docks** menu to show/hide panels (Controls, Parameters, Elements, Log, Save Data, Resolution, Fit Settings, Instrument).
- On startup, PUFFIN can restore the last loaded dataset and any saved fit (when available).
- Instrument configurations are stored in `config/instruments/` as YAML files — you can create new configurations for different spectrometers.

## License

Licensed under the GNU General Public License v3.0. See `LICENSE`.
