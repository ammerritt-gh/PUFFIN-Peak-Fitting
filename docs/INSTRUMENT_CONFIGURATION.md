# Instrument Configuration System

## Overview

PUFFIN now supports dynamic instrument configuration through YAML files. This allows different instruments to be defined without modifying code, making it easy to adapt the application to various spectrometers and experimental setups.

## Features

### Configurable Components

- **Arm Lengths**: Define distances between instrument components for angle/energy conversions
- **Crystals**: Specify monochromator and analyzer crystals with d-spacing and mosaic values
- **Focusing Options**: Define available focusing configurations (flat, horizontal, vertical, double)
- **Collimators**: Configure collimator positions and available aperture sizes
- **Slits**: Define slit positions with adjustable dimensions
- **Experimental Modules**: Add sample environment equipment (cryostats, furnaces, magnetic fields)

### GUI Integration

The Instrument dock widget provides:
- Instrument selector to switch between configurations
- Dynamic controls generated from YAML definitions
- Slit size controls with appropriate ranges and step sizes
- Collimator selection dropdowns
- Crystal and focusing option selectors
- Experimental module controls with enable/disable checkboxes

## Configuration File Format

Instrument configurations are stored as YAML files in `config/instruments/`.

### Example Configuration

```yaml
name: "Example Spectrometer"
type: "triple_axis"
description: "Example triple-axis spectrometer configuration"

# Arm lengths (in mm)
arm_lengths:
  monochromator_to_sample: 1500.0
  sample_to_analyzer: 1200.0
  analyzer_to_detector: 800.0

# Monochromator crystal options
monochromator:
  crystals:
    - name: "PG(002)"
      d_spacing: 3.354  # Angstroms
      mosaic: 0.4       # degrees
    - name: "Cu(220)"
      d_spacing: 1.278
      mosaic: 0.2
  focusing:
    - "flat"
    - "horizontal"
    - "vertical"

# Analyzer crystal options (similar structure)
analyzer:
  crystals:
    - name: "PG(002)"
      d_spacing: 3.354
      mosaic: 0.4
  focusing:
    - "flat"
    - "horizontal"

# Collimators (in arcminutes)
collimators:
  positions:
    - name: "before_sample"
      label: "Before Sample"
      options: [20, 40, 60, "open"]

# Slits
slits:
  positions:
    - name: "sample_slits"
      label: "Sample Slits"
      dimensions:
        - name: "horizontal"
          label: "Horizontal (mm)"
          min: 0.0
          max: 30.0
          default: 10.0
          step: 0.5
        - name: "vertical"
          label: "Vertical (mm)"
          min: 0.0
          max: 30.0
          default: 10.0
          step: 0.5

# Experimental modules
modules:
  - name: "cryostat"
    label: "Cryostat"
    enabled: false
    parameters:
      - name: "temperature"
        label: "Temperature (K)"
        type: "float"
        min: 1.5
        max: 300.0
        default: 10.0
```

## Usage

### In the GUI

1. **Open the Instrument Dock**: Use the Docks menu to show the Instrument dock
2. **Select an Instrument**: Choose from the dropdown list of available instruments
3. **Click Load**: The instrument configuration will be loaded and controls will appear
4. **Adjust Parameters**: Use the generated controls to set:
   - Slit sizes
   - Collimator settings
   - Crystal selections
   - Focusing options
   - Experimental module parameters

### Creating New Instruments

1. Create a new YAML file in `config/instruments/`
2. Follow the structure shown in the example configurations
3. Restart PUFFIN (or use the Load button)
4. The new instrument will appear in the selector

### Programmatic Access

```python
from dataio import load_instrument_config, list_available_instruments
from pathlib import Path

# List available instruments
instruments = list_available_instruments()
for inst in instruments:
    print(f"{inst['name']}: {inst['description']}")

# Load a specific instrument
config = load_instrument_config(Path('config/instruments/example_spectrometer.yaml'))

# Access configuration data
arm_length = config.get_arm_length('monochromator_to_sample')
crystals = config.get_monochromator_crystals()
focusing_options = config.get_monochromator_focusing_options()
```

## ViewModel Integration

The FitterViewModel provides methods to manage instrument state:

```python
# Load an instrument by name
viewmodel.load_instrument_from_name("Example Spectrometer")

# Get current configuration
config = viewmodel.get_instrument_config()

# Get current parameter values
state = viewmodel.get_instrument_state()

# Set individual parameters
viewmodel.set_slit_value("sample_slits", "horizontal", 15.0)
viewmodel.set_collimator_value("before_sample", 40)
viewmodel.set_crystal("monochromator", "PG(002)")
viewmodel.set_focusing("analyzer", "horizontal")
viewmodel.set_module_enabled("cryostat", True)
viewmodel.set_module_parameter("cryostat", "temperature", 20.0)
```

## File Structure

```
config/
  instruments/
    example_spectrometer.yaml
    simple_spectrometer.yaml
dataio/
  instrument_config.py          # Configuration loading and data structures
view/
  docks/
    instrument_dock.py           # GUI dock widget
viewmodel/
  fitter_vm.py                   # Instrument state management
tests/
  test_instrument_config.py      # Test suite
```

## Data Structures

### InstrumentConfig

Main configuration object containing:
- `name`: Instrument name
- `type`: Instrument type (e.g., "triple_axis", "basic")
- `description`: Human-readable description
- `arm_lengths`: Dictionary of arm length values
- `monochromator`: ComponentSpec with crystals and focusing options
- `analyzer`: ComponentSpec with crystals and focusing options
- `collimators`: List of CollimatorPosition objects
- `slits`: List of SlitPosition objects
- `modules`: List of ExperimentalModule objects

### Supporting Classes

- **CrystalSpec**: Crystal definition (name, d_spacing, mosaic)
- **CollimatorPosition**: Collimator location and available options
- **SlitPosition**: Slit location with multiple SlitDimension objects
- **SlitDimension**: Single dimension specification (min, max, default, step)
- **ExperimentalModule**: Module definition with parameters
- **ModuleParameter**: Parameter specification (type, range, default)

## Testing

Run the test suite to verify the system:

```bash
python tests/test_instrument_config.py
```

Tests cover:
- YAML file parsing
- Configuration loading
- Data structure creation
- ViewModel integration
- Dock widget structure

## Future Enhancements

Potential additions:
- Save/restore instrument state between sessions
- Export current instrument configuration
- Validate instrument parameters against physical constraints
- Integration with data analysis pipeline for automatic corrections
- Support for complex instrument geometries
- Instrument-specific calculation modules

## Troubleshooting

### Instrument doesn't appear in list

- Check that the YAML file is in `config/instruments/`
- Verify the YAML syntax is valid
- Ensure required fields (name, type) are present

### Controls don't generate properly

- Check the YAML structure matches the examples
- Verify parameter types and ranges are specified correctly
- Look for error messages in the log dock

### Changes not taking effect

- Ensure you clicked the Load button after selecting an instrument
- Check that the ViewModel is properly connected
- Review log messages for any errors
