# Instrument Configuration System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PUFFIN Application                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────┐
        │          Main Window (view)                │
        │  ┌──────────────────────────────────────┐  │
        │  │      Instrument Dock Widget          │  │
        │  │  - Instrument Selector               │  │
        │  │  - Slit Controls (dynamically        │  │
        │  │    generated)                        │  │
        │  │  - Collimator Selectors              │  │
        │  │  - Crystal Selectors                 │  │
        │  │  - Focusing Options                  │  │
        │  │  - Module Controls                   │  │
        │  └──────────────────────────────────────┘  │
        └────────────────────────────────────────────┘
                        │         ▲
                        │ signals │ updates
                        ▼         │
        ┌────────────────────────────────────────────┐
        │      FitterViewModel (viewmodel)           │
        │  - load_instrument_from_name()             │
        │  - get_instrument_config()                 │
        │  - set_slit_value()                        │
        │  - set_collimator_value()                  │
        │  - set_crystal()                           │
        │  - set_focusing()                          │
        │  - set_module_enabled()                    │
        │  - _instrument_config: InstrumentConfig    │
        │  - _instrument_state: dict                 │
        └────────────────────────────────────────────┘
                        │         ▲
                        │ calls   │ returns
                        ▼         │
        ┌────────────────────────────────────────────┐
        │   instrument_config.py (dataio)            │
        │  - load_instrument_config()                │
        │  - list_available_instruments()            │
        │  - InstrumentConfig dataclass              │
        │  - CrystalSpec dataclass                   │
        │  - SlitPosition dataclass                  │
        │  - CollimatorPosition dataclass            │
        │  - ExperimentalModule dataclass            │
        └────────────────────────────────────────────┘
                        │         ▲
                        │ reads   │ returns
                        ▼         │
        ┌────────────────────────────────────────────┐
        │      YAML Configuration Files              │
        │    config/instruments/*.yaml               │
        │  - example_spectrometer.yaml               │
        │  - simple_spectrometer.yaml                │
        │  - (user-defined configurations)           │
        └────────────────────────────────────────────┘
```

## Data Flow

### Loading an Instrument

```
User Action: Select Instrument & Click Load
         │
         ▼
InstrumentDock.instrument_selected(name) [signal]
         │
         ▼
MainWindow._on_instrument_selected(name)
         │
         ▼
FitterViewModel.load_instrument_from_name(name)
         │
         ├─► list_available_instruments()
         │   └─► Returns: [{name, path, description}, ...]
         │
         ├─► load_instrument_config(path)
         │   ├─► Parse YAML file
         │   └─► Returns: InstrumentConfig object
         │
         ├─► _initialize_instrument_state(config)
         │   └─► Returns: Initial state dict
         │
         └─► Store in _instrument_config & _instrument_state
         
MainWindow receives InstrumentConfig
         │
         ▼
InstrumentDock.set_instrument_config(config)
         │
         ▼
InstrumentDock._rebuild_controls()
         │
         ├─► _build_crystal_section()
         ├─► _build_focusing_section()
         ├─► _build_collimator_section()
         ├─► _build_slit_section()
         └─► _build_module_section()
         
UI Controls Rendered
```

### Adjusting a Parameter

```
User Action: Change Slit Size
         │
         ▼
QDoubleSpinBox.valueChanged [Qt signal]
         │
         ▼
InstrumentDock._on_slit_changed(pos, dim, value)
         │
         ▼
InstrumentDock.slit_changed(pos, dim, value) [signal]
         │
         ▼
MainWindow._on_slit_changed(pos, dim, value)
         │
         ▼
FitterViewModel.set_slit_value(pos, dim, value)
         │
         ├─► Update _instrument_state dict
         └─► Log message to UI
```

## YAML Configuration Structure

```yaml
name: "Instrument Name"
type: "instrument_type"
description: "Human-readable description"

arm_lengths:
  component1_to_component2: 1000.0  # in mm

monochromator/analyzer:
  crystals:
    - name: "Crystal Name"
      d_spacing: 3.354  # Angstroms
      mosaic: 0.4       # degrees
  focusing:
    - "flat"
    - "horizontal"

collimators:
  positions:
    - name: "position_id"
      label: "Display Label"
      options: [value1, value2, "open"]

slits:
  positions:
    - name: "position_id"
      label: "Display Label"
      dimensions:
        - name: "dimension_id"
          label: "Display Label"
          min: 0.0
          max: 50.0
          default: 10.0
          step: 0.5

modules:
  - name: "module_id"
    label: "Display Label"
    enabled: false
    parameters:
      - name: "param_id"
        label: "Display Label"
        type: "float"
        min: 0.0
        max: 100.0
        default: 50.0
```

## Extension Points

To extend the system:

1. **Add New Instrument**: Create YAML file in `config/instruments/`
2. **Add New Parameter Type**: Extend `_create_parameter_widget()` in InstrumentDock
3. **Add Custom Validation**: Override parameter setters in FitterViewModel
4. **Integrate with Analysis**: Use `viewmodel.get_instrument_state()` in analysis code
5. **Add Persistence**: Extend configuration.py to save/restore instrument state

## Key Design Principles

- **Separation of Concerns**: View (dock) ↔ ViewModel ↔ Data (YAML)
- **Dynamic Generation**: UI controls created from configuration at runtime
- **Type Safety**: Dataclasses enforce structure and validation
- **Extensibility**: New instruments added without code changes
- **Testability**: Module can be tested without GUI dependencies
