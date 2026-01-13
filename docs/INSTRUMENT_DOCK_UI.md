# Instrument Dock UI Layout

This document shows the expected appearance of the Instrument Dock based on the configuration.

## Dock Layout (Example Spectrometer)

```
┌─────────────────────────────────────────────────┐
│ Instrument                                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  Instrument: [Example Spectrometer ▼]  [Load]  │
│                                                 │
│  ┌──────────── Crystals ─────────────┐         │
│  │                                    │         │
│  │  Monochromator: [PG(002) (d=3.354Å) ▼]     │
│  │  Analyzer:      [PG(002) (d=3.354Å) ▼]     │
│  │                                    │         │
│  └────────────────────────────────────┘         │
│                                                 │
│  ┌──────────── Focusing ──────────────┐        │
│  │                                     │        │
│  │  Monochromator: [flat ▼]           │        │
│  │  Analyzer:      [flat ▼]           │        │
│  │                                     │        │
│  └─────────────────────────────────────┘        │
│                                                 │
│  ┌──────────── Collimators ────────────┐       │
│  │                                      │       │
│  │  Before Mono:     [40 ▼]            │       │
│  │  Before Sample:   [20 ▼]            │       │
│  │  Before Analyzer: [20 ▼]            │       │
│  │  Before Detector: [20 ▼]            │       │
│  │                                      │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  ┌──────────── Slits ──────────────────┐       │
│  │                                      │       │
│  │  ┌─ Monochromator Slits ──────────┐ │       │
│  │  │  Horizontal (mm): [20.0] mm    │ │       │
│  │  │  Vertical (mm):   [20.0] mm    │ │       │
│  │  └────────────────────────────────┘ │       │
│  │                                      │       │
│  │  ┌─ Sample Slits ──────────────────┐ │      │
│  │  │  Horizontal (mm): [10.0] mm     │ │      │
│  │  │  Vertical (mm):   [10.0] mm     │ │      │
│  │  └─────────────────────────────────┘ │      │
│  │                                       │      │
│  │  ┌─ Analyzer Slits ─────────────────┐ │     │
│  │  │  Horizontal (mm): [15.0] mm      │ │     │
│  │  │  Vertical (mm):   [15.0] mm      │ │     │
│  │  └──────────────────────────────────┘ │     │
│  │                                        │     │
│  └────────────────────────────────────────┘     │
│                                                 │
│  ┌──────────── Experimental Modules ────┐      │
│  │                                       │      │
│  │  ┌─ Cryostat ────────────────────┐   │      │
│  │  │  ☐ Enable                     │   │      │
│  │  │  Temperature (K): [10.0]      │   │      │
│  │  └───────────────────────────────┘   │      │
│  │                                       │      │
│  │  ┌─ Furnace ─────────────────────┐   │      │
│  │  │  ☐ Enable                     │   │      │
│  │  │  Temperature (K): [500.0]     │   │      │
│  │  └───────────────────────────────┘   │      │
│  │                                       │      │
│  │  ┌─ Magnetic Field ──────────────┐   │      │
│  │  │  ☐ Enable                     │   │      │
│  │  │  Field (T): [0.0]             │   │      │
│  │  └───────────────────────────────┘   │      │
│  │                                       │      │
│  └───────────────────────────────────────┘      │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Widget Types by Section

### Instrument Selector
- **ComboBox**: Lists all available instruments from `config/instruments/*.yaml`
- **Button**: "Load" - triggers instrument configuration loading

### Crystals
- **ComboBox** (per component): Shows crystal name and d-spacing
  - Options from `monochromator.crystals` and `analyzer.crystals`
  - Format: "PG(002) (d=3.354Å)"

### Focusing
- **ComboBox** (per component): Simple dropdown
  - Options from `monochromator.focusing` and `analyzer.focusing`
  - Values: "flat", "horizontal", "vertical", "double"

### Collimators
- **ComboBox** (per position): Shows aperture size or "open"
  - Options from `collimators.positions[].options`
  - Mixed types: numbers (20, 40, 60) or strings ("open")

### Slits
- **GroupBox** (per position): Contains dimension controls
- **QDoubleSpinBox** (per dimension): Adjustable slit size
  - Min/max from `slits.positions[].dimensions[].min/max`
  - Default from `slits.positions[].dimensions[].default`
  - Step size from `slits.positions[].dimensions[].step`
  - Suffix: " mm"

### Experimental Modules
- **GroupBox** (per module): Contains enable checkbox and parameters
- **QCheckBox**: "Enable" - toggles module on/off
- **Widget** (per parameter): Type depends on parameter spec
  - Float: QDoubleSpinBox
  - Int: QSpinBox
  - Bool: QCheckBox
  - Choice: QComboBox
  - String: QLineEdit

## Interaction Flow

### Loading an Instrument
1. User selects instrument from dropdown
2. User clicks "Load" button
3. System loads YAML file
4. Dock rebuilds with instrument-specific controls
5. Log shows: "Loaded instrument: Example Spectrometer"

### Adjusting a Slit
1. User changes "Sample Slits → Horizontal" to 15.0
2. Signal emitted: `slit_changed("sample_slits", "horizontal", 15.0)`
3. ViewModel updates: `_instrument_state['slits']['sample_slits']['horizontal'] = 15.0`
4. Log shows: "Slit sample_slits horizontal: 15.0 mm"

### Changing a Crystal
1. User selects "Cu(220)" from Monochromator dropdown
2. Signal emitted: `crystal_changed("monochromator", "Cu(220)")`
3. ViewModel updates: `_instrument_state['crystals']['monochromator'] = "Cu(220)"`
4. Log shows: "Monochromator crystal: Cu(220)"

### Enabling a Module
1. User checks "Enable" on Cryostat
2. Signal emitted: `module_enabled_changed("cryostat", True)`
3. ViewModel updates: `_instrument_state['modules']['cryostat']['enabled'] = True`
4. Log shows: "Module cryostat: enabled"
5. User adjusts temperature to 20.0 K
6. Signal emitted: `module_parameter_changed("cryostat", "temperature", 20.0)`
7. ViewModel updates: `_instrument_state['modules']['cryostat']['parameters']['temperature'] = 20.0`
8. Log shows: "Module cryostat temperature: 20.0"

## Responsive Behavior

- Dock is scrollable to accommodate many controls
- Minimum width: automatically sized to fit content
- Initially hidden - shown via Docks menu or button
- Can be docked or floating
- Controls are disabled until an instrument is loaded

## Style Consistency

All controls follow PUFFIN's existing style:
- GroupBoxes for logical sections
- FormLayout for label-control pairs
- Consistent spacing (8px margins, 8px spacing)
- Standard Qt widgets (no custom styling)
- Tooltips could be added in future enhancement

## Accessibility

- All controls have descriptive labels
- Spin boxes show units in suffix
- ComboBoxes show full information (e.g., crystal name + d-spacing)
- Enable checkboxes clearly indicate module state
- Log feedback for all changes

## Future Enhancements

Potential UI improvements:
- Add tooltips to controls explaining physical meaning
- Color-code modules by type (sample environment, detection, etc.)
- Add "Reset to Defaults" button
- Show current instrument name in dock title
- Add icons for different module types
- Collapsible sections for rarely-used controls
- Quick presets (e.g., "High Resolution", "High Flux")
