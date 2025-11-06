# Custom Composite Models Feature

## Overview

BigFit now supports creating custom composite models that combine multiple base models. This allows users to fit complex data with multiple peaks, backgrounds, and other features.

## Key Features

- **Create Custom Models**: Define new composite models with user-chosen names
- **Add Components**: Add any base model (Gaussian, Lorentzian, Voigt, Linear, etc.) as a component
- **Grouped Parameters**: Parameters are organized by component in the UI for clarity
- **Reorder Components**: Change the order of components as needed
- **Remove Components**: Delete unwanted components from custom models
- **Persistence**: Custom models are saved as JSON files and automatically loaded on startup
- **Summation**: The composite model evaluates as the sum of all component models

## Usage

### Creating a Custom Model via GUI

1. **Open the Application**: Launch BigFit
2. **Click "New Model..."**: In the Parameters panel (right dock)
3. **Enter Model Name**: Type a descriptive name for your custom model
4. **Select the Model**: Your new model will appear in the dropdown as "Custom: <name>"

### Adding Components

1. **Select Your Custom Model**: Choose it from the model dropdown
2. **Click "Add Element"**: Button appears at the top of the parameters panel
3. **Choose Base Model**: Select from available models (Gaussian, Lorentzian, etc.)
4. **Configure Parameters**: Each component's parameters appear in a grouped section

### Managing Components

Each component group has controls:
- **↑ (Up Arrow)**: Move component earlier in the evaluation order
- **↓ (Down Arrow)**: Move component later in the evaluation order
- **Remove**: Delete this component from the model

### Parameters

Parameters are organized in **labeled frames** (QGroupBox), one per component:

```
┌─ Gaussian 1 ─────────────────┐
│ ↑ ↓ Remove                   │
│ Area:   1.0                   │
│ Width:  1.0                   │
│ Center: 0.0                   │
└──────────────────────────────┘

┌─ Lorentzian 1 ───────────────┐
│ ↑ ↓ Remove                   │
│ Area:   0.5                   │
│ Width:  0.5                   │
│ Center: 5.0                   │
└──────────────────────────────┘
```

## Programmatic Usage

### Creating Models Programmatically

```python
from models.custom_model_registry import get_custom_model_registry

# Get the registry
registry = get_custom_model_registry()

# Create a new custom model
registry.create_model("Two Peaks")

# Add components
registry.add_component(
    "Two Peaks",
    base_spec="gaussian",
    label="Peak 1",
    params={"Area": 10.0, "Width": 1.5, "Center": 0.0}
)

registry.add_component(
    "Two Peaks",
    base_spec="lorentzian",
    label="Peak 2",
    params={"Area": 5.0, "Width": 0.8, "Center": 5.0}
)
```

### Using Custom Models

```python
from models import get_model_spec
import numpy as np

# Load a custom model
model_spec = get_model_spec("Custom: Two Peaks")

# Evaluate the model
x = np.linspace(-5, 10, 500)
y = model_spec.evaluate(x)
```

### Accessing Components

```python
# Get component information
model_data = registry.get_model("Two Peaks")
components = model_data["components"]

for comp in components:
    print(f"{comp['label']}: {comp['base_spec']}")
    print(f"  Parameters: {comp['params']}")
```

## Storage

Custom models are stored as JSON files in:
```
BigFit/config/custom_models/<model_name>.json
```

Example JSON structure:
```json
{
  "name": "Two Peaks",
  "components": [
    {
      "base_spec": "gaussian",
      "label": "Peak 1",
      "params": {
        "Area": 10.0,
        "Width": 1.5,
        "Center": 0.0
      }
    },
    {
      "base_spec": "lorentzian",
      "label": "Peak 2",
      "params": {
        "Area": 5.0,
        "Width": 0.8,
        "Center": 5.0
      }
    }
  ]
}
```

**Note**: These files are user-specific and are ignored by git (added to `.gitignore`).

## Model Evaluation

The composite model is evaluated as the **sum** of all component outputs:

```
y_composite(x) = Σ component_i.evaluate(x, params_i)
```

Each component is evaluated independently with its own parameters, then the results are summed.

## Available Base Models

The following models can be added as components:

- **Gaussian**: Symmetric peak with FWHM-based width
- **Lorentzian**: Peak with longer tails
- **Voigt**: Convolution of Gaussian and Lorentzian
- **DHO**: Damped Harmonic Oscillator
- **Linear**: Linear background (slope + constant)

**Note**: Constructed models like "DHO+Voigt" cannot be added to custom models to prevent excessive nesting.

## ViewModel API

The `FitterViewModel` provides methods for custom model management:

### Creating Models
```python
viewmodel.create_custom_model(name: str) -> bool
```

### Adding Components
```python
viewmodel.add_component_to_custom_model(
    model_name: str,
    base_spec: str,
    label: str = None
) -> bool
```

### Removing Components
```python
viewmodel.remove_component_from_custom_model(
    model_name: str,
    index: int
) -> bool
```

### Reordering Components
```python
viewmodel.move_component_in_custom_model(
    model_name: str,
    old_index: int,
    new_index: int
) -> bool
```

### Querying Components
```python
viewmodel.get_custom_model_components(
    model_name: str
) -> List[Dict[str, Any]]
```

### Checking Active Model
```python
viewmodel.is_custom_model_active() -> bool
```

## Implementation Details

### Architecture

- **CustomModelRegistry**: Singleton class managing persistence and CRUD operations
- **CompositeModelSpec**: Model specification that holds multiple component specs
- **Grouped Parameters**: Special parameter structure with "groups" key for UI rendering
- **Parameter Prefixing**: Component parameters are stored with "Label::ParamName" format

### Key Classes

1. **`models/custom_model_registry.py`**
   - `CustomModelRegistry`: Manages custom model lifecycle
   - `get_custom_model_registry()`: Singleton accessor

2. **`models/model_specs.py`**
   - `CompositeModelSpec`: Composite model implementation
   - Updated `get_model_spec()`: Recognizes "custom:" prefix
   - Updated `get_available_model_names()`: Includes custom models

3. **`viewmodel/fitter_vm.py`**
   - Custom model management methods
   - Updated `get_parameters()`: Supports grouped structure

4. **`view/main_window.py`**
   - "New Model..." button
   - "Add Element" button for active custom models
   - Component management controls (move up/down, remove)
   - Grouped parameter display with QGroupBox

## Examples

See `/tmp/example_custom_model.py` for a comprehensive example demonstrating:
- Creating custom models
- Adding multiple components
- Evaluating composite models
- Modifying component order
- Visualizing results

## Limitations

- Constructed models (e.g., "DHO+Voigt") cannot be added as components
- Custom models cannot be nested (no composite of composites)
- Component evaluation is serial (sum computed sequentially)

## Future Enhancements

Possible future improvements:
- Export/import custom models
- Duplicate existing custom models
- Rename custom models
- Different combination operations (product, convolution)
- Component templates/presets
- Undo/redo for model modifications
