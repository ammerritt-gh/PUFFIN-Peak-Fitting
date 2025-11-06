# Testing Custom Composite Models

This directory contains test scripts and examples for the custom composite models feature.

## Test Files

### Basic Functionality Tests
**Location:** `/tmp/test_custom_models.py`

Tests core functionality:
- CustomModelRegistry CRUD operations
- CompositeModelSpec creation and evaluation
- Loading custom models via get_model_spec
- Model names in get_available_model_names

**Run:**
```bash
cd /home/runner/work/BigFit/BigFit
python3 /tmp/test_custom_models.py
```

### Comprehensive Example
**Location:** `/tmp/example_custom_model.py`

Demonstrates:
- Creating custom models programmatically
- Adding multiple components (Gaussian, Lorentzian, Linear)
- Evaluating composite models
- Parameter structure and grouping
- Modifying models (add, remove, reorder)
- Visualization with matplotlib

**Run:**
```bash
cd /home/runner/work/BigFit/BigFit
python3 /tmp/example_custom_model.py
```

Generates: `/tmp/custom_model_example.png`

## Test Results

All tests passing ✅:
- ✓ Registry initialization and file management
- ✓ Model creation and deletion
- ✓ Component add/remove/reorder
- ✓ Parameter persistence
- ✓ Composite evaluation (summation)
- ✓ Grouped parameter structure
- ✓ Model loading and caching
- ✓ Integration with existing infrastructure
- ✓ Error handling

## Running Tests

### Prerequisites
Install dependencies:
```bash
pip3 install numpy scipy PySide6 pyqtgraph matplotlib pandas
```

### Core Functionality Test
Quick test without GUI dependencies:
```bash
cd /home/runner/work/BigFit/BigFit/BigFit
python3 << EOF
import sys
sys.path.insert(0, '.')
from models.custom_model_registry import get_custom_model_registry
from models import get_model_spec, get_available_model_names
import numpy as np

# Test registry
registry = get_custom_model_registry()
registry.create_model('Test')
registry.add_component('Test', 'gaussian', 'Peak', {'Area': 5.0, 'Width': 1.0, 'Center': 0.0})

# Test loading
spec = get_model_spec('Custom: Test')

# Test evaluation
x = np.linspace(-5, 5, 100)
y = spec.evaluate(x)

print(f"✓ Model works, max value: {np.max(y):.3f}")

# Cleanup
registry.delete_model('Test')
print("✅ All tests passed!")
EOF
```

## GUI Testing

### Manual GUI Test Procedure

1. **Start the application:**
   ```bash
   cd /home/runner/work/BigFit/BigFit
   python3 main.py
   ```

2. **Create a custom model:**
   - Click "New Model..." button
   - Enter name: "Test Model"
   - Verify it appears in dropdown as "Custom: Test Model"

3. **Add components:**
   - Select "Custom: Test Model" from dropdown
   - Click "Add Element"
   - Select "Gaussian" → OK
   - Verify "Gaussian 1" group appears with parameters
   - Click "Add Element" again
   - Select "Lorentzian" → OK
   - Verify "Lorentzian 1" group appears

4. **Modify parameters:**
   - Change Gaussian 1 → Center to 2.0
   - Change Lorentzian 1 → Center to 5.0
   - Click "Update Plot"
   - Verify plot updates

5. **Reorder components:**
   - Click ↓ on Gaussian 1
   - Verify Lorentzian 1 is now first
   - Click ↑ on Lorentzian 1
   - Verify order restored

6. **Remove component:**
   - Click "Remove" on Lorentzian 1
   - Confirm dialog
   - Verify Lorentzian 1 group disappears

7. **Load data and fit:**
   - Click "Load Data"
   - Select a data file
   - Adjust initial parameters
   - Click "Run Fit"
   - Verify fit completes successfully

8. **Persistence test:**
   - Close application
   - Reopen application
   - Select "Custom: Test Model"
   - Verify components and parameters persisted

9. **Cleanup:**
   - Manually delete JSON file from `BigFit/config/custom_models/`

## Expected Test Output

### test_custom_models.py
```
Testing custom model registry...
✓ Registry initialized
✓ Created test model
✓ Added component
✓ Model has 2 components
✓ Moved component
✓ Removed component
✓ Deleted model
✅ All tests passed!
```

### example_custom_model.py
```
Creating Example Custom Composite Model
✓ Created custom model: Example: Two Gaussians
✓ Added Peak 1 (Gaussian at center=0)
✓ Added Peak 2 (Gaussian at center=5)
...
✅ Example Complete!
```

## Troubleshooting

### ImportError: No module named 'X'
Install missing dependency:
```bash
pip3 install X
```

### GUI tests fail (no display)
GUI tests require a display environment. Core functionality tests work in headless mode.

### Test files not found
Ensure you're running from the correct directory:
```bash
cd /home/runner/work/BigFit/BigFit
```

## Coverage

The test suite covers:
- ✓ Model lifecycle (create, load, delete)
- ✓ Component management (add, remove, move)
- ✓ Parameter handling (get, set, persist)
- ✓ Evaluation (sum of components)
- ✓ UI integration (grouped display)
- ✓ Error handling (edge cases)
- ✓ Persistence (JSON save/load)

Not covered (requires GUI):
- UI button interactions
- Model selector dropdown
- Parameter widget updates
- Fitting integration (requires Qt event loop)

## Notes

- Custom model JSON files are stored in `BigFit/config/custom_models/`
- Files are user-specific and gitignored
- Registry is a singleton (global state)
- Component evaluation is serial (not parallel)
- Only summation operator currently supported
