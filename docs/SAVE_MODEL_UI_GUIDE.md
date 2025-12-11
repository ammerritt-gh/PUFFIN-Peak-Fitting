# Save Custom Model - UI Guide

## Visual Walkthrough

This guide shows the user interface for the Save Custom Model feature.

## Step 1: Create Your Custom Model

**Elements Dock** - Build your model:
```
┌─────────────────────────────┐
│ Elements                    │
├─────────────────────────────┤
│ • peak1 (Gaussian)         │
│ • peak2 (Voigt)            │
│ • bg (Linear Background)    │
│                             │
│ [Add Element] [Remove]      │
│                             │
│ [ Save Model... ]           │ ← NEW BUTTON
└─────────────────────────────┘
```

**Parameters Dock** - Configure your parameters:
```
┌─────────────────────────────────────────┐
│ Parameters                              │
├─────────────────────────────────────────┤
│ Model: [Custom Model ▼]                │
│                                         │
│ peak1_Area:      100.0  [F] [Link: -] │ ← Fixed
│ peak1_Width:     2.5    [ ] [Link: 1] │ ← Linked (group 1)
│ peak1_Center:    0.0    [ ] [Link: -] │
│                                         │
│ peak2_Area:      50.0   [ ] [Link: -] │
│ peak2_Gauss:     1.5    [ ] [Link: 1] │ ← Linked (group 1)
│ peak2_Lorentz:   0.5    [ ] [Link: -] │
│ peak2_Center:    5.0    [F] [Link: -] │ ← Fixed
│                                         │
│ bg_Slope:        0.1    [ ] [Link: -] │
│ bg_Intercept:    10.0   [ ] [Link: -] │
│                                         │
│ [Apply] [Refresh]                       │
└─────────────────────────────────────────┘
```

## Step 2: Click "Save Model..." Button

The Save Model Dialog opens:

```
╔═══════════════════════════════════════════════════════════════════╗
║ Save Custom Model                                            [X]  ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║ ┌─ Model Information ────────────────────────────────────────┐   ║
║ │                                                             │   ║
║ │ Model Name:    [Two Peaks with Background____________]     │   ║
║ │                                                             │   ║
║ │ Description:   [Custom model with two peaks and a  ]       │   ║
║ │                [linear background                  ]       │   ║
║ │                                                             │   ║
║ └─────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║ ┌─ Model Structure ──────────────────────────────────────────┐   ║
║ │                                                             │   ║
║ │ Components and Parameters:                                 │   ║
║ │                                                             │   ║
║ │ ┌─────────────────────────────────────────────────────────┐│   ║
║ │ │ Element/Parameter  Value   Fixed  Link  Min    Max     ││   ║
║ │ │ ────────────────────────────────────────────────────────││   ║
║ │ │ ▼ Component 1: peak1 (Gaussian)                        ││   ║
║ │ │     Area           100.0   Yes    -     0.0    1000.0  ││   ║
║ │ │     Width          2.5     No     1     0.1    -       ││   ║
║ │ │     Center         0.0     No     -     -      -       ││   ║
║ │ │                                                          ││   ║
║ │ │ ▼ Component 2: peak2 (Voigt)                           ││   ║
║ │ │     Area           50.0    No     -     0.0    -       ││   ║
║ │ │     Gauss FWHM     1.5     No     1     -      -       ││   ║
║ │ │     Lorentz FWHM   0.5     No     -     -      -       ││   ║
║ │ │     Center         5.0     Yes    -     -      -       ││   ║
║ │ │                                                          ││   ║
║ │ │ ▼ Component 3: bg (Linear Background)                  ││   ║
║ │ │     Slope          0.1     No     -     -      -       ││   ║
║ │ │     Intercept      10.0    No     -     -      -       ││   ║
║ │ └─────────────────────────────────────────────────────────┘│   ║
║ │                                                             │   ║
║ └─────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║ ┌─ Save Location ────────────────────────────────────────────┐   ║
║ │                                                             │   ║
║ │ [/path/to/BigFit/models/custom_models] [Browse...]        │   ║
║ │                                                             │   ║
║ │ The model will be saved as a .yaml file                    │   ║
║ │                                                             │   ║
║ └─────────────────────────────────────────────────────────────┘   ║
║                                                                    ║
║                                         [  Save  ]  [ Cancel ]    ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Step 3: Model is Saved

**Status Message** in log:
```
Model saved to: /path/to/BigFit/models/custom_models/two_peaks_with_background.yaml
Model 'Two Peaks with Background' is now available for use.
```

**File Created**:
```yaml
# models/custom_models/two_peaks_with_background.yaml
name: Two Peaks with Background
description: Custom model with two peaks and a linear background
version: 1
author: BigFit User
category: composite
is_composite: true
components:
  - element: Gaussian
    prefix: peak1_
    default_parameters:
      Area:
        value: 100.0
        fixed: true
        min: 0.0
        max: 1000.0
      Width:
        value: 2.5
        link_group: 1
        min: 0.1
      Center:
        value: 0.0
  - element: Voigt
    prefix: peak2_
    default_parameters:
      Area:
        value: 50.0
        min: 0.0
      Gauss FWHM:
        value: 1.5
        link_group: 1
      Lorentz FWHM:
        value: 0.5
      Center:
        value: 5.0
        fixed: true
  - element: Linear Background
    prefix: bg_
    default_parameters:
      Slope:
        value: 0.1
      Intercept:
        value: 10.0
```

## Step 4: Model Appears in Selector

**Model Selector** (after restart or continuing):
```
┌─────────────────────────────┐
│ Model: [▼]                  │
├─────────────────────────────┤
│ Voigt                       │
│ Gaussian                    │
│ Linear Background           │
│ Custom Model                │
│ Two Peaks with Background   │ ← YOUR MODEL!
└─────────────────────────────┘
```

## Step 5: Load Your Saved Model

Select "Two Peaks with Background" from the model selector.

**Result**: The model loads with:
- ✅ All 3 components (2 peaks + background)
- ✅ peak1_Area is FIXED at 100.0
- ✅ peak2_Center is FIXED at 5.0
- ✅ peak1_Width and peak2_Gauss FWHM are LINKED (group 1)
- ✅ All bounds are preserved
- ✅ Ready to fit!

## Feature Highlights

### Tree View Benefits
- **Clear hierarchy**: See all components and their parameters
- **At-a-glance status**: Quickly identify fixed/linked parameters
- **Color coding**: Components shown in their assigned colors
- **Complete information**: Values, bounds, and constraints visible

### Smart Validation
- ✅ Model name must start with a letter
- ✅ Only alphanumeric, spaces, hyphens, underscores allowed
- ✅ Automatic filename sanitization (spaces → underscores)
- ✅ Duplicate filename detection
- ✅ Invalid character handling

### Error Messages
```
┌─────────────────────────────────────────┐
│ Invalid Name                      [!]   │
├─────────────────────────────────────────┤
│ Model name must start with a letter    │
│ and contain only letters, numbers,     │
│ spaces, hyphens, and underscores.      │
│                                         │
│ Note: Spaces and hyphens will be       │
│ converted to underscores in filename.  │
│                                         │
│                           [ OK ]        │
└─────────────────────────────────────────┘
```

### Filename Preview
If you enter "My Custom Model", the file will be saved as:
```
my_custom_model.yaml
```

Transformation:
- "My Custom Model" → "my_custom_model"
- Lowercase
- Spaces → underscores
- Hyphens → underscores
- Invalid chars removed

## Usage Tips

### 1. Descriptive Names
✅ Good: "Two Gaussian Peaks", "Voigt with Background", "Triple Peak Model"
❌ Bad: "Model1", "Test", "temp"

### 2. Use Descriptions
Add context about when to use the model:
```
Description: Two Gaussian peaks with linked widths.
Use for samples with two closely-spaced emission lines
where the resolution is expected to be the same.
```

### 3. Link Related Parameters
Link parameters that should vary together:
- Resolution-related widths → Same link group
- Background parameters → Same link group (if linear across all)
- Peak positions with known spacing → Link with constraints

### 4. Fix Known Values
Fix parameters when you know the value:
- Laser line positions
- Known background levels
- Instrumental resolution (if calibrated)

### 5. Set Reasonable Bounds
Prevent fitting from exploring unphysical regions:
- `Area: min=0.0` (negative areas are unphysical)
- `Width: min=0.1, max=10.0` (reasonable range for your data)
- `Center: min=-20.0, max=20.0` (if your x-axis spans this range)

## Keyboard Shortcuts

While in the dialog:
- `Enter` - Save (if validation passes)
- `Esc` - Cancel
- `Tab` - Navigate between fields

## Common Workflows

### Workflow 1: Create Template
1. Build a model structure (e.g., 3 Gaussians)
2. Don't set specific values yet
3. Save as "Three Peak Template"
4. Use this as starting point for new fits

### Workflow 2: Best Fit Model
1. Load data and fit with custom model
2. Get good fit results
3. Save the fitted model with current values
4. Save as "Sample_XYZ_Best_Fit"
5. Use for future samples with similar features

### Workflow 3: Share with Team
When sharing models with teammates, be aware there are two different kinds of saved YAMLs:

- **Composite (saved custom model)**: Configuration files describing a composite built from existing elements. These are stored by default in `models/custom_models/` and use the category `saved_custom_model`.
- **Atomic element (model element)**: Individual element definitions (e.g., `Gaussian`, `Linear Background`) are stored in `models/model_elements/` and are used when constructing composites.

Sharing steps:
1. Create and validate a model
2. Add clear description
3. Save to shared location
4. Team members copy composite YAMLs to their `models/custom_models/` and element YAMLs (if sharing new element types) to `models/model_elements/`
5. Everyone has consistent model for analysis

## Troubleshooting

### Model doesn't appear in selector
**Solution**: Restart BigFit to reload model elements

### Can't save to default location
**Solution**: Browse to a location where you have write permissions

### Parameters not preserved
**Solution**: Check that you set fixed/link states BEFORE saving

### Filename conflicts
**Solution**: Use more specific names or check existing files in models/model_elements/

## See Also
- [SAVE_CUSTOM_MODEL.md](SAVE_CUSTOM_MODEL.md) - Complete feature documentation
- [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Technical details
- Model elements in `models/model_elements/*.yaml` - Examples of YAML format
