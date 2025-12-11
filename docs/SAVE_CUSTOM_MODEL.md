# Save Custom Model Feature

## Overview

The Save Custom Model feature allows you to save your custom composite models as reusable YAML files. Once saved, these models can be loaded just like built-in models and will appear in the model selector dropdown.

## Features

- **Save composite models** with all their components and parameters
- **Preserve parameter states** including:
  - Fixed/unfixed state
  - Parameter linking (link groups)
  - Parameter bounds (min/max)
  - Parameter values
- **Automatic model discovery** - saved models appear in the model selector
- **YAML format** - human-readable and easy to share

## Usage

### Saving a Custom Model

1. **Create a custom composite model**:
   - Select "Custom Model" from the model selector
   - Add components using the "Add Element" button in the Elements dock
   - Configure parameters in the Parameters dock
   - Set parameters as fixed using the "F" button
   - Link parameters by assigning them the same link group number

2. **Click "Save Model..."** in the Elements dock:
   - This opens the Save Model dialog

3. **In the Save Model dialog**:
   - **Model Name**: Enter a descriptive name (e.g., "Two Gaussian Peaks")
   - **Description** (optional): Add details about the model
   - **Review the model structure**: The dialog shows all components and parameters with their current states
   - **Save Location**: By default, models are saved to `models/model_elements/`
   - Click **Save** to save the model

4. **The model is now available**:
   - The saved model will appear in the model selector
   - You can load it like any built-in model
   - The model will be available every time you start BigFit

### Loading a Saved Model

1. Open the model selector dropdown in the Parameters dock
2. Find your saved model in the list
3. Select it - the model will load with all components and parameter states

### Example Workflow

#### Create and Save a Two-Peak Model

```
1. Start with "Custom Model"
2. Add two Gaussian components:
   - peak1_: First peak
   - peak2_: Second peak
3. Configure parameters:
   - Set peak1_Area = 100, fixed
   - Set peak1_Width = 2.5, link_group = 1
   - Set peak2_Area = 50
   - Set peak2_Width = 2.5, link_group = 1 (links with peak1)
   - Set peak2_Center = 5.0, fixed
4. Click "Save Model..."
5. Name it "Two Linked Gaussians"
6. Save
```

The model is now saved and can be loaded anytime. The widths of both peaks will be linked (they'll change together during fitting), and the specified parameters will remain fixed.

## File Format

Saved models are stored as YAML files with the following structure:

```yaml
name: My Custom Model
description: A custom model description
version: 1
author: Your Name
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
      Center:
        value: 0.0
  - element: Voigt
    prefix: peak2_
    default_parameters:
      Area:
        value: 50.0
      Gauss FWHM:
        value: 1.5
        link_group: 1
      Lorentz FWHM:
        value: 0.5
      Center:
        value: 5.0
        fixed: true
```

## Parameter Properties

### Fixed Parameters

Parameters can be marked as `fixed: true` to prevent them from being modified during fitting. This is useful for:
- Known values (e.g., laser line position)
- Constraints (e.g., background level)
- Reducing fit degrees of freedom

### Linked Parameters

Parameters with the same `link_group` number are linked together. During fitting:
- They share the same value
- Only one free parameter is used for all linked parameters
- Useful for enforcing physical constraints (e.g., same resolution across all peaks)

### Parameter Bounds

Set `min` and `max` to constrain parameter values:
```yaml
Area:
  value: 100.0
  min: 0.0
  max: 1000.0
```

## Tips

1. **Organize your models**: Give them descriptive names and descriptions
2. **Test before saving**: Make sure your model works as expected before saving
3. **Share models**: YAML files can be shared with colleagues
4. **Version control**: Keep your custom models in version control
5. **Backup**: Save important models to a safe location

## Technical Details

- Models are saved to `models/model_elements/` by default
- File names are derived from the model name (lowercased, underscored)
- Models are automatically discovered on startup
- The system uses the same loader as built-in model elements

## Troubleshooting

### Model doesn't appear in selector
- Check that the file is saved in `models/model_elements/`
- Restart BigFit to reload model elements
- Check that the YAML syntax is valid

### Parameters not preserved correctly
- Ensure you set fixed/link_group states before saving
- Check the saved YAML file to verify the structure

### Can't save model
- Make sure you have write permissions to the save location
- Check that the model name is valid (alphanumeric characters only)
- Ensure you're working with a Custom Model (composite model)

## See Also

- [Model Elements Documentation](../models/model_elements/README.md)
- [Parameter Linking Guide](PARAMETER_LINKING.md)
- [Custom Model Tutorial](CUSTOM_MODEL_TUTORIAL.md)
