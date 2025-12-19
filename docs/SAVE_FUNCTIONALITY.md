# BigFit Save Functionality

## Overview

The BigFit application now provides comprehensive data export options through an enhanced "Save Data" dialog. When you click the "Save Data" button, you can choose from four different save modes to export your data, fits, and parameters in various formats.

## Save Modes

### 1. Save Everything (Default)
Exports all three file types with appropriate suffixes:
- **Image file** (`basename.png`): Plot with data, fits, and components
- **ASCII data file** (`basename_data.txt`): Data and fit curves
- **Parameters file** (`basename_parameters.txt`): Fit parameters with errors

### 2. Save as Image Only
Exports a high-quality PNG image of the current plot showing:
- Data points with error bars
- Excluded data points (marked with gray X symbols)
- Total fit curve (purple line)
- Component fits (dashed lines, if composite model)
- Configurable margin around the data (default: 10%)

**Image Options:**
- **Margin**: Percentage of data range to add as whitespace around the plot
  - Default: 10%
  - Range: 0% - 50%
  - Note: Margin is calculated relative to the data range, NOT the fit range
  - This prevents fits that extend beyond data from affecting the view

### 3. Save as ASCII Only
Exports a tab-separated text file with two sections:

**Data Section** (3 columns):
```
Energy    Counts    Errors
```
- Excluded points are commented out with `#`
- All data points are included for reference

**Fit Section** (variable columns):
```
X_fit    Y_fit_total    Y_fit_Component1    Y_fit_Component2    ...
```
- Evaluated on a fine grid (typically 500 points) for smooth plotting
- Includes total fit and all component fits
- Grid spans the data range

**Use Cases:**
- Import into other plotting programs (Origin, Igor, Excel, etc.)
- Share results with collaborators
- Archive data with fits for future reference

### 4. Save Parameters Only
Exports a tab-separated text file with fit parameter information:

**Columns:**
- **Parameter**: Parameter name
- **Value**: Current or fitted value
- **Error**: Standard error from covariance matrix (Jacobian)
- **Fixed**: Yes/No indicating if parameter was held constant
- **Min**: Lower bound for fitting (N/A if unbounded)
- **Max**: Upper bound for fitting (N/A if unbounded)

**Error Calculation:**
- Errors are calculated from the covariance matrix returned by `scipy.optimize.curve_fit`
- Represents the standard error (1σ uncertainty) for each parameter
- Fixed parameters show "N/A" for errors
- Errors are only available after a successful fit

## Configuration

### Default Save Folder
The default save folder can be configured through:
1. Click the "Edit Config" button in the Controls panel
2. Set the "Default Save Folder" path
3. Click "Save"

This folder will be used as the starting location in the save dialog.

## File Formats

### Image Format
- **Format**: PNG (Portable Network Graphics)
- **Resolution**: 300 DPI
- **Size**: 10" × 6" (3000 × 1800 pixels)
- **Quality**: High quality with tight bounding box

### ASCII Format
- **Encoding**: UTF-8
- **Separator**: Tab (`\t`)
- **Decimals**: 6 significant figures for X values, scientific notation for Y values
- **Comments**: Lines starting with `#` contain metadata

### Parameters Format
- **Encoding**: UTF-8
- **Separator**: Tab (`\t`)
- **Decimals**: 6 significant figures (scientific notation)
- **Header**: Contains model name and column descriptions

## Usage Examples

### Example 1: Export for Publication
1. Fit your data
2. Click "Save Data"
3. Select "Save as Image Only"
4. Set margin to 5% for a tighter view
5. Choose save location
6. Import the PNG into your manuscript

### Example 2: Share Results
1. Fit your data
2. Click "Save Data"
3. Select "Save Everything"
4. Choose a descriptive base name (e.g., "sample_XY_fit")
5. Share all three files with collaborators:
   - Image for quick visual inspection
   - ASCII file for replotting/analysis
   - Parameters file for documentation

### Example 3: Archive Fit Parameters
1. After fitting multiple datasets
2. Click "Save Data" for each
3. Select "Save Parameters Only"
4. Use sequential names (e.g., "run_01_params", "run_02_params", ...)
5. Compile parameters into a spreadsheet for comparison

## Technical Details

### Fit Grid Resolution
- Data is plotted on the original X points
- Fits are evaluated on a fine grid for smooth curves
- Grid size adapts to data density (400-8000 points)
- When resolution convolution is active, grid is extended with padding

### Error Calculation
Parameter errors are calculated using the covariance matrix from `scipy.optimize.curve_fit`:
```
error = sqrt(diag(covariance_matrix))
```

This gives the standard error (1σ) for each parameter, assuming:
- The model is correct
- Errors are Gaussian distributed
- No correlation between errors in different data points

### Margin Calculation
For image export with M% margin:
```
x_range = max(x_data) - min(x_data)
x_margin = x_range * (M / 100)
x_limits = [min(x_data) - x_margin, max(x_data) + x_margin]
```

This ensures the margin is consistent regardless of how far the fit extends.

## Troubleshooting

### No errors in parameter file
- Errors are only available after fitting
- Run the fit before saving parameters
- Check that the fit converged successfully

### Image looks cut off
- Increase the margin percentage
- Default 10% should work for most cases
- Try 20% if components extend significantly beyond data

### ASCII file is very large
- Normal for fine-grid fits (500 points per component)
- Reduces file size by selecting "Save Parameters Only"
- Or import into plotting software which can handle large files

### File locations
All files are saved to the location you select in the dialog:
- Click "Browse..." to choose a location
- The default location is set in the configuration
- Base name is used for all files with appropriate suffixes

## Future Enhancements

Potential improvements for future versions:
- PDF export option for vector graphics
- Multiple image format support (SVG, EPS)
- Customizable image dimensions
- Batch export for multiple datasets
- Export fit statistics (chi-squared, R², etc.)
