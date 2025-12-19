# dataio/data_exporter.py
"""
Comprehensive data export module for saving data, fits, and parameters.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import io


def save_as_image(x_data, y_data, y_fit_dict, y_errors, save_path, margin_percent=10.0, 
                  excluded_mask=None, file_info=None, view_range=None):
    """Save the plot as an image file (PNG or PDF).
    
    Args:
        x_data: X data array
        y_data: Y data array
        y_fit_dict: Dictionary with fit data (can be dict with 'total' and 'components' or array)
        y_errors: Error bars array (can be None)
        save_path: Path for the image file (with .png or .pdf extension)
        margin_percent: Percentage margin beyond data range (default 10%, ignored if view_range is provided)
        excluded_mask: Boolean array marking excluded points
        file_info: Optional file info dict for title
        view_range: Optional tuple of ((x_min, x_max), (y_min, y_max)) to use instead of auto-calculating
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate included and excluded data
        if excluded_mask is not None and np.any(excluded_mask):
            incl_mask = ~excluded_mask
            x_incl = x_data[incl_mask]
            y_incl = y_data[incl_mask]
            x_excl = x_data[excluded_mask]
            y_excl = y_data[excluded_mask]
            
            # Plot excluded points as gray X marks
            ax.plot(x_excl, y_excl, 'x', color='gray', markersize=8, 
                   label='Excluded Data', alpha=0.6)
            
            # Plot included data
            if y_errors is not None:
                err_incl = y_errors[incl_mask] if len(y_errors) == len(y_data) else None
                if err_incl is not None:
                    ax.errorbar(x_incl, y_incl, yerr=err_incl, fmt='o', 
                              color='black', markersize=4, capsize=3, label='Data')
                else:
                    ax.plot(x_incl, y_incl, 'o', color='black', 
                           markersize=4, label='Data')
            else:
                ax.plot(x_incl, y_incl, 'o', color='black', 
                       markersize=4, label='Data')
        else:
            # No exclusions - plot all data
            if y_errors is not None:
                ax.errorbar(x_data, y_data, yerr=y_errors, fmt='o', 
                          color='black', markersize=4, capsize=3, label='Data')
            else:
                ax.plot(x_data, y_data, 'o', color='black', 
                       markersize=4, label='Data')
        
        # Plot fits
        if y_fit_dict is not None:
            if isinstance(y_fit_dict, dict):
                # Extract total fit
                total = y_fit_dict.get('total')
                if total is not None:
                    if isinstance(total, dict):
                        fit_x = total.get('x', x_data)
                        fit_y = total.get('y')
                    else:
                        fit_x = x_data
                        fit_y = total
                    
                    if fit_y is not None:
                        ax.plot(fit_x, fit_y, '-', color='purple', 
                               linewidth=2, label='Total Fit')
                
                # Extract component fits
                components = y_fit_dict.get('components', [])
                for comp in components:
                    if isinstance(comp, dict):
                        comp_x = comp.get('x', x_data)
                        comp_y = comp.get('y')
                        comp_label = comp.get('label', 'Component')
                        comp_color = comp.get('color', 'blue')
                        
                        if comp_y is not None:
                            ax.plot(comp_x, comp_y, '--', color=comp_color, 
                                   linewidth=1.5, label=comp_label, alpha=0.7)
            else:
                # Simple array fit
                ax.plot(x_data, y_fit_dict, '-', color='purple', 
                       linewidth=2, label='Fit')
        
        # Calculate axis limits
        if view_range is not None:
            # Use provided view range
            (x_min, x_max), (y_min, y_max) = view_range
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            # Calculate with margin based on data only (not fits)
            x_min, x_max = np.min(x_data), np.max(x_data)
            x_range = x_max - x_min
            x_margin = x_range * (margin_percent / 100.0)
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            
            y_min, y_max = np.min(y_data), np.max(y_data)
            y_range = y_max - y_min
            y_margin = y_range * (margin_percent / 100.0)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Labels and title
        ax.set_xlabel('Energy (meV)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        
        title = 'Data and Fit'
        if file_info and isinstance(file_info, dict):
            name = file_info.get('name')
            if name:
                title = f'{name}'
        ax.set_title(title, fontsize=14)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"Error saving image: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_as_ascii(x_data, y_data, y_fit_dict, y_errors, save_path, excluded_mask=None, delimiter=","):
    """Save data and fits as ASCII file with fine grid for fits.
    
    Format:
    - First 3 columns: Energy (data), Counts (data), Errors (data)
    - Following columns: X_fit (fine grid), Y_fit_total, Y_fit_comp1, Y_fit_comp2, ...
    
    Args:
        x_data: X data array
        y_data: Y data array
        y_fit_dict: Dictionary with fit data (can be dict with 'total' and 'components' or array)
        y_errors: Error bars array (can be None)
        save_path: Path for the ASCII file
        excluded_mask: Boolean array marking excluded points
        delimiter: Delimiter character ("," , "\t", or " ")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(save_path, 'w') as f:
            # Determine delimiter display for header
            delim_name = "tab" if delimiter == "\t" else ("space" if delimiter == " " else "comma")
            
            # Write header
            f.write("# BigFit Data Export\n")
            f.write(f"# Delimiter: {delim_name}\n")
            f.write("# \n")
            f.write("# Data columns (rows may be excluded from fit):\n")
            f.write(f"#   Energy{delimiter}Counts{delimiter}Errors\n")
            f.write("# \n")
            
            # Determine fit columns
            fit_columns = []
            if y_fit_dict is not None:
                if isinstance(y_fit_dict, dict):
                    total = y_fit_dict.get('total')
                    if total is not None:
                        fit_columns.append('Y_fit_total')
                    
                    components = y_fit_dict.get('components', [])
                    for i, comp in enumerate(components):
                        if isinstance(comp, dict):
                            label = comp.get('label', f'Component_{i+1}')
                            fit_columns.append(f'Y_fit_{label}')
                else:
                    fit_columns.append('Y_fit')
            
            if fit_columns:
                f.write("# Fit columns (evaluated on fine grid):\n")
                f.write(f"#   X_fit{delimiter}{delimiter.join(fit_columns)}\n")
                f.write("# \n")
            
            f.write("# ===== DATA SECTION =====\n")
            
            # Write data section
            for i in range(len(x_data)):
                x_val = x_data[i]
                y_val = y_data[i]
                err_val = y_errors[i] if y_errors is not None else 0.0
                
                # Mark excluded points in comment
                if excluded_mask is not None and i < len(excluded_mask) and excluded_mask[i]:
                    f.write(f"# {x_val:.6f}{delimiter}{y_val:.6e}{delimiter}{err_val:.6e}\n")
                else:
                    f.write(f"{x_val:.6f}{delimiter}{y_val:.6e}{delimiter}{err_val:.6e}\n")
            
            # Write fit section if available
            if y_fit_dict is not None and fit_columns:
                f.write("\n# ===== FIT SECTION =====\n")
                
                # Extract fit data on fine grid
                if isinstance(y_fit_dict, dict):
                    total = y_fit_dict.get('total')
                    components = y_fit_dict.get('components', [])
                    
                    # Get x array for fits (use fine grid if available)
                    if isinstance(total, dict) and 'x' in total:
                        fit_x = np.array(total['x'])
                    else:
                        # Create fine grid spanning data range
                        fit_x = np.linspace(np.min(x_data), np.max(x_data), 500)
                    
                    # Collect all fit y arrays
                    fit_y_arrays = []
                    
                    # Total fit
                    if isinstance(total, dict):
                        fit_y_arrays.append(np.array(total.get('y', [])))
                    elif total is not None:
                        fit_y_arrays.append(np.array(total))
                    
                    # Component fits
                    for comp in components:
                        if isinstance(comp, dict):
                            comp_y = comp.get('y', [])
                            fit_y_arrays.append(np.array(comp_y))
                    
                    # Write fit data
                    n_points = len(fit_x)
                    for i in range(n_points):
                        line = f"{fit_x[i]:.6f}"
                        for fit_y in fit_y_arrays:
                            if i < len(fit_y):
                                line += f"{delimiter}{fit_y[i]:.6e}"
                            else:
                                line += delimiter
                        f.write(line + "\n")
                else:
                    # Simple array fit
                    for i in range(len(x_data)):
                        f.write(f"{x_data[i]:.6f}{delimiter}{y_fit_dict[i]:.6e}\n")
        
        return True
    
    except Exception as e:
        print(f"Error saving ASCII: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_parameters(parameters, fit_result, save_path, model_name=None, delimiter=","):
    """Save fit parameters with values and errors.
    
    Args:
        parameters: Dict of parameter specs from viewmodel.get_parameters()
        fit_result: Fit result dict with parameter values and errors (if available)
        save_path: Path for the parameters file
        model_name: Optional model name for header
        delimiter: Delimiter character ("," , "\t", or " ")
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(save_path, 'w') as f:
            # Determine delimiter display for header
            delim_name = "tab" if delimiter == "\t" else ("space" if delimiter == " " else "comma")
            
            # Write header
            f.write("# BigFit Parameter Export\n")
            if model_name:
                f.write(f"# Model: {model_name}\n")
            f.write(f"# Delimiter: {delim_name}\n")
            f.write("# \n")
            f.write("# Columns:\n")
            f.write(f"#   Parameter{delimiter}Value{delimiter}Error{delimiter}Fixed{delimiter}Min{delimiter}Max\n")
            f.write("# \n")
            f.write("# Error values are standard errors calculated from the covariance matrix (Jacobian)\n")
            f.write("# of the fit. 'N/A' indicates the parameter was fixed or error unavailable.\n")
            f.write("# \n")
            
            # Extract errors from fit result if available
            errors = {}
            if fit_result is not None and isinstance(fit_result, dict):
                # Look for 'perr' key (dict of parameter name -> error)
                perr_dict = fit_result.get('perr')
                if perr_dict is not None and isinstance(perr_dict, dict):
                    errors = perr_dict
                elif 'errors' in fit_result:
                    # Fallback to 'errors' key
                    errors_data = fit_result['errors']
                    if isinstance(errors_data, dict):
                        errors = errors_data
                elif 'perr' in fit_result:
                    # Legacy: perr might be an array
                    perr = fit_result.get('perr')
                    if perr is not None and not isinstance(perr, dict):
                        # Try to match perr array with free parameter names
                        try:
                            param_names = [k for k, v in parameters.items() 
                                         if not v.get('fixed', False)]
                            if len(perr) == len(param_names):
                                errors = dict(zip(param_names, perr))
                        except Exception:
                            pass
            
            # Write parameters
            for name, spec in parameters.items():
                value = spec.get('value', 0.0)
                fixed = spec.get('fixed', False)
                min_val = spec.get('min', '')
                max_val = spec.get('max', '')
                error = errors.get(name, '')
                
                # Format values
                value_str = f"{value:.6e}" if value is not None else "N/A"
                error_str = f"{error:.6e}" if error not in ('', None) and error == error else "N/A"  # Check for NaN
                fixed_str = "Yes" if fixed else "No"
                min_str = f"{min_val:.6e}" if min_val not in ('', None) else "N/A"
                max_str = f"{max_val:.6e}" if max_val not in ('', None) else "N/A"
                
                f.write(f"{name}{delimiter}{value_str}{delimiter}{error_str}{delimiter}{fixed_str}{delimiter}{min_str}{delimiter}{max_str}\n")
        
        return True
    
    except Exception as e:
        print(f"Error saving parameters: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_all(x_data, y_data, y_fit_dict, y_errors, parameters, fit_result, 
               base_path, margin_percent=10.0, excluded_mask=None, file_info=None, 
               model_name=None, delimiter=","):
    """Export all data, fits, and parameters.
    
    Args:
        x_data: X data array
        y_data: Y data array
        y_fit_dict: Dictionary with fit data
        y_errors: Error bars array
        parameters: Parameter specs dict
        fit_result: Fit result dict
        base_path: Base path for files (extensions will be added)
        margin_percent: Image margin percentage
        excluded_mask: Boolean exclusion mask
        file_info: File info dict
        model_name: Model name for parameter file
        delimiter: Delimiter character for text files
    
    Returns:
        Dict with success status for each export type
    """
    results = {
        'image': False,
        'ascii': False,
        'parameters': False,
    }
    
    # Save image
    image_path = f"{base_path}.png"
    results['image'] = save_as_image(
        x_data, y_data, y_fit_dict, y_errors, image_path, 
        margin_percent, excluded_mask, file_info
    )
    
    # Save ASCII
    ascii_path = f"{base_path}_data.txt"
    results['ascii'] = save_as_ascii(
        x_data, y_data, y_fit_dict, y_errors, ascii_path, excluded_mask, delimiter
    )
    
    # Save parameters
    params_path = f"{base_path}_parameters.txt"
    results['parameters'] = save_parameters(
        parameters, fit_result, params_path, model_name, delimiter
    )
    
    return results
