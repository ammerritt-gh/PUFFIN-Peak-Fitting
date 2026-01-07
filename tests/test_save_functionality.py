#!/usr/bin/env python
"""
Test script for the new save functionality.
Tests the data export functions without requiring the GUI.
"""
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataio.data_exporter import save_as_image, save_as_ascii, save_parameters


def test_ascii_export():
    """Test ASCII export with data and fits."""
    print("Testing ASCII export...")
    
    # Create test data
    x_data = np.linspace(0, 10, 50)
    y_data = np.sin(x_data) + 0.1 * np.random.randn(50)
    y_errors = np.ones_like(y_data) * 0.1
    
    # Create test fit
    x_fit = np.linspace(0, 10, 200)
    y_fit_total = np.sin(x_fit)
    y_fit_comp1 = 0.5 * np.sin(x_fit)
    y_fit_comp2 = 0.5 * np.sin(x_fit)
    
    y_fit_dict = {
        'total': {'x': x_fit, 'y': y_fit_total},
        'components': [
            {'x': x_fit, 'y': y_fit_comp1, 'label': 'Component1', 'color': 'blue'},
            {'x': x_fit, 'y': y_fit_comp2, 'label': 'Component2', 'color': 'red'},
        ]
    }
    
    # Create exclusion mask
    excluded_mask = np.zeros_like(x_data, dtype=bool)
    excluded_mask[10:15] = True
    
    # Export
    save_path = '/tmp/test_ascii_export.txt'
    success = save_as_ascii(x_data, y_data, y_fit_dict, y_errors, save_path, excluded_mask)
    
    if success and os.path.exists(save_path):
        print(f"✓ ASCII export successful: {save_path}")
        # Show first few lines
        with open(save_path, 'r') as f:
            lines = f.readlines()[:20]
            print("  First 20 lines:")
            for line in lines:
                print(f"    {line.rstrip()}")
        return True
    else:
        print("✗ ASCII export failed")
        return False


def test_parameter_export():
    """Test parameter export with errors."""
    print("\nTesting parameter export...")
    
    # Create test parameters
    parameters = {
        'amplitude': {'value': 1.5, 'fixed': False, 'min': 0.0, 'max': 10.0},
        'center': {'value': 5.2, 'fixed': False, 'min': None, 'max': None},
        'width': {'value': 0.8, 'fixed': True, 'min': 0.1, 'max': 5.0},
    }
    
    # Create test fit result with errors
    fit_result = {
        'amplitude': 1.52,
        'center': 5.18,
        'width': 0.8,
        'perr': {
            'amplitude': 0.03,
            'center': 0.05,
            'width': None,  # Fixed parameter
        }
    }
    
    # Export
    save_path = '/tmp/test_parameters.txt'
    success = save_parameters(parameters, fit_result, save_path, model_name='TestModel')
    
    if success and os.path.exists(save_path):
        print(f"✓ Parameter export successful: {save_path}")
        # Show content
        with open(save_path, 'r') as f:
            content = f.read()
            print("  Content:")
            for line in content.split('\n'):
                print(f"    {line}")
        return True
    else:
        print("✗ Parameter export failed")
        return False


def test_image_export():
    """Test image export."""
    print("\nTesting image export...")
    
    # Create test data
    x_data = np.linspace(0, 10, 50)
    y_data = np.sin(x_data) + 0.1 * np.random.randn(50)
    y_errors = np.ones_like(y_data) * 0.1
    
    # Create test fit
    x_fit = np.linspace(0, 10, 200)
    y_fit_total = np.sin(x_fit)
    
    y_fit_dict = {
        'total': {'x': x_fit, 'y': y_fit_total},
        'components': []
    }
    
    # Export with 10% margin
    save_path = '/tmp/test_plot.png'
    success = save_as_image(x_data, y_data, y_fit_dict, y_errors, save_path, 
                           margin_percent=10.0, excluded_mask=None, file_info={'name': 'Test Data'})
    
    if success and os.path.exists(save_path):
        print(f"✓ Image export successful: {save_path}")
        # Check file size
        size = os.path.getsize(save_path)
        print(f"  File size: {size} bytes")
        return True
    else:
        print("✗ Image export failed")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PUFFIN Save Functionality Tests")
    print("=" * 60)
    
    results = []
    results.append(("ASCII Export", test_ascii_export()))
    results.append(("Parameter Export", test_parameter_export()))
    results.append(("Image Export", test_image_export()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
