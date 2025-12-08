"""
End-to-end test for the save custom model feature.

This test verifies that:
1. A composite model can be created
2. Parameters can be set with fixed/linked states
3. The model can be saved to YAML
4. The saved model can be reloaded
5. All parameter properties are preserved (values, fixed, link_groups, bounds)
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CompositeModelSpec, get_model_spec, reload_model_elements


def test_save_and_load_custom_model():
    """Test the full save/load cycle for a custom composite model."""
    print("=" * 70)
    print("Testing Save Custom Model Feature")
    print("=" * 70)
    
    # Step 1: Create a composite model
    print("\n1. Creating composite model with multiple components...")
    model_spec = CompositeModelSpec()
    
    # Add first component: Gaussian with fixed Area and linked Width
    comp1 = model_spec.add_component('Gaussian', prefix='peak1_')
    comp1.spec.params['Area'].value = 100.0
    comp1.spec.params['Area'].fixed = True
    comp1.spec.params['Area'].min = 0.0
    comp1.spec.params['Area'].max = 1000.0
    comp1.spec.params['Width'].value = 2.5
    comp1.spec.params['Width'].link_group = 1
    comp1.spec.params['Width'].min = 0.1
    comp1.spec.params['Center'].value = 0.0
    
    # Add second component: Voigt with linked Gauss FWHM and fixed Center
    comp2 = model_spec.add_component('Voigt', prefix='peak2_')
    comp2.spec.params['Area'].value = 50.0
    comp2.spec.params['Area'].min = 0.0
    comp2.spec.params['Gauss FWHM'].value = 1.5
    comp2.spec.params['Gauss FWHM'].link_group = 1  # Linked with peak1 Width
    comp2.spec.params['Lorentz FWHM'].value = 0.5
    comp2.spec.params['Center'].value = 5.0
    comp2.spec.params['Center'].fixed = True
    
    # Add third component: Linear background
    comp3 = model_spec.add_component('Linear Background', prefix='bg_')
    comp3.spec.params['Slope'].value = 0.1
    comp3.spec.params['Intercept'].value = 10.0
    
    # Rebuild flat params to propagate changes
    model_spec._rebuild_flat_params()
    
    print("   ✓ Created 3 components (2 peaks + background)")
    print(f"   ✓ Set fixed parameters: peak1_Area, peak2_Center")
    print(f"   ✓ Linked parameters: peak1_Width and peak2_Gauss FWHM (group 1)")
    
    # Step 2: Simulate the save process
    print("\n2. Extracting model data for save...")
    components_list = []
    for component in model_spec.list_components():
        comp_data = {
            'type': component.spec.__class__.__name__.replace('ModelSpec', ''),
            'prefix': component.prefix,
            'label': component.label,
            'color': component.color,
            'parameters': {}
        }
        
        for name, param in component.spec.params.items():
            param_data = {
                'value': getattr(param, 'value', None),
                'fixed': bool(getattr(param, 'fixed', False)),
                'link_group': getattr(param, 'link_group', None),
                'min': getattr(param, 'min', None),
                'max': getattr(param, 'max', None),
            }
            comp_data['parameters'][name] = param_data
        
        components_list.append(comp_data)
    
    print(f"   ✓ Extracted {len(components_list)} components")
    
    # Step 3: Build and save YAML
    print("\n3. Building YAML structure...")
    import yaml
    
    yaml_data = {
        'name': 'Two Peaks with Background',
        'description': 'Custom model with two peaks (Gaussian + Voigt) and a linear background',
        'version': 1,
        'author': 'Test User',
        'category': 'composite',
        'is_composite': True,
        'components': []
    }
    
    for comp in components_list:
        component_def = {
            'element': comp.get('type', 'Unknown'),
            'prefix': comp.get('prefix', ''),
            'default_parameters': {}
        }
        
        params = comp.get('parameters', {})
        for param_name, param_info in params.items():
            param_def = {'value': param_info.get('value')}
            
            if param_info.get('fixed'):
                param_def['fixed'] = True
            
            link_group = param_info.get('link_group')
            if link_group is not None and link_group != 0:
                param_def['link_group'] = link_group
            
            if param_info.get('min') is not None:
                param_def['min'] = param_info['min']
            if param_info.get('max') is not None:
                param_def['max'] = param_info['max']
            
            component_def['default_parameters'][param_name] = param_def
        
        yaml_data['components'].append(component_def)
    
    # Save to a temporary file in models/model_elements/
    models_dir = Path(__file__).parent.parent / 'models' / 'model_elements'
    test_file = models_dir / 'two_peaks_with_background.yaml'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False, 
                 allow_unicode=True, indent=2)
    
    print(f"   ✓ Saved to: {test_file.name}")
    
    # Step 4: Reload model elements
    print("\n4. Reloading model elements...")
    reload_model_elements()
    print("   ✓ Model elements reloaded")
    
    # Step 5: Load the saved model
    print("\n5. Loading saved model...")
    loaded_model = get_model_spec('Two Peaks with Background')
    print(f"   ✓ Loaded: {loaded_model.__class__.__name__}")
    print(f"   ✓ Is composite: {getattr(loaded_model, 'is_composite', False)}")
    print(f"   ✓ Components: {len(loaded_model.list_components())}")
    
    # Step 6: Verify all properties
    print("\n6. Verifying parameter properties...")
    params = loaded_model.get_parameters()
    
    # Check fixed parameters
    assert params['peak1_Area']['fixed'] == True, "peak1_Area should be fixed"
    assert params['peak2_Center']['fixed'] == True, "peak2_Center should be fixed"
    print("   ✓ Fixed parameters preserved")
    
    # Check link groups
    assert params['peak1_Width']['link_group'] == 1, "peak1_Width should be in link group 1"
    assert params['peak2_Gauss FWHM']['link_group'] == 1, "peak2_Gauss FWHM should be in link group 1"
    print("   ✓ Link groups preserved")
    
    # Check bounds
    assert params['peak1_Area']['min'] == 0.0, "peak1_Area min should be 0.0"
    assert params['peak1_Area']['max'] == 1000.0, "peak1_Area max should be 1000.0"
    assert params['peak1_Width']['min'] == 0.1, "peak1_Width min should be 0.1"
    print("   ✓ Parameter bounds preserved")
    
    # Check values
    assert abs(params['peak1_Area']['value'] - 100.0) < 0.01, "peak1_Area value should be 100.0"
    assert abs(params['peak1_Width']['value'] - 2.5) < 0.01, "peak1_Width value should be 2.5"
    assert abs(params['peak2_Center']['value'] - 5.0) < 0.01, "peak2_Center value should be 5.0"
    print("   ✓ Parameter values preserved")
    
    # Step 7: Test that the model appears in available models
    print("\n7. Checking model availability...")
    from models import get_available_model_names
    available = get_available_model_names()
    assert 'TwoPeakswithBackgroundModelSpec' in available, "Saved model should appear in available models"
    print(f"   ✓ Model appears in available models list")
    
    # Step 8: Clean up
    print("\n8. Cleaning up...")
    test_file.unlink()
    reload_model_elements()  # Reload to remove the test model
    print("   ✓ Test file removed")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFeature Summary:")
    print("  • Custom composite models can be saved to YAML files")
    print("  • Fixed parameter states are preserved")
    print("  • Parameter linking (link groups) is preserved")
    print("  • Parameter bounds (min/max) are preserved")
    print("  • Saved models automatically appear in the model selector")
    print("  • Models can be loaded and used like built-in models")
    print("=" * 70)


if __name__ == '__main__':
    test_save_and_load_custom_model()
