#!/usr/bin/env python3
"""
Test script for instrument configuration system.
Tests loading, parsing, and basic functionality without requiring GUI.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def test_yaml_loading():
    """Test that YAML files can be loaded."""
    import yaml
    
    print("=" * 60)
    print("Testing YAML file loading...")
    print("=" * 60)
    
    yaml_files = [
        'config/instruments/example_spectrometer.yaml',
        'config/instruments/simple_spectrometer.yaml'
    ]
    
    for yaml_file in yaml_files:
        print(f"\nLoading: {yaml_file}")
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            print(f"  ✓ Name: {data['name']}")
            print(f"  ✓ Type: {data['type']}")
            
            if 'slits' in data and 'positions' in data['slits']:
                print(f"  ✓ Slits: {len(data['slits']['positions'])} positions")
            
            if 'collimators' in data and 'positions' in data['collimators']:
                print(f"  ✓ Collimators: {len(data['collimators']['positions'])} positions")
    
    print("\n✓ YAML loading test passed!\n")
    return True

def test_instrument_config_module():
    """Test the instrument_config module."""
    print("=" * 60)
    print("Testing instrument_config module...")
    print("=" * 60)
    
    print("\n  Loading instrument_config module directly...")
    
    # Import the module directly without going through dataio.__init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "instrument_config",
        parent_dir / "dataio" / "instrument_config.py"
    )
    instrument_config = importlib.util.module_from_spec(spec)
    sys.modules['instrument_config'] = instrument_config
    spec.loader.exec_module(instrument_config)
    
    print("  ✓ Module imports successful")
    
    # Test listing available instruments
    print("\n  Testing list_available_instruments()...")
    instruments = instrument_config.list_available_instruments()
    print(f"  ✓ Found {len(instruments)} instruments")
    for inst in instruments:
        print(f"    - {inst['name']}")
    
    # Test loading specific instrument
    print("\n  Testing load_instrument_config()...")
    config = instrument_config.load_instrument_config(parent_dir / 'config' / 'instruments' / 'example_spectrometer.yaml')
    print(f"  ✓ Loaded: {config.name}")
    print(f"    Type: {config.type}")
    print(f"    Description: {config.description}")
    print(f"    Arm lengths: {len(config.arm_lengths)}")
    print(f"    Monochromator crystals: {len(config.get_monochromator_crystals())}")
    print(f"    Analyzer crystals: {len(config.get_analyzer_crystals())}")
    print(f"    Collimators: {len(config.collimators)}")
    print(f"    Slits: {len(config.slits)}")
    print(f"    Modules: {len(config.modules)}")
    
    # Test slit details
    print("\n  Slit positions:")
    for slit in config.slits:
        print(f"    - {slit.label}: {len(slit.dimensions)} dimensions")
        for dim in slit.dimensions:
            print(f"      - {dim.label}: {dim.default} mm (range {dim.min}-{dim.max})")
    
    # Test collimator details
    print("\n  Collimator positions:")
    for coll in config.collimators:
        print(f"    - {coll.label}: options = {coll.options}")
    
    # Test crystal details
    print("\n  Crystals:")
    for crystal in config.get_monochromator_crystals():
        print(f"    - Mono: {crystal.name} (d={crystal.d_spacing} Å)")
    for crystal in config.get_analyzer_crystals():
        print(f"    - Analyzer: {crystal.name} (d={crystal.d_spacing} Å)")
    
    print("\n✓ instrument_config module test passed!\n")
    return True

def test_viewmodel_integration():
    """Test that viewmodel methods exist (without actually running them)."""
    print("=" * 60)
    print("Testing viewmodel integration...")
    print("=" * 60)
    
    # Check that the methods exist in the source code
    with open('viewmodel/fitter_vm.py', 'r') as f:
        content = f.read()
        
        methods = [
            'load_instrument_from_name',
            '_initialize_instrument_state',
            'get_instrument_config',
            'get_instrument_state',
            'set_slit_value',
            'set_collimator_value',
            'set_crystal',
            'set_focusing',
            'set_module_enabled',
            'set_module_parameter',
        ]
        
        print("\n  Checking for required methods:")
        for method in methods:
            if f'def {method}' in content:
                print(f"    ✓ {method}")
            else:
                print(f"    ✗ {method} NOT FOUND!")
                return False
    
    print("\n✓ Viewmodel integration test passed!\n")
    return True

def test_dock_widget():
    """Test that dock widget exists and has required structure."""
    print("=" * 60)
    print("Testing instrument dock widget...")
    print("=" * 60)
    
    with open('view/docks/instrument_dock.py', 'r') as f:
        content = f.read()
        
        # Check for class definition
        if 'class InstrumentDock' not in content:
            print("  ✗ InstrumentDock class not found!")
            return False
        print("  ✓ InstrumentDock class found")
        
        # Check for signals
        signals = [
            'instrument_selected',
            'slit_changed',
            'collimator_changed',
            'crystal_changed',
            'focusing_changed',
            'module_enabled_changed',
            'module_parameter_changed',
        ]
        
        print("\n  Checking for required signals:")
        for signal in signals:
            if f'{signal} = Signal' in content:
                print(f"    ✓ {signal}")
            else:
                print(f"    ✗ {signal} NOT FOUND!")
                return False
        
        # Check for key methods
        methods = [
            'populate_instruments',
            'set_instrument_config',
            '_rebuild_controls',
            '_build_slit_section',
            '_build_collimator_section',
            '_build_crystal_section',
        ]
        
        print("\n  Checking for required methods:")
        for method in methods:
            if f'def {method}' in content:
                print(f"    ✓ {method}")
            else:
                print(f"    ✗ {method} NOT FOUND!")
                return False
    
    print("\n✓ Dock widget test passed!\n")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INSTRUMENT CONFIGURATION SYSTEM TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("YAML Loading", test_yaml_loading),
        ("Instrument Config Module", test_instrument_config_module),
        ("Viewmodel Integration", test_viewmodel_integration),
        ("Dock Widget", test_dock_widget),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
