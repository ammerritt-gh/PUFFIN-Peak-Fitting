#!/usr/bin/env python3
"""
Simple verification script to check that the implementation is correct.
This script doesn't require numpy or PySide6.
"""
import sys
from pathlib import Path

# Add BigFit to path (handle both running from repo root and from BigFit dir)
repo_root = Path(__file__).parent
bigfit_dir = repo_root / "BigFit"
if bigfit_dir.exists():
    sys.path.insert(0, str(bigfit_dir))
else:
    # Assume we're already in BigFit directory
    sys.path.insert(0, str(repo_root))

def verify_gaussian_spec():
    """Verify that GaussianModelSpec has the correct configuration."""
    print("=" * 70)
    print("VERIFICATION: Gaussian Model Interactive Controls Implementation")
    print("=" * 70)
    
    # Helper function to get input hint (handles both 'input_hint' and legacy 'input')
    def get_input_hint(param_spec):
        """Get input hint from parameter spec, checking both current and legacy names."""
        if not isinstance(param_spec, dict):
            return None
        return param_spec.get("input_hint") or param_spec.get("input")
    
    try:
        from models.model_specs import GaussianModelSpec
        
        print("\n✓ Successfully imported GaussianModelSpec")
        
        # Create instance
        spec = GaussianModelSpec()
        print("✓ Successfully created GaussianModelSpec instance")
        
        # Get parameters
        params = spec.get_parameters()
        print("✓ Successfully got parameters")
        
        # Check parameter names
        expected_params = ["Area", "Width", "Center"]
        actual_params = list(params.keys())
        
        print(f"\n  Expected parameters: {expected_params}")
        print(f"  Actual parameters: {actual_params}")
        
        for param in expected_params:
            if param not in actual_params:
                print(f"✗ FAIL: Missing parameter '{param}'")
                return False
        
        print("✓ All expected parameters present")
        
        # Check default values
        print("\nChecking default values:")
        checks = [
            ("Area", "value", 1.0),
            ("Width", "value", 1.0),
            ("Center", "value", 0.0),
        ]
        
        for param_name, key, expected in checks:
            param_spec = params.get(param_name, {})
            if not isinstance(param_spec, dict):
                print(f"✗ FAIL: {param_name} spec is not a dict")
                return False
                
            actual = param_spec.get(key)
            if actual == expected:
                print(f"  ✓ {param_name}.{key} = {actual}")
            else:
                print(f"  ✗ FAIL: {param_name}.{key} = {actual}, expected {expected}")
                return False
        
        # Check input_hint configurations
        print("\nChecking input_hint configurations:")
        
        # Area should have wheel control (no modifier)
        area_input = get_input_hint(params["Area"])
        if not area_input:
            print("  ✗ FAIL: Area has no input hint")
            return False
        if not isinstance(area_input, dict):
            print(f"  ✗ FAIL: Area input hint is not a dict: {area_input}")
            return False
        if "wheel" not in area_input:
            print(f"  ✗ FAIL: Area input hint has no 'wheel' key: {area_input}")
            return False
        
        wheel_spec = area_input["wheel"]
        if wheel_spec.get("action") != "scale":
            print(f"  ✗ FAIL: Area wheel action is '{wheel_spec.get('action')}', expected 'scale'")
            return False
        if wheel_spec.get("factor") != 1.1:
            print(f"  ✗ FAIL: Area wheel factor is {wheel_spec.get('factor')}, expected 1.1")
            return False
        
        print("  ✓ Area: wheel control configured (scale by 1.1)")
        
        # Width should have wheel control with Ctrl modifier
        width_input = get_input_hint(params["Width"])
        if not width_input:
            print("  ✗ FAIL: Width has no input hint")
            return False
        if "wheel" not in width_input:
            print(f"  ✗ FAIL: Width input hint has no 'wheel' key: {width_input}")
            return False
        
        width_wheel = width_input["wheel"]
        if width_wheel.get("action") != "scale":
            print(f"  ✗ FAIL: Width wheel action is '{width_wheel.get('action')}', expected 'scale'")
            return False
        if width_wheel.get("factor") != 1.05:
            print(f"  ✗ FAIL: Width wheel factor is {width_wheel.get('factor')}, expected 1.05")
            return False
        
        modifiers = width_wheel.get("modifiers", [])
        if "Ctrl" not in modifiers:
            print(f"  ✗ FAIL: Width wheel modifiers {modifiers} don't include 'Ctrl'")
            return False
        
        print("  ✓ Width: Ctrl+wheel control configured (scale by 1.05)")
        
        # Center should have drag control
        center_input = get_input_hint(params["Center"])
        if not center_input:
            print("  ✗ FAIL: Center has no input hint")
            return False
        if "drag" not in center_input:
            print(f"  ✗ FAIL: Center input hint has no 'drag' key: {center_input}")
            return False
        
        drag_spec = center_input["drag"]
        if drag_spec.get("action") != "set":
            print(f"  ✗ FAIL: Center drag action is '{drag_spec.get('action')}', expected 'set'")
            return False
        if drag_spec.get("value_from") != "x":
            print(f"  ✗ FAIL: Center drag value_from is '{drag_spec.get('value_from')}', expected 'x'")
            return False
        
        print("  ✓ Center: drag control configured (set from x)")
        
        print("\n" + "=" * 70)
        print("✓ ALL VERIFICATIONS PASSED")
        print("=" * 70)
        print("\nThe Gaussian model is correctly configured for interactive controls:")
        print("  - Area: Mouse wheel (no modifier) scales by 10%")
        print("  - Width: Ctrl+wheel scales by 5%")
        print("  - Center: Horizontal drag sets from x-coordinate")
        print("\nTo test interactively, run:")
        print("  cd BigFit")
        print("  python main.py")
        print("\nThen:")
        print("  1. Select 'Gaussian' model from Parameters panel")
        print("  2. Click near x=0 to select Center parameter")
        print("  3. Drag horizontally to move peak")
        print("  4. Scroll wheel to adjust Area")
        print("  5. Ctrl+scroll to adjust Width")
        print("  6. Press Spacebar to deselect")
        print("")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_gaussian_spec()
    sys.exit(0 if success else 1)
