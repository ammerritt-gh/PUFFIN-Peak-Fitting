#!/usr/bin/env python3
"""
Test script to verify Gaussian peak interaction functionality.

This script tests:
1. Default Gaussian model with area=1, width=1, center=0
2. Parameter specs include proper input_hint configurations
3. The interaction flow is properly set up
"""
import sys
import numpy as np
from pathlib import Path

# Add BigFit to path (handle both running from repo root and from BigFit dir)
repo_root = Path(__file__).parent
bigfit_dir = repo_root / "BigFit"
if bigfit_dir.exists():
    sys.path.insert(0, str(bigfit_dir))
else:
    # Assume we're already in BigFit directory
    sys.path.insert(0, str(repo_root))

def test_gaussian_defaults():
    """Test that GaussianModelSpec has correct default values."""
    from models.model_specs import GaussianModelSpec
    
    print("=" * 60)
    print("Test 1: Gaussian Default Values")
    print("=" * 60)
    
    spec = GaussianModelSpec()
    params = spec.get_parameters()
    
    print("\nParameter defaults:")
    for name, param_spec in params.items():
        value = param_spec.get("value")
        print(f"  {name}: {value}")
    
    # Verify defaults
    assert params["Area"]["value"] == 1.0, "Area should default to 1.0"
    assert params["Width"]["value"] == 1.0, "Width should default to 1.0"
    assert params["Center"]["value"] == 0.0, "Center should default to 0.0"
    
    print("\n✓ All defaults are correct!")
    return True

def test_input_hints():
    """Test that input_hint specifications are properly configured."""
    from models.model_specs import GaussianModelSpec
    
    print("\n" + "=" * 60)
    print("Test 2: Input Hint Specifications")
    print("=" * 60)
    
    spec = GaussianModelSpec()
    params = spec.get_parameters()
    
    print("\nParameter input hints:")
    for name, param_spec in params.items():
        input_hint = param_spec.get("input")
        print(f"\n  {name}:")
        if input_hint:
            if isinstance(input_hint, dict):
                for event_type, action_spec in input_hint.items():
                    print(f"    {event_type}: {action_spec}")
            else:
                print(f"    {input_hint}")
        else:
            print("    (no input hint)")
    
    # Verify input hints
    area_input = params["Area"].get("input_hint") or params["Area"].get("input")
    assert area_input is not None, "Area should have input hint"
    assert "wheel" in area_input, "Area should have wheel control"
    assert area_input["wheel"]["action"] == "scale", "Area wheel should scale"
    
    width_input = params["Width"].get("input_hint") or params["Width"].get("input")
    assert width_input is not None, "Width should have input hint"
    assert "wheel" in width_input, "Width should have wheel control"
    assert "Ctrl" in width_input["wheel"]["modifiers"], "Width wheel should require Ctrl"
    
    center_input = params["Center"].get("input_hint") or params["Center"].get("input")
    assert center_input is not None, "Center should have input hint"
    assert "drag" in center_input, "Center should have drag control"
    assert center_input["drag"]["value_from"] == "x", "Center drag should use x coordinate"
    
    print("\n✓ All input hints are properly configured!")
    return True

def test_evaluation():
    """Test that the Gaussian model evaluates correctly."""
    from models.model_specs import GaussianModelSpec
    
    print("\n" + "=" * 60)
    print("Test 3: Model Evaluation")
    print("=" * 60)
    
    spec = GaussianModelSpec()
    x = np.linspace(-5, 5, 100)
    y = spec.evaluate(x)
    
    print(f"\nEvaluated Gaussian over x range: [{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"  Max value: {np.max(y):.4f}")
    print(f"  Value at center: {y[50]:.4f}")
    print(f"  Integral (approximate): {np.trapz(y, x):.4f}")
    
    # Verify evaluation works
    assert len(y) == len(x), "Output should match input length"
    assert np.max(y) > 0, "Gaussian should have positive values"
    assert np.abs(np.trapz(y, x) - 1.0) < 0.1, "Integral should be close to area (1.0)"
    
    print("\n✓ Model evaluation works correctly!")
    return True

def test_input_map_building():
    """Test that the ViewModel can build an input_map from the specs."""
    from models.model_specs import GaussianModelSpec
    from viewmodel.fitter_vm import FitterViewModel
    from models.model_state import ModelState
    
    print("\n" + "=" * 60)
    print("Test 4: Input Map Building")
    print("=" * 60)
    
    # Create a ViewModel with Gaussian model
    state = ModelState(model_name="Gaussian")
    vm = FitterViewModel(model_state=state)
    
    # Get parameters (this should build the input_map)
    params = vm.get_parameters()
    
    print("\nBuilt input_map:")
    for event_type, handlers in vm._input_map.items():
        print(f"\n  {event_type}:")
        for handler in handlers:
            param = handler.get("param")
            action = handler.get("action", {})
            print(f"    Parameter: {param}")
            print(f"    Action: {action}")
    
    # Verify input_map
    assert "drag" in vm._input_map, "Should have drag handlers"
    assert "wheel" in vm._input_map, "Should have wheel handlers"
    
    drag_params = [h["param"] for h in vm._input_map["drag"]]
    assert "Center" in drag_params, "Center should have drag handler"
    
    wheel_params = [h["param"] for h in vm._input_map["wheel"]]
    assert "Area" in wheel_params, "Area should have wheel handler"
    assert "Width" in wheel_params, "Width should have wheel handler"
    
    print("\n✓ Input map built correctly!")
    return True

def test_selection_flow():
    """Test the parameter selection flow."""
    from models.model_specs import GaussianModelSpec
    from viewmodel.fitter_vm import FitterViewModel
    from models.model_state import ModelState
    
    print("\n" + "=" * 60)
    print("Test 5: Selection Flow")
    print("=" * 60)
    
    # Create a ViewModel with Gaussian model
    state = ModelState(model_name="Gaussian")
    vm = FitterViewModel(model_state=state)
    
    # Get parameters to build input_map
    params = vm.get_parameters()
    
    # Test begin_selection
    print("\nTesting selection of Center parameter...")
    success = vm.begin_selection("Center", 0.5, 1.0)
    assert success, "begin_selection should succeed for Center"
    
    selected = getattr(state, "_selected_param", None)
    print(f"  Selected parameter: {selected}")
    assert selected == "Center", "Center should be selected"
    
    # Test end_selection
    print("\nEnding selection...")
    vm.end_selection()
    
    selected = getattr(state, "_selected_param", None)
    print(f"  Selected parameter after end: {selected}")
    assert selected is None or selected == "", "Selection should be cleared"
    
    print("\n✓ Selection flow works correctly!")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BigFit Gaussian Interaction Tests")
    print("=" * 60)
    
    tests = [
        test_gaussian_defaults,
        test_input_hints,
        test_evaluation,
        test_input_map_building,
        test_selection_flow,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result, None))
        except Exception as e:
            results.append((test.__name__, False, str(e)))
            print(f"\n✗ Test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
