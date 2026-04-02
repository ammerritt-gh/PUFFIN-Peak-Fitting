import unittest

from models import Parameter
from viewmodel.fitter_vm import FitterViewModel


class _DummySignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _ResolutionSpec:
    def __init__(self):
        sigma = Parameter("Sigma", value=1.0, minimum=0.0)
        width = Parameter("Width", value=2.0, minimum=0.0)
        self.params = {"Sigma": sigma, "Width": width}


class _DummyResolutionViewModel:
    def __init__(self):
        self._resolution_spec = _ResolutionSpec()
        self._resolution_model_name = "Gaussian"
        self.resolution_updated = _DummySignal()
        self.log_message = _DummySignal()
        self.saved = 0

    def _schedule_fit_save(self):
        self.saved += 1

    def _log_message(self, message):
        self.log_message.emit(message)

    def _split_parameter_updates(self, params):
        return FitterViewModel._split_parameter_updates(self, params)

    def _collect_link_groups(self, model_spec):
        return FitterViewModel._collect_link_groups(self, model_spec)

    def _apply_fixed_state_to_group(self, model_spec, model_obj, base, fixed_value, link_groups):
        return FitterViewModel._apply_fixed_state_to_group(self, model_spec, model_obj, base, fixed_value, link_groups)

    def _set_param_fixed_state(self, model_spec, model_obj, name, fixed_value):
        return FitterViewModel._set_param_fixed_state(self, model_spec, model_obj, name, fixed_value)

    def _values_close(self, a, b):
        return FitterViewModel._values_close(self, a, b)


class FitterViewModelResolutionParameterTests(unittest.TestCase):
    def test_linked_resolution_value_propagates_to_group(self):
        viewmodel = _DummyResolutionViewModel()
        viewmodel._resolution_spec.params["Sigma"].link_group = 1
        viewmodel._resolution_spec.params["Width"].link_group = 1

        FitterViewModel.apply_resolution_parameters(viewmodel, {"Sigma": 4.5})

        self.assertEqual(viewmodel._resolution_spec.params["Sigma"].value, 4.5)
        self.assertEqual(viewmodel._resolution_spec.params["Width"].value, 4.5)
        self.assertEqual(len(viewmodel.resolution_updated.calls), 1)

    def test_fixed_resolution_update_blocks_value_change(self):
        viewmodel = _DummyResolutionViewModel()
        viewmodel._resolution_spec.params["Sigma"].value = 1.25

        FitterViewModel.apply_resolution_parameters(
            viewmodel,
            {"Sigma__fixed": True, "Sigma": 9.0},
        )

        self.assertTrue(viewmodel._resolution_spec.params["Sigma"].fixed)
        self.assertEqual(viewmodel._resolution_spec.params["Sigma"].value, 1.25)
        self.assertTrue(any("Skipped resolution update" in call[0] for call in viewmodel.log_message.calls if call))

    def test_unfix_then_apply_resolution_value_in_same_batch(self):
        viewmodel = _DummyResolutionViewModel()
        viewmodel._resolution_spec.params["Sigma"].fixed = True

        FitterViewModel.apply_resolution_parameters(
            viewmodel,
            {"Sigma__fixed": False, "Sigma": 3.0},
        )

        self.assertFalse(viewmodel._resolution_spec.params["Sigma"].fixed)
        self.assertEqual(viewmodel._resolution_spec.params["Sigma"].value, 3.0)


if __name__ == "__main__":
    unittest.main()