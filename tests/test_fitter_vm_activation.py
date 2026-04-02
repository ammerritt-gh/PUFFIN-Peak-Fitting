import unittest

import numpy as np

from models import ModelState
from viewmodel.fitter_vm import FitterViewModel


class _DummySignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _DummyViewModel:
    def __init__(self):
        self.state = ModelState()
        self._datasets = []
        self._active_dataset_index = None
        self._resolution_model_name = "None"
        self._resolution_spec = None
        self.parameters_updated = _DummySignal()
        self.resolution_updated = _DummySignal()
        self.log_messages = []

    def _prepare_dataset_state(self, dataset, model_name="Voigt"):
        return FitterViewModel._prepare_dataset_state(self, dataset, model_name)

    def _apply_dataset_to_state(self, dataset, model_name="Voigt"):
        return FitterViewModel._apply_dataset_to_state(self, dataset, model_name)

    def clear_selected_curve(self):
        return None

    def _get_current_file_path(self):
        return None

    def _log_message(self, message):
        self.log_messages.append(message)

    def _load_fit_for_current_file(self):
        return False

    def _get_default_model_choice(self):
        return None

    def _load_default_fit(self):
        return False

    def _synchronize_model_state(self):
        return None

    def _emit_file_queue(self):
        return None

    def update_plot(self):
        return None


class FitterViewModelActivationTests(unittest.TestCase):
    def test_activate_file_preserves_state_when_dataset_is_invalid(self):
        viewmodel = _DummyViewModel()

        original_x = np.array(viewmodel.state.x_data, copy=True)
        original_y = np.array(viewmodel.state.y_data, copy=True)
        original_errors = np.array(viewmodel.state.errors, copy=True)
        original_model_name = viewmodel.state.model_name

        viewmodel._datasets = [
            {
                "x": [1.0, 2.0],
                "y": [5.0],
                "err": [1.0],
                "info": {"path": "broken.dat", "name": "broken.dat"},
            }
        ]
        viewmodel._active_dataset_index = None

        FitterViewModel.activate_file(viewmodel, 0)

        self.assertIsNone(viewmodel._active_dataset_index)
        self.assertEqual(viewmodel.state.model_name, original_model_name)
        np.testing.assert_array_equal(viewmodel.state.x_data, original_x)
        np.testing.assert_array_equal(viewmodel.state.y_data, original_y)
        np.testing.assert_array_equal(viewmodel.state.errors, original_errors)


if __name__ == "__main__":
    unittest.main()
