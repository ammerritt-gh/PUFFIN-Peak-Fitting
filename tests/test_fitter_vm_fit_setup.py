import unittest
from unittest import mock

import numpy as np

from models import ModelState, Parameter
from viewmodel.fitter_vm import FitterViewModel


class _DummySignal:
    def __init__(self):
        self.callbacks = []
        self.calls = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        self.calls.append(args)
        for callback in list(self.callbacks):
            callback(*args)


class _SimpleModelSpec:
    def __init__(self):
        amplitude = Parameter("Amplitude", value=1.0)
        amplitude.fixed = True
        self.params = {"Amplitude": amplitude}

    def evaluate(self, x, params=None):
        return np.ones_like(np.asarray(x, dtype=float))


class _ResolutionSpec:
    def __init__(self):
        sigma = Parameter("Sigma", value=1.0, minimum=0.0)
        sigma.link_group = 1
        width = Parameter("Width", value=1.0, minimum=0.0)
        width.link_group = 1
        self.params = {"Sigma": sigma, "Width": width}


class _AllFixedResolutionSpec:
    def __init__(self):
        sigma = Parameter("Sigma", value=1.0, minimum=0.0)
        sigma.fixed = True
        self.params = {"Sigma": sigma}


class _CapturingWorker:
    last_init = None

    def __init__(self, x, y, model_func, params, err=None, bounds=None, **kwargs):
        type(self).last_init = {
            "x": np.asarray(x),
            "y": np.asarray(y),
            "params": dict(params),
            "bounds": bounds,
            "kwargs": dict(kwargs),
        }
        self.progress = _DummySignal()
        self.finished = _DummySignal()
        self.error_occurred = _DummySignal()
        self.step_completed = _DummySignal()
        self.started = False

    def start(self):
        self.started = True


class _DummyFitViewModel:
    def __init__(self, model_spec=None, resolution_spec=None):
        self.state = ModelState()
        self.state.model_spec = model_spec or _SimpleModelSpec()
        self.state.model_name = "Dummy"
        self._fit_worker = None
        self._resolution_model_name = "Resolution" if resolution_spec is not None else "None"
        self._resolution_spec = resolution_spec
        self.fit_started = _DummySignal()
        self.fit_finished = _DummySignal()
        self.fit_progress = _DummySignal()
        self.fit_step_completed = _DummySignal()
        self.parameters_updated = _DummySignal()
        self.resolution_updated = _DummySignal()
        self.revert_available_changed = _DummySignal()
        self.messages = []
        self.pre_fit_store_calls = 0

    def _log_message(self, message):
        self.messages.append(message)

    def evaluate_with_resolution(self, x, y):
        return y

    def has_resolution(self):
        return FitterViewModel.has_resolution(self)

    def get_resolution_state(self):
        return FitterViewModel.get_resolution_state(self)

    def _store_pre_fit_state(self):
        self.pre_fit_store_calls += 1

    def update_plot(self):
        return None


class FitterViewModelFitSetupTests(unittest.TestCase):
    def setUp(self):
        _CapturingWorker.last_init = None

    def test_run_fit_deduplicates_linked_resolution_parameters(self):
        viewmodel = _DummyFitViewModel(resolution_spec=_ResolutionSpec())

        with mock.patch("worker.fit_worker.FitWorker", _CapturingWorker):
            FitterViewModel.run_fit(viewmodel)

        self.assertIsNotNone(_CapturingWorker.last_init)
        self.assertEqual(list(_CapturingWorker.last_init["params"].keys()), ["res__Sigma"])
        self.assertEqual(viewmodel.pre_fit_store_calls, 1)
        self.assertEqual(len(viewmodel.fit_started.calls), 1)

    def test_run_fit_steps_deduplicates_linked_resolution_parameters(self):
        viewmodel = _DummyFitViewModel(resolution_spec=_ResolutionSpec())

        with mock.patch("worker.fit_worker.IterativeFitWorker", _CapturingWorker):
            FitterViewModel.run_fit_steps(viewmodel, num_steps=2, live_preview=False)

        self.assertIsNotNone(_CapturingWorker.last_init)
        self.assertEqual(list(_CapturingWorker.last_init["params"].keys()), ["res__Sigma"])
        self.assertEqual(_CapturingWorker.last_init["kwargs"].get("max_steps"), 2)
        self.assertEqual(viewmodel.pre_fit_store_calls, 1)
        self.assertEqual(len(viewmodel.fit_started.calls), 1)

    def test_run_fit_steps_does_not_store_revert_state_when_nothing_is_fit_free(self):
        viewmodel = _DummyFitViewModel(resolution_spec=_AllFixedResolutionSpec())

        with mock.patch("worker.fit_worker.IterativeFitWorker", _CapturingWorker):
            FitterViewModel.run_fit_steps(viewmodel, num_steps=1, live_preview=False)

        self.assertIsNone(_CapturingWorker.last_init)
        self.assertEqual(viewmodel.pre_fit_store_calls, 0)
        self.assertEqual(len(viewmodel.fit_started.calls), 0)
        self.assertEqual(len(viewmodel.fit_finished.calls), 1)
        self.assertIn("No free parameters to fit.", viewmodel.messages)


if __name__ == "__main__":
    unittest.main()