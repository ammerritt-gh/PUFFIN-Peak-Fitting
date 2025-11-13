# main.py
import sys
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QLineEdit
from models import ModelState
from view.main_window import MainWindow
from viewmodel.fitter_vm import FitterViewModel
import os
import traceback


def main():
    app = QApplication(sys.argv)

    # Model + ViewModel + View
    model_state = ModelState()
    viewmodel = FitterViewModel(model_state)
    window = MainWindow(viewmodel)

    # Connect ViewModel → View signals
    viewmodel.plot_updated.connect(window.update_plot_data)
    viewmodel.log_message.connect(window.append_log)
    # Refresh parameter panel when viewmodel reports parameter changes (e.g. after a fit)
    # If the MainWindow implements its own auto-apply / refresh handling (exposed
    # via `_auto_apply_param`) then the window will manage parameter refreshes
    # itself and we should avoid double-connecting the same handler here because
    # it can cause duplicate refreshes (which may rebuild widgets and steal
    # focus from active spinboxes during held adjustments).
    try:
        if not getattr(window, "_auto_apply_param", False):
            viewmodel.parameters_updated.connect(window._refresh_parameters)
    except Exception as e:
        try:
            viewmodel.log_message.emit(f"Failed to connect parameters_updated: {e}\n{traceback.format_exc()}")
        except Exception:
            print(f"Failed to connect parameters_updated: {e}")

    # --- Dynamic parameter handling (replace legacy gauss_spin / lorentz_spin / temp_spin) ---
    def read_all_params():
        params = {}
        for name, widget in getattr(window, "param_widgets", {}).items():
            try:
                if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    params[name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[name] = bool(widget.isChecked())
                elif isinstance(widget, QComboBox):
                    params[name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    params[name] = widget.text()
                else:
                    # fallback: try to call value()
                    params[name] = getattr(widget, "value", lambda: None)()
            except Exception:
                params[name] = None
        return params

    def on_param_changed(_=None):
        params = read_all_params()
        # Prefer centralized dispatcher when available, fall back to direct call
        try:
            if hasattr(viewmodel, "handle_action"):
                try:
                    viewmodel.handle_action("apply_parameters", params=params)
                except Exception:
                    viewmodel.apply_parameters(params)
            else:
                viewmodel.apply_parameters(params)
        except Exception as e:
            try:
                viewmodel.log_message.emit(f"Failed to apply parameters: {e}\n{traceback.format_exc()}")
            except Exception:
                print(f"Failed to apply parameters: {e}")

    def connect_param_signals():
        # Disconnecting previous connections is tricky; keep it simple: try to connect,
        # ignore errors from duplicate connects.
        for name, widget in getattr(window, "param_widgets", {}).items():
            try:
                if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    widget.valueChanged.connect(on_param_changed)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.connect(on_param_changed)
                elif isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(on_param_changed)
                elif isinstance(widget, QLineEdit):
                    widget.editingFinished.connect(on_param_changed)
            except Exception:
                try:
                    viewmodel.log_message.emit(f"Failed to connect param signal for '{name}': {widget}")
                except Exception:
                    print(f"Failed to connect param signal for '{name}'")

    # connect initially (if any params already present)
    # If the MainWindow provides its own per-widget auto-apply handler, skip
    # the legacy global connect to avoid duplicate applies.
    if not hasattr(window, "_auto_apply_param"):
        connect_param_signals()

    # reconnect whenever MainWindow rebuilds the parameters form
    if not hasattr(window, "_auto_apply_param"):
        try:
            window.parameters_updated.connect(connect_param_signals)
        except Exception as e:
            try:
                viewmodel.log_message.emit(f"Failed to hook parameters_updated -> connect_param_signals: {e}\n{traceback.format_exc()}")
            except Exception:
                print(f"Failed to hook parameters_updated -> connect_param_signals: {e}")

    # Apply initial parameters to ViewModel
    try:
        on_param_changed()
    except Exception as e:
        try:
            viewmodel.log_message.emit(f"Failed during initial parameter application: {e}\n{traceback.format_exc()}")
        except Exception:
            print(f"Failed during initial parameter application: {e}")

    # Attempt to restore the last-loaded dataset from the config (if present).
    # This keeps the app state persistent between runs.
    try:
        from dataio import get_config, load_data_from_file
        cfg = get_config()
        last = getattr(cfg, "last_loaded_file", None)
        if last and os.path.isfile(last):
            try:
                x, y, err, info = load_data_from_file(last)
                # Use ModelState.set_data if available so it resets derived state
                try:
                    model_state.set_data(x, y)
                except Exception:
                    model_state.x_data = x
                    model_state.y_data = y
                try:
                    model_state.errors = err
                except Exception:
                    pass
                try:
                    model_state.file_info = info
                except Exception:
                    pass
                # notify via viewmodel and update plot
                try:
                    viewmodel.log_message.emit(f"Restored last dataset: {info.get('name')}")
                except Exception:
                    pass
                try:
                    viewmodel.update_plot()
                except Exception:
                    pass
            except Exception as e:
                try:
                    viewmodel.log_message.emit(f"Failed to restore last dataset: {e}")
                except Exception:
                    pass
    except Exception as e:
        try:
            viewmodel.log_message.emit(f"Config/restore failed: {e}\n{traceback.format_exc()}")
        except Exception:
            print(f"Config/restore failed: {e}")

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
