# main.py
import sys
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QLineEdit
from models import ModelState
from view.main_window import MainWindow
from viewmodel.fitter_vm import FitterViewModel


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
    try:
        viewmodel.parameters_updated.connect(window._refresh_parameters)
    except Exception:
        pass

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
        viewmodel.apply_parameters(params)

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
                pass

    # connect initially (if any params already present)
    connect_param_signals()

    # reconnect whenever MainWindow rebuilds the parameters form
    try:
        window.parameters_updated.connect(connect_param_signals)
    except Exception:
        pass

    # Apply initial parameters to ViewModel
    try:
        on_param_changed()
    except Exception:
        pass

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
