# main.py
import sys
from PySide6.QtWidgets import QApplication
from models.model_state import ModelState
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

    # Connect View → ViewModel inputs
    window.gauss_spin.valueChanged.connect(
        lambda v: viewmodel.apply_parameters(gauss=v, lorentz=window.lorentz_spin.value(), temp=window.temp_spin.value())
    )
    window.lorentz_spin.valueChanged.connect(
        lambda v: viewmodel.apply_parameters(gauss=window.gauss_spin.value(), lorentz=v, temp=window.temp_spin.value())
    )
    window.temp_spin.valueChanged.connect(
        lambda v: viewmodel.apply_parameters(gauss=window.gauss_spin.value(), lorentz=window.lorentz_spin.value(), temp=v)
    )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
