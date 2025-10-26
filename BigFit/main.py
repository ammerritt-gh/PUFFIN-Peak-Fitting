# main.py
import sys
from PySide6.QtWidgets import QApplication

# Import your modules from the project structure
from model.model_state import ModelState
from viewmodel.fitter_vm import FitterViewModel
from view.main_window import MainWindow


def main():
    """Entry point for the MVVM-based PUMA Fitter application."""
    app = QApplication(sys.argv)

    # --- Model layer ---
    model = ModelState()

    # --- ViewModel layer ---
    viewmodel = FitterViewModel(model)

    # --- View layer ---
    window = MainWindow(viewmodel)
    window.resize(1200, 800)
    window.show()

    # --- Start event loop ---
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

