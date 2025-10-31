# worker/fit_worker.py
from PySide6.QtCore import QThread, Signal
import numpy as np


class FitWorker(QThread):
    progress = Signal(float)
    finished = Signal(object)  # Fit result

    def __init__(self, model_state):
        super().__init__()
        self.model_state = model_state
        self.running = True

    def run(self):
        """Example background task — just simulate work."""
        import time
        for i in range(1, 11):
            time.sleep(0.1)
            self.progress.emit(i / 10)
        y_fit = self.model_state.evaluate()
        self.finished.emit(y_fit)

