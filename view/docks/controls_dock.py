"""
Controls dock widget for data operations and file management.
"""

from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QPushButton, QLabel,
    QHBoxLayout, QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class ControlsDock(QDockWidget):
    """Dock widget for data controls and file management."""

    # Signals
    load_data_clicked = Signal()
    save_data_clicked = Signal()
    run_fit_clicked = Signal()
    update_plot_clicked = Signal()
    edit_config_clicked = Signal()
    exclude_toggled = Signal(bool)
    include_all_clicked = Signal()
    file_selected = Signal(int)  # row number
    remove_file_clicked = Signal()
    clear_files_clicked = Signal()
    resolution_clicked = Signal()  # Open resolution window
    fit_settings_clicked = Signal()  # Open fit settings window
    default_model_changed = Signal(str)  # default model for new files

    def __init__(self, parent=None):
        """
        Initialize the controls dock.

        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__("Controls", parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components."""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        # Action buttons
        self.load_btn = QPushButton("Load Data")
        self.save_btn = QPushButton("Save Data")
        self.fit_btn = QPushButton("Run Fit")
        self.fit_settings_btn = QPushButton("Fit Settings...")
        self.update_btn = QPushButton("Update Plot")
        self.config_btn = QPushButton("Edit Config")
        self.resolution_btn = QPushButton("Resolution...")

        layout.addWidget(QLabel("Data Controls"))
        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.fit_btn)
        layout.addWidget(self.fit_settings_btn)
        layout.addWidget(self.config_btn)
        layout.addWidget(self.update_btn)
        layout.addWidget(self.resolution_btn)

        # Exclude toggle (click to enable box/point exclusion)
        self.exclude_btn = QPushButton("Exclude")
        self.exclude_btn.setCheckable(True)
        layout.addWidget(self.exclude_btn)

        # Include All button placed directly under the Exclude toggle for convenience
        self.include_all_btn = QPushButton("Include All")
        layout.addWidget(self.include_all_btn)

        # File list
        layout.addWidget(QLabel("Loaded Files"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.file_list)

        # File management buttons
        file_btn_row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected")
        self.clear_list_btn = QPushButton("Clear List")
        file_btn_row.addWidget(self.remove_btn)
        file_btn_row.addWidget(self.clear_list_btn)
        layout.addLayout(file_btn_row)

        # Default model selector for new files
        layout.addWidget(QLabel("Default Model for New Files:"))
        self.default_model_combo = QComboBox()
        self.default_model_combo.addItem("(Use Last Fit)", None)
        self.default_model_combo.addItem("(None - User Select)", None)
        # Populate with available models dynamically
        try:
            from models import get_available_model_names
            import re
            
            def _pretty(name: str) -> str:
                s = re.sub(r"ModelSpec$", "", name)
                s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
                pretty = s.strip()
                if pretty.lower() == "composite":
                    return "Custom Model"
                return pretty
            
            spec_names = get_available_model_names()
            for spec_name in spec_names:
                display_name = _pretty(spec_name)
                self.default_model_combo.addItem(display_name, spec_name)
        except Exception:
            # Fallback if model discovery fails
            pass
        layout.addWidget(self.default_model_combo)

        layout.addStretch(1)

        self.setWidget(left_widget)

        # Connect internal signals to emit dock signals
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.fit_btn.clicked.connect(self._on_fit_clicked)
        self.fit_settings_btn.clicked.connect(self._on_fit_settings_clicked)
        self.update_btn.clicked.connect(self._on_update_clicked)
        self.config_btn.clicked.connect(self._on_config_clicked)
        self.resolution_btn.clicked.connect(self._on_resolution_clicked)
        self.exclude_btn.toggled.connect(self._on_exclude_toggled)
        self.include_all_btn.clicked.connect(self._on_include_all_clicked)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        self.clear_list_btn.clicked.connect(self._on_clear_clicked)
        self.default_model_combo.currentIndexChanged.connect(self._on_default_model_changed)

    def _on_load_clicked(self):
        """Handle load button click."""
        self.load_data_clicked.emit()

    def _on_save_clicked(self):
        """Handle save button click."""
        self.save_data_clicked.emit()

    def _on_fit_clicked(self):
        """Handle fit button click."""
        self.run_fit_clicked.emit()

    def _on_update_clicked(self):
        """Handle update button click."""
        self.update_plot_clicked.emit()

    def _on_config_clicked(self):
        """Handle config button click."""
        self.edit_config_clicked.emit()

    def _on_resolution_clicked(self):
        """Handle resolution button click."""
        self.resolution_clicked.emit()

    def _on_fit_settings_clicked(self):
        """Handle fit settings button click."""
        self.fit_settings_clicked.emit()

    def _on_exclude_toggled(self, checked):
        """Handle exclude button toggle."""
        self.exclude_toggled.emit(checked)

    def _on_include_all_clicked(self):
        """Handle include all button click."""
        self.include_all_clicked.emit()

    def _on_file_selected(self, row):
        """Handle file selection."""
        self.file_selected.emit(row)

    def _on_remove_clicked(self):
        """Handle remove button click."""
        self.remove_file_clicked.emit()

    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.clear_files_clicked.emit()

    def _on_default_model_changed(self, index):
        """Handle default model selection change."""
        model_data = self.default_model_combo.itemData(index)
        if model_data:
            self.default_model_changed.emit(model_data)
        else:
            # Special values like "(Use Last Fit)" or "(None)"
            text = self.default_model_combo.itemText(index)
            self.default_model_changed.emit(text)

    def update_files(self, files):
        """
        Update the file list with the provided files.

        Args:
            files: List of file entries (dicts with 'name', 'path', 'active', etc.)
        """
        if not hasattr(self, "file_list"):
            return

        entries = files or []
        active_row = -1
        self.file_list.blockSignals(True)
        self.file_list.clear()

        for entry in entries:
            entry_dict = entry if isinstance(entry, dict) else {}
            name = entry_dict.get("name")
            if not name:
                idx = entry_dict.get("index")
                name = f"Dataset {idx + 1}" if idx is not None else "Dataset"

            item = QListWidgetItem(name)
            info_obj = entry_dict.get("info")
            info = info_obj if isinstance(info_obj, dict) else {}
            path = entry_dict.get("path")
            if not path and info:
                path = info.get("path")
            if path:
                item.setToolTip(str(path))
            if entry_dict.get("active"):
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            item.setData(Qt.UserRole, entry)
            self.file_list.addItem(item)
            if entry_dict.get("active"):
                active_row = self.file_list.count() - 1

        if active_row >= 0:
            self.file_list.setCurrentRow(active_row)

        self.file_list.blockSignals(False)
        self._update_file_action_state()

    def _update_file_action_state(self):
        """Update the enabled state of file action buttons."""
        has_files = self.file_list.count() > 0
        has_selection = self.file_list.currentRow() >= 0
        self.remove_btn.setEnabled(has_selection)
        # Allow Clear List to be used even when the visible list is empty.
        self.clear_list_btn.setEnabled(True)

    def get_current_file_row(self):
        """Get the currently selected file row."""
        return self.file_list.currentRow()

    def set_exclude_button_checked(self, checked):
        """Set the exclude button checked state."""
        self.exclude_btn.setChecked(checked)
