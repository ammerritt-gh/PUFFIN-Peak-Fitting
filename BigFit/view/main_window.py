# view/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QFormLayout, QDoubleSpinBox,
    QDialog, QLineEdit, QDialogButtonBox, QHBoxLayout, QFileDialog,
    QScrollArea, QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QAbstractItemView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
import pyqtgraph as pg
import numpy as np

from view.view_box import CustomViewBox
from view.input_handler import InputHandler

# -- color palette (change these) --
PLOT_BG = "white"       # plot background
POINT_COLOR = "black"   # scatter points
ERROR_COLOR = "black"   # error bars (can match points)
FIT_COLOR = "purple"     # fit line
AXIS_COLOR = "black"    # axis and tick labels
GRID_ALPHA = 0.5

class MainWindow(QMainWindow):
    def __init__(self, viewmodel=None):
        super().__init__()
        self.setWindowTitle("PUMA Peak Fitter")
        self.viewmodel = viewmodel
        self.param_widgets = {}   # name -> widget
        self.curves = {}  # curve_id -> PlotDataItem
        self.selected_curve_id = None

        # --- Central Plot ---
        # Initialize the plot widget inside _init_plot() to avoid duplicate
        # creation and ensure a single central widget is used.
        self._init_plot()

        # --- Docks ---
        self._init_left_dock()
        # create the bottom (log) dock before the right dock so logging is available
        self._init_bottom_dock()
        self._init_right_dock()
        # Elements dock (separate dock placed under Parameters)
        try:
            self._init_elements_dock()
        except Exception:
            pass

        for dock in [self.left_dock, self.right_dock, self.bottom_dock]:
            dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        self.resize(1400, 800)

    # --------------------------
    # Plot setup
    # --------------------------
    def _init_plot(self):
        # Use custom ViewBox for interactive behavior
        self.viewbox = CustomViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self.viewbox, title="Data and Fit")
        self.setCentralWidget(self.plot_widget)

        # optional InputHandler for global mouse/key binding (pass viewbox so it can connect signals)
        self.input_handler = InputHandler(self.viewbox, self.viewmodel)
        self.plot_widget.installEventFilter(self.input_handler)
        # Also install event filter on the ViewBox so we can intercept wheel events
        # (ViewBox handles zoom by default). This allows the InputHandler to
        # intercept wheel events when a curve is selected and use them for
        # parameter updates instead of zooming.
        try:
            self.viewbox.installEventFilter(self.input_handler)
            # keep a backref so the ViewBox can delegate wheel handling when needed
            try:
                setattr(self.viewbox, "_input_handler", self.input_handler)
            except Exception:
                pass
        except Exception:
            pass

        # connect ViewBox interaction signals → ViewModel
        if self.viewmodel:
            self.viewbox.peakSelected.connect(self._on_peak_selected)
            self.viewbox.peakDeselected.connect(lambda: self.viewmodel.set_selected_curve(None))
            self.viewbox.peakMoved.connect(self._on_peak_moved)
            self.viewbox.excludePointClicked.connect(self._on_exclude_point)
            self.viewbox.excludeBoxDrawn.connect(self._on_exclude_box)

        # connect curve selection changes
        # Let the InputHandler know when the selected curve changes (no notify back)
        try:
            self.viewmodel.curve_selection_changed.connect(lambda cid: self.input_handler.set_selected_curve(cid, notify_vm=False))
        except Exception:
            pass
        # Guard the secondary connect so the absence of a ViewModel or signal
        # won't raise during initialization.
        try:
            self.viewmodel.curve_selection_changed.connect(self._on_curve_selected)
        except Exception:
            pass


        # --- visual setup ---
        self.plot_widget.setBackground(PLOT_BG)
        self.plot_widget.showGrid(x=True, y=True, alpha=GRID_ALPHA)

        # scatter (data points)
        self.scatter = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(POINT_COLOR))
        self.plot_widget.addItem(self.scatter)
        # excluded points overlay (small grey 'x' markers)
        try:
            self.excluded_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen('gray'), brush=None, symbol='x')
            self.plot_widget.addItem(self.excluded_scatter)
        except Exception:
            self.excluded_scatter = None

        # error bars
        self.err_item = pg.ErrorBarItem(pen=pg.mkPen(ERROR_COLOR))
        self.plot_widget.addItem(self.err_item)

    # Note: fit overlay is managed by update_plot_data via self.curves['fit']
    # so we avoid creating a separate persistent `self.fit_curve` here to
    # prevent duplicate plot items.

        # axis colors
        for ax in ("left", "bottom", "right", "top"):
            try:
                axis = self.plot_widget.getAxis(ax)
                axis.setPen(pg.mkPen(AXIS_COLOR))
                axis.setTextPen(pg.mkPen(AXIS_COLOR))
            except Exception:
                pass


    # --------------------------
    # Plot interaction callbacks
    # --------------------------
    def _on_peak_selected(self, x, y):
        # Only forward peak selection to the ViewModel when a curve is selected
        try:
            if hasattr(self, "input_handler") and getattr(self.input_handler, "selected_curve_id", None) is None:
                # ignore peak selection unless a curve is selected
                self.append_log("Peak clicked but no curve selected — ignored.")
                return
        except Exception:
            pass

        if self.viewmodel and hasattr(self.viewmodel, "on_peak_selected"):
            self.viewmodel.on_peak_selected(x, y)
        self.append_log(f"Selected peak near ({x:.3f}, {y:.3f})")

    def _on_peak_moved(self, info):
        # Only forward peak movement when a curve is selected
        try:
            if hasattr(self, "input_handler") and getattr(self.input_handler, "selected_curve_id", None) is None:
                self.append_log("Peak moved but no curve selected — ignored.")
                return
        except Exception:
            pass

        if self.viewmodel and hasattr(self.viewmodel, "on_peak_moved"):
            self.viewmodel.on_peak_moved(info)
        self.append_log(f"Moved peak → center={info.get('center', 0):.3f}")

    def _on_exclude_point(self, x, y):
        if self.viewmodel and hasattr(self.viewmodel, "on_exclude_point"):
            self.viewmodel.on_exclude_point(x, y)
        self.append_log(f"Toggled exclusion at ({x:.3f}, {y:.3f})")

    def _on_exclude_box(self, x0, y0, x1, y1):
        if self.viewmodel and hasattr(self.viewmodel, "on_exclude_box"):
            self.viewmodel.on_exclude_box(x0, y0, x1, y1)
        self.append_log(f"Box exclusion from ({x0:.3f}, {y0:.3f}) → ({x1:.3f}, {y1:.3f})")

    # --------------------------
    # Docks
    # --------------------------
    def _init_left_dock(self):
        self.left_dock = QDockWidget("Controls", self)
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)

        load_btn = QPushButton("Load Data")
        save_btn = QPushButton("Save Data")
        fit_btn = QPushButton("Run Fit")
        update_btn = QPushButton("Update Plot")
        config_btn = QPushButton("Edit Config")

        layout.addWidget(QLabel("Data Controls"))
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(fit_btn)
        layout.addWidget(config_btn)
        layout.addWidget(update_btn)
        # Exclude toggle (click to enable box/point exclusion)
        exclude_btn = QPushButton("Exclude")
        exclude_btn.setCheckable(True)
        layout.addWidget(exclude_btn)
        # keep a reference for external updates (e.g. hotkey toggles)
        self.exclude_btn = exclude_btn
        # wire the button to the viewbox so clicking updates exclude mode
        try:
            if hasattr(self, "viewbox") and self.viewbox is not None:
                exclude_btn.toggled.connect(self.viewbox.set_exclude_mode)
                # Keep the button state synced when the viewbox mode is changed
                if hasattr(self.viewbox, "excludeModeChanged"):
                    try:
                        self.viewbox.excludeModeChanged.connect(exclude_btn.setChecked)
                    except Exception:
                        pass
        except Exception:
            pass

        # Include All button placed directly under the Exclude toggle for convenience
        include_all_btn = QPushButton("Include All")
        layout.addWidget(include_all_btn)

        layout.addWidget(QLabel("Loaded Files"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.file_list)

        file_btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        clear_list_btn = QPushButton("Clear List")
        file_btn_row.addWidget(remove_btn)
        file_btn_row.addWidget(clear_list_btn)
        layout.addLayout(file_btn_row)

        self.file_remove_btn = remove_btn
        self.file_clear_btn = clear_list_btn
        # keep a backref to the include-all button placed above
        self.include_all_btn = include_all_btn

        layout.addStretch(1)

        self.left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # Connect UI → ViewModel
        if self.viewmodel:
            load_btn.clicked.connect(self.viewmodel.load_data)
            save_btn.clicked.connect(self.viewmodel.save_data)
            fit_btn.clicked.connect(self.viewmodel.run_fit)
            update_btn.clicked.connect(self.viewmodel.update_plot)
            # The file-list Clear button handles clearing queued datasets (see below)
            config_btn.clicked.connect(self._on_edit_config_clicked)
            # Exclude mode button toggles the CustomViewBox exclude_mode
            def _on_exclude_toggled(checked):
                # Toggle exclude mode in the ViewBox and log the change.
                # IMPORTANT: do not touch or reapply any model parameters here —
                # that responsibility belongs to the Apply/Run Fit controls in the
                # parameter panel and ViewModel. Removing parameter snapshot/reapply
                # prevents accidental fit redraws when toggling exclusion mode.
                try:
                    # Setting exclude mode on the viewbox updates interaction only
                    if hasattr(self, 'viewbox') and self.viewbox is not None:
                        self.viewbox.set_exclude_mode(bool(checked))

                    # Keep input handler selection state unchanged (no notify)
                    try:
                        if hasattr(self, 'input_handler'):
                            self.input_handler.selected_curve_id = getattr(self.input_handler, 'selected_curve_id', None)
                    except Exception:
                        pass

                    if checked:
                        self.append_log("Exclude mode enabled.")
                    else:
                        self.append_log("Exclude mode disabled.")
                except Exception as e:
                    self.append_log(f"Failed to toggle exclude mode: {e}")

            exclude_btn.toggled.connect(_on_exclude_toggled)
            include_all_btn.clicked.connect(lambda: getattr(self.viewmodel, 'clear_exclusions', lambda: None)())

            try:
                self.viewmodel.files_updated.connect(self._on_files_updated)
            except Exception:
                pass
            self.file_list.currentRowChanged.connect(self._on_file_selected)
            remove_btn.clicked.connect(self._on_remove_file_clicked)
            clear_list_btn.clicked.connect(self._on_clear_files_clicked)
            # Element list wiring (connect if element widgets exist in a separate dock)
            if hasattr(self, 'element_list'):
                try:
                    self.element_list.currentRowChanged.connect(self._on_element_selected)
                except Exception:
                    pass
            if hasattr(self, 'element_add_btn'):
                try:
                    self.element_add_btn.clicked.connect(self._on_element_added_clicked)
                except Exception:
                    pass
            if hasattr(self, 'element_remove_btn'):
                try:
                    self.element_remove_btn.clicked.connect(self._on_element_remove_clicked)
                except Exception:
                    pass
            try:
                self.viewmodel.notify_file_queue()
            except Exception:
                pass
            try:
                self._refresh_element_list()
            except Exception:
                pass
        else:
            self._update_file_action_state()

        # ensure element list is populated even when no ViewModel present
        try:
            self._refresh_element_list()
        except Exception:
            pass

        self._update_file_action_state()

    def _on_files_updated(self, files):
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

    def _on_file_selected(self, row):
        self._update_file_action_state()
        if not self.viewmodel or row is None or row < 0:
            return
        try:
            self.viewmodel.activate_file(row)
        except Exception as e:
            self.append_log(f"Failed to load dataset: {e}")

    def _on_remove_file_clicked(self):
        if not self.viewmodel:
            return
        row = self.file_list.currentRow() if hasattr(self, "file_list") else -1
        if row < 0:
            return
        try:
            self.viewmodel.remove_file_at(row)
        except Exception as e:
            self.append_log(f"Failed to remove dataset: {e}")

    def _on_clear_files_clicked(self):
        if not self.viewmodel:
            return
        try:
            if hasattr(self.viewmodel, "clear_plot"):
                self.viewmodel.clear_plot()
            else:
                getattr(self.viewmodel, "clear_loaded_files", lambda: None)()
        except Exception as e:
            self.append_log(f"Failed to clear datasets: {e}")

    # --------------------------
    # Element list handlers
    # --------------------------
    def _on_element_selected(self, row):
        """Called when user selects an element in the Elements list.
        Tries to select the corresponding curve via the ViewModel if available
        otherwise falls back to local highlighting.
        """
        try:
            if not hasattr(self, 'element_list'):
                return
            if row is None or row < 0:
                return
            item = self.element_list.item(row)
            if item is None:
                return
            data = item.data(Qt.UserRole)
            elem_id = None
            if isinstance(data, dict):
                elem_id = data.get('id') or data.get('name') or item.text()
            else:
                elem_id = item.text()

            # prefer ViewModel selection API when available
            if self.viewmodel and hasattr(self.viewmodel, 'set_selected_curve'):
                try:
                    self.viewmodel.set_selected_curve(elem_id)
                except Exception:
                    pass
            else:
                try:
                    # fallback to view-level highlighting
                    self._on_curve_selected(elem_id)
                except Exception:
                    pass
        except Exception as e:
            try:
                self.append_log(f"Element select failed: {e}")
            except Exception:
                pass

    def _on_element_added_clicked(self):
        """Add a new element. Calls ViewModel.add_component_to_model() if present;
        otherwise adds a local placeholder item so the UI remains usable.
        """
        try:
            if self.viewmodel and hasattr(self.viewmodel, 'add_component_to_model'):
                try:
                    self.viewmodel.add_component_to_model()
                except Exception as e:
                    self.append_log(f"ViewModel failed to add component: {e}")
                # allow ViewModel to emit updates; refresh local list
                try:
                    self._refresh_element_list()
                except Exception:
                    pass
                return

            # local placeholder behavior
            name = f"element{self.element_list.count() + 1}"
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, {'id': name, 'name': name})
            self.element_list.addItem(item)
            self.element_list.setCurrentRow(self.element_list.count() - 1)
            self.append_log(f"Added placeholder element: {name}")
        except Exception as e:
            try:
                self.append_log(f"Failed to add element: {e}")
            except Exception:
                pass

    def _on_element_remove_clicked(self):
        """Remove the currently selected element. Attempts to call ViewModel
        removal APIs if present, otherwise removes from the local list.
        """
        try:
            if not hasattr(self, 'element_list'):
                return
            row = self.element_list.currentRow()
            if row is None or row < 0:
                return

            # Prevent removing the special 'model' entry or the last remaining element
            total = self.element_list.count() if hasattr(self, 'element_list') else 0
            # if only one item remains, disallow removal
            if total <= 1:
                try:
                    self.append_log("Cannot remove the last element.")
                except Exception:
                    pass
                return

            # inspect selected item data to prevent removing the model placeholder
            try:
                it = self.element_list.item(row)
                itdata = it.data(Qt.UserRole) if it is not None else None
                if isinstance(itdata, dict) and itdata.get('id') == 'model':
                    try:
                        self.append_log("Cannot remove the active model entry.")
                    except Exception:
                        pass
                    return
            except Exception:
                pass

            # prefer ViewModel removal APIs
            if self.viewmodel and hasattr(self.viewmodel, 'remove_component_at'):
                try:
                    self.viewmodel.remove_component_at(row)
                    self._refresh_element_list()
                    return
                except Exception:
                    pass
            if self.viewmodel and hasattr(self.viewmodel, 'remove_last_component'):
                try:
                    self.viewmodel.remove_last_component()
                    self._refresh_element_list()
                    return
                except Exception:
                    pass

            # fallback: remove from UI only
            item = self.element_list.item(row)
            # protect model item
            if item is not None:
                try:
                    dat = item.data(Qt.UserRole)
                    if isinstance(dat, dict) and dat.get('id') == 'model':
                        try:
                            self.append_log("Cannot remove the active model entry.")
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
            taken = self.element_list.takeItem(row)
            if taken:
                self.append_log(f"Removed element: {taken.text()}")
            # clear selection in ViewModel if present
            try:
                if self.viewmodel and hasattr(self.viewmodel, 'set_selected_curve'):
                    self.viewmodel.set_selected_curve(None)
                else:
                    self._on_curve_selected(None)
            except Exception:
                pass
        except Exception as e:
            try:
                self.append_log(f"Failed to remove element: {e}")
            except Exception:
                pass

    def _refresh_element_list(self):
        """Populate the element list from the ViewModel/state when possible.
        This is tolerant of non-composite specs and will try to infer prefixes
        from parameter names as a fallback.
        """
        try:
            if not hasattr(self, 'element_list'):
                return
            self.element_list.blockSignals(True)
            self.element_list.clear()
            if self.viewmodel and hasattr(self.viewmodel, 'state') and hasattr(self.viewmodel.state, 'model_spec'):
                spec = self.viewmodel.state.model_spec
                comps = getattr(spec, '_components', None)
                if isinstance(comps, list) and len(comps) > 0:
                    for prefix, sub in comps:
                        name = prefix.rstrip('_')
                        item = QListWidgetItem(name)
                        item.setData(Qt.UserRole, {'id': prefix, 'name': name})
                        self.element_list.addItem(item)
                else:
                    # infer prefixes from param names (prefix_name pattern)
                    params = getattr(spec, 'params', {}) or {}
                    prefixes = {}
                    for k in params.keys():
                        if '_' in k:
                            p = k.split('_', 1)[0] + '_'
                            prefixes[p] = prefixes.get(p, 0) + 1
                    for p in sorted(prefixes.keys()):
                        name = p.rstrip('_')
                        item = QListWidgetItem(name)
                        item.setData(Qt.UserRole, {'id': p, 'name': name})
                        self.element_list.addItem(item)
            # Prepend an entry representing the active/current model so it's always visible
            try:
                model_label = None
                if self.viewmodel and hasattr(self.viewmodel, 'state'):
                    model_label = getattr(self.viewmodel.state, 'model_name', None)
                if not model_label:
                    model_label = 'Model'
                # Avoid duplicate if a component uses the same id
                first = self.element_list.item(0)
                # Safely determine whether the first item is the injected 'model'
                first_has_model_id = False
                if first is not None:
                    try:
                        fd = first.data(Qt.UserRole)
                        if isinstance(fd, dict) and fd.get('id') == 'model':
                            first_has_model_id = True
                    except Exception:
                        first_has_model_id = False
                if first is None or not first_has_model_id:
                    model_item = QListWidgetItem(str(model_label))
                    model_item.setData(Qt.UserRole, {'id': 'model', 'name': model_label})
                    # make it visually distinct (bold + subtle background)
                    try:
                        f = model_item.font()
                        f.setBold(True)
                        model_item.setFont(f)
                    except Exception:
                        pass
                    try:
                        model_item.setBackground(QBrush(QColor("#f2f2f7")))
                        model_item.setToolTip("Active model (non-removable)")
                    except Exception:
                        pass
                    self.element_list.insertItem(0, model_item)
            except Exception:
                pass
            # If nothing detected (no composite components or inferred prefixes),
            # ensure we include the active/current model as a single element so
            # the UI always shows at least one selectable entry.
            if self.element_list.count() == 0:
                try:
                    name = None
                    if self.viewmodel and hasattr(self.viewmodel, 'state'):
                        name = getattr(self.viewmodel.state, 'model_name', None)
                    if not name:
                        name = "Model"
                    # attempt to annotate with a center value if available
                    tooltip = None
                    try:
                        mdl = getattr(self.viewmodel.state, 'model', None) if self.viewmodel and hasattr(self.viewmodel, 'state') else None
                        if mdl is not None:
                            center_val = None
                            if hasattr(mdl, 'center'):
                                center_val = getattr(mdl, 'center')
                            elif hasattr(mdl, 'Center'):
                                center_val = getattr(mdl, 'Center')
                            if center_val is not None:
                                tooltip = f"center={center_val}"
                    except Exception:
                        tooltip = None

                    item = QListWidgetItem(str(name))
                    item.setData(Qt.UserRole, {'id': 'model', 'name': name})
                    if tooltip:
                        item.setToolTip(tooltip)
                    self.element_list.addItem(item)
                except Exception:
                    pass

            self.element_list.blockSignals(False)
        except Exception:
            try:
                self.element_list.blockSignals(False)
            except Exception:
                pass

    def _update_file_action_state(self):
        if not hasattr(self, "file_list"):
            return
        has_files = self.file_list.count() > 0
        has_selection = self.file_list.currentRow() >= 0
        if hasattr(self, "file_remove_btn"):
            self.file_remove_btn.setEnabled(has_selection)
        if hasattr(self, "file_clear_btn"):
            # Allow Clear List to be used even when the visible list is empty.
            # This helps when a file is loaded but for some reason doesn't appear
            # in the list — the user can still force-clear queued/loaded state.
            self.file_clear_btn.setEnabled(True)

    def _init_right_dock(self):
        # Replaced static parameter controls with a dynamic, scrollable form.
        self.right_dock = QDockWidget("Parameters", self)
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # Model selector placed at the top of the parameters panel
        self.model_selector = QComboBox()
        # Populate model selector dynamically from discovered model specs where possible.
        try:
            # lazy import to avoid circular imports at module import time
            from models import get_available_model_names
            import re

            def _pretty(name: str) -> str:
                # remove trailing 'ModelSpec' and split CamelCase into words
                s = re.sub(r"ModelSpec$", "", name)
                s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
                return s.strip()

            spec_class_names = get_available_model_names()
            display_names = [_pretty(n) for n in spec_class_names]
            # ensure we have at least the historical defaults as fallback
            if display_names:
                self.model_selector.addItems(display_names)
            else:
                self.model_selector.addItems(["Voigt", "Gaussian"])
        except Exception:
            # Fall back to the original static list if discovery fails
            self.model_selector.addItems(["Voigt", "Gaussian"])
        vlayout.addWidget(QLabel("Model:"))
        vlayout.addWidget(self.model_selector)

        # Chi-squared display (placed above the parameter list)
        self.chi_label = QLabel("Chi-squared: N/A")
        try:
            f = self.chi_label.font()
            f.setPointSize(max(8, f.pointSize() - 1))
            self.chi_label.setFont(f)
        except Exception:
            pass
        vlayout.addWidget(self.chi_label)

        # set initial selection from viewmodel/state if available
        try:
            if self.viewmodel:
                initial = getattr(self.viewmodel.state, "model_name", None)
                if initial:
                    idx = self.model_selector.findText(str(initial), Qt.MatchFixedString)
                    if idx >= 0:
                        self.model_selector.setCurrentIndex(idx)
        except Exception:
            pass

        # when user changes model, instruct viewmodel and refresh the parameter panel
        def _on_model_selected(idx):
            name = self.model_selector.currentText()
            try:
                if self.viewmodel and hasattr(self.viewmodel, "set_model"):
                    self.viewmodel.set_model(name)
                # refresh UI parameters for the selected model
                self._refresh_parameters()
                # ensure elements list reflects the newly selected model
                try:
                    self._refresh_element_list()
                except Exception:
                    pass
            except Exception as e:
                self.append_log(f"Failed to switch model: {e}")

        self.model_selector.currentIndexChanged.connect(_on_model_selected)

        # Scrollable area to hold the form (so many parameters fit)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)

        # initial empty form widget (will be replaced by _populate_parameters)
        self.param_form_widget = QWidget()
        self.param_form = QFormLayout(self.param_form_widget)
        self.param_scroll.setWidget(self.param_form_widget)

        vlayout.addWidget(self.param_scroll)

        # Apply + Refresh buttons
        btn_h = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        refresh_btn = QPushButton("Refresh")
        btn_h.addWidget(apply_btn)
        btn_h.addWidget(refresh_btn)
        vlayout.addLayout(btn_h)

        self.right_dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        # Make the parameters dock wider by default so controls and hints are visible
        try:
            # A minimum width allows the user to resize smaller/larger while
            # providing a comfortable default layout on startup.
            self.right_dock.setMinimumWidth(360)
        except Exception:
            pass

        # Connect to ViewModel (if present)
        if self.viewmodel:
            apply_btn.clicked.connect(self._on_apply_clicked)
            refresh_btn.clicked.connect(self._refresh_parameters)
            # initial populate
            self._refresh_parameters()
            # Update elements list whenever parameters (and model selection) change
            try:
                self.viewmodel.parameters_updated.connect(self._refresh_element_list)
            except Exception:
                pass

    def _init_elements_dock(self):
        """Create a separate dock on the right to host model elements (peaks).
        This sits beneath the Parameters dock and holds the element list and
        add/remove buttons.
        """
        self.elements_dock = QDockWidget("Elements", self)
        container = QWidget()
        vlayout = QVBoxLayout(container)

        # light-weight element list
        self.element_list = QListWidget()
        self.element_list.setSelectionMode(QAbstractItemView.SingleSelection)
        vlayout.addWidget(self.element_list)

        # add / remove row
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Element")
        remove_btn = QPushButton("Remove Element")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        vlayout.addLayout(btn_row)

        # keep references for other methods
        self.element_add_btn = add_btn
        self.element_remove_btn = remove_btn

        self.elements_dock.setWidget(container)
        # Prefer placing the Elements dock to the right of the Log (bottom)
        # dock so it appears at the lower-right corner of the UI. If the
        # bottom dock is not present, fall back to placing it in the right
        # column under Parameters (previous behavior).
        try:
            # attempt to add into the bottom area first
            self.addDockWidget(Qt.BottomDockWidgetArea, self.elements_dock)
            if hasattr(self, 'bottom_dock'):
                # split horizontally so the elements dock sits to the right of the Log
                self.splitDockWidget(self.bottom_dock, self.elements_dock, Qt.Horizontal)
        except Exception:
            try:
                # fallback: add to the right column and stack under Parameters
                self.addDockWidget(Qt.RightDockWidgetArea, self.elements_dock)
                if hasattr(self, 'right_dock'):
                    self.splitDockWidget(self.right_dock, self.elements_dock, Qt.Vertical)
            except Exception:
                pass
        try:
            self.elements_dock.setMinimumWidth(260)
        except Exception:
            pass

        # Wire UI handlers
        try:
            self.element_list.currentRowChanged.connect(self._on_element_selected)
        except Exception:
            pass
        try:
            add_btn.clicked.connect(self._on_element_added_clicked)
        except Exception:
            pass
        try:
            remove_btn.clicked.connect(self._on_element_remove_clicked)
        except Exception:
            pass

        # Populate initial contents
        try:
            self._refresh_element_list()
        except Exception:
            pass

    def _init_bottom_dock(self):
        self.bottom_dock = QDockWidget("Log", self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.append("System initialized.")
        self.bottom_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

    # --------------------------
    # View-only public methods
    # --------------------------
    def append_log(self, msg: str):
        # Safely append to the log widget if present; otherwise fall back to stdout.
        try:
            if hasattr(self, "log_text") and self.log_text is not None:
                self.log_text.append(msg)
            else:
                print(msg)
        except Exception:
            # avoid raising from logging
            try:
                print(msg)
            except Exception:
                pass

    def update_plot_data(self, x, y_data, y_fit=None, y_err=None):
        # Draw scatter points
        if x is None or y_data is None:
            return

        # Ensure numeric numpy arrays are used (prevents list subtraction errors in ErrorBarItem)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)

        # If the ViewModel exposes an exclusion mask, split included/excluded points
        excl_mask = None
        try:
            if self.viewmodel is not None and hasattr(self.viewmodel, "state") and hasattr(self.viewmodel.state, "excluded"):
                excl_mask = np.asarray(self.viewmodel.state.excluded, dtype=bool)
                # ensure mask length matches
                if len(excl_mask) != len(x_arr):
                    # mismatch — ignore mask and log for debugging
                    try:
                        self.append_log(f"Exclusion mask length {len(excl_mask)} != x length {len(x_arr)} — ignoring exclusions")
                    except Exception:
                        pass
                    excl_mask = None
        except Exception:
            excl_mask = None
        # normalize incoming error array as numpy array when present
        y_err_arr = None
        if y_err is not None:
            try:
                y_err_arr = np.asarray(y_err, dtype=float)
            except Exception:
                y_err_arr = None

        if excl_mask is None:
            # no exclusions — show all points in primary scatter
            self.scatter.setData(x=x_arr, y=y_arr)
            # clear excluded overlay
            if self.excluded_scatter is not None:
                try:
                    self.excluded_scatter.setData(x=np.array([], dtype=float), y=np.array([], dtype=float))
                except Exception:
                    pass
            # Draw vertical error bars when provided (all points)
            if y_err_arr is not None and len(y_err_arr) == len(y_arr):
                top = np.abs(y_err_arr)
                bottom = top
                self.err_item.setData(x=x_arr, y=y_arr, top=top, bottom=bottom)
            else:
                # clear error bars using numpy arrays (avoid passing Python lists)
                empty = np.array([], dtype=float)
                try:
                    self.err_item.setData(x=empty, y=empty, top=empty, bottom=empty)
                except Exception:
                    pass
        else:
            incl = ~excl_mask
            try:
                x_in = x_arr[incl]
                y_in = y_arr[incl]
            except Exception:
                x_in = np.array([], dtype=float)
                y_in = np.array([], dtype=float)
            try:
                x_ex = x_arr[excl_mask]
                y_ex = y_arr[excl_mask]
            except Exception:
                x_ex = np.array([], dtype=float)
                y_ex = np.array([], dtype=float)

            self.scatter.setData(x=x_in, y=y_in)
            if self.excluded_scatter is not None:
                try:
                    self.excluded_scatter.setData(x=x_ex, y=y_ex)
                except Exception:
                    pass

            # Error bars only for included points
            if y_err_arr is not None:
                try:
                    # if full-length, index it by included mask; if it already matches included length, use directly
                    if len(y_err_arr) == len(y_arr):
                        top = np.abs(y_err_arr[incl])
                        bottom = top
                        self.err_item.setData(x=x_in, y=y_in, top=top, bottom=bottom)
                    elif len(y_err_arr) == len(x_in):
                        top = np.abs(y_err_arr)
                        bottom = top
                        self.err_item.setData(x=x_in, y=y_in, top=top, bottom=bottom)
                    else:
                        empty = np.array([], dtype=float)
                        try:
                            self.err_item.setData(x=empty, y=empty, top=empty, bottom=empty)
                        except Exception:
                            pass
                except Exception:
                    pass

        # Fit line (if present)
        # Compute and display reduced chi-squared (2 decimals) and Cash statistic (delegated to ViewModel)
        try:
            # determine included arrays (preserve original inclusion logic)
            if excl_mask is None:
                xin = x_arr
                yin = y_arr
                if y_err_arr is not None and len(y_err_arr) == len(y_arr):
                    errin = y_err_arr
                else:
                    errin = None
                yfit_for_chi = None if y_fit is None else np.asarray(y_fit, dtype=float)
            else:
                xin = x_in if 'x_in' in locals() else np.array([], dtype=float)
                yin = y_in if 'y_in' in locals() else np.array([], dtype=float)
                if y_err_arr is None:
                    errin = None
                else:
                    if len(y_err_arr) == len(y_arr):
                        errin = y_err_arr[~excl_mask]
                    elif len(y_err_arr) == len(xin):
                        errin = y_err_arr
                    else:
                        errin = None
                if y_fit is None:
                    yfit_for_chi = None
                else:
                    yfit_full = np.asarray(y_fit, dtype=float)
                    if len(yfit_full) == len(x_arr):
                        yfit_for_chi = yfit_full[~excl_mask]
                    elif len(yfit_full) == len(xin):
                        yfit_for_chi = yfit_full
                    else:
                        yfit_for_chi = None

            reduced_str = "N/A"
            red_cash_str = "N/A"

            if self.viewmodel is not None and yfit_for_chi is not None and xin.size > 0:
                try:
                    n_params = max(0, len(self.param_widgets))
                    stats = self.viewmodel.compute_statistics(y_fit=yfit_for_chi, n_params=n_params)
                    red = stats.get("reduced_chi2")
                    red_cash = stats.get("reduced_cash")
                    if red is not None:
                        reduced_str = f"{red:.2f}"
                    if red_cash is not None:
                        red_cash_str = f"{red_cash:.2f}"
                except Exception:
                    reduced_str = "N/A"
                    red_cash_str = "N/A"

            label_text = f"reduced χ²={reduced_str}; reduced Cash={red_cash_str}"
            try:
                if hasattr(self, 'chi_label') and self.chi_label is not None:
                    self.chi_label.setText(label_text)
            except Exception:
                pass
        except Exception:
            try:
                if hasattr(self, 'chi_label') and self.chi_label is not None:
                    self.chi_label.setText("reduced χ²=N/A; Cash=N/A")
            except Exception:
                pass

        if y_fit is not None:
            yfit_arr = np.asarray(y_fit, dtype=float)
            if "fit" not in self.curves:
                curve = self.plot_widget.plot(x_arr, yfit_arr, pen=pg.mkPen(FIT_COLOR, width=2))
                self.curves["fit"] = curve
                curve.curve_id = "fit"
                # Enable clicking on curve
                curve.scene().sigMouseClicked.connect(lambda ev, cid="fit": self._on_curve_clicked(ev, cid))
            else:
                self.curves["fit"].setData(x_arr, yfit_arr)
        else:
            # remove fit curve if exists
            if "fit" in self.curves:
                self.plot_widget.removeItem(self.curves["fit"])
                del self.curves["fit"]


    def _on_apply_clicked(self):
        if not self.viewmodel:
            return

        # Collect values dynamically from widgets into a dict
        params = {}
        for name, widget in self.param_widgets.items():
            try:
                if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
                    params[name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    params[name] = widget.text()
                else:
                    # fallback: try to read "value" attribute if a custom widget
                    params[name] = getattr(widget, "value", lambda: None)()
            except Exception:
                params[name] = None

        # Send params dict to ViewModel; ViewModel handles validation/usage
        try:
            # ViewModel.apply_parameters should accept a dict of {name: value}
            self.viewmodel.apply_parameters(params)
            self.append_log("Parameters applied.")
        except Exception as e:
            self.append_log(f"Failed to apply parameters: {e}")

    # --------------------------
    # Dynamic parameter helpers
    # --------------------------
    def _refresh_parameters(self):
        """Ask the ViewModel for parameter specs and rebuild the form."""
        if not self.viewmodel:
            return
        try:
            specs = getattr(self.viewmodel, "get_parameters", lambda: {})()
            if specs is None:
                specs = {}
            # allow list-like or dict-like returns
            if isinstance(specs, list):
                # list of (name, spec) or list of names -> convert to dict
                converted = {}
                for item in specs:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        converted[item[0]] = item[1]
                    elif isinstance(item, str):
                        converted[item] = None
                specs = converted
            elif not isinstance(specs, dict):
                # unknown shape — nothing to do
                specs = {}
            self._populate_parameters(specs)
            self.append_log("Parameter panel refreshed.")
        except Exception as e:
            self.append_log(f"Failed to refresh parameters: {e}")

    def _populate_parameters(self, specs: dict):
        """Populate the parameter form from a specs dict.

        Each specs entry may be:
          name: value                      # infer type from value
          name: { 'value': v, 'type': 'float'|'int'|'str'|'bool'|'choice',
                  'min': ..., 'max': ..., 'choices': [...], 'decimals': ..., 'step': ... }
        Note: 'decimals' controls the number of decimal places shown by the
        QDoubleSpinBox (display precision). 'step' controls the single-step
        increment used when the user clicks the up/down arrows.
        """
        # Build a fresh form widget and replace the scroll area's widget
        new_widget = QWidget()
        new_form = QFormLayout(new_widget)
        new_param_widgets = {}

        for name, spec in specs.items():
            # Normalize spec into dict with at least 'value' and 'type'
            if isinstance(spec, dict):
                val = spec.get("value")
                ptype = spec.get("type", None)
            else:
                val = spec
                ptype = None

            # Infer type if not provided
            if ptype is None:
                if isinstance(val, bool):
                    ptype = "bool"
                elif isinstance(val, int) and not isinstance(val, bool):
                    ptype = "int"
                elif isinstance(val, float):
                    ptype = "float"
                elif isinstance(val, (list, tuple)):
                    ptype = "choice"
                else:
                    ptype = "str"

            widget = None
            try:
                if ptype == "float":
                    w = QDoubleSpinBox()
                    # Apply provided min/max if available, otherwise use safe large bounds
                    w.setRange(spec.get("min", -1e9), spec.get("max", 1e9) if isinstance(spec, dict) else 1e9)
                    # 'decimals' controls how many decimal places the spinbox displays.
                    w.setDecimals(spec.get("decimals", 6) if isinstance(spec, dict) else 6)
                    # 'step' controls the increment when the user clicks arrows or presses keys
                    w.setSingleStep(spec.get("step", 0.1) if isinstance(spec, dict) else 0.1)
                    if val is not None:
                        w.setValue(float(val))
                    widget = w

                elif ptype == "int":
                    w = QSpinBox()
                    w.setRange(int(spec.get("min", -2147483648)), int(spec.get("max", 2147483647)))
                    w.setSingleStep(int(spec.get("step", 1)))
                    if val is not None:
                        w.setValue(int(val))
                    widget = w

                elif ptype == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(val))
                    widget = w

                elif ptype == "choice":
                    w = QComboBox()
                    choices = []
                    if isinstance(spec, dict) and "choices" in spec:
                        choices = spec.get("choices") or []
                    elif isinstance(val, (list, tuple)):
                        choices = list(val)
                    for c in choices:
                        w.addItem(str(c))
                    # set current
                    cur = spec.get("value") if isinstance(spec, dict) else (val[0] if val else "")
                    if cur is not None:
                        idx = w.findText(str(cur))
                        if idx >= 0:
                            w.setCurrentIndex(idx)
                    widget = w

                else:  # str and fallback
                    w = QLineEdit()
                    if val is not None:
                        w.setText(str(val))
                    widget = w
            except Exception:
                # on any error, fallback to a simple line edit
                w = QLineEdit()
                if val is not None:
                    w.setText(str(val))
                widget = w

            # Keep the label text as the parameter name. If the spec provides an
            # interactive control hint, expose it both as a tooltip and as a
            # small visible hint label next to the control so users can discover
            # what mouse/keyboard actions affect the parameter.
            hint = ""
            try:
                if isinstance(spec, dict) and spec.get("control"):
                    ctrl = spec.get("control")
                    action = ctrl.get("action") or ""
                    mods = "+".join(ctrl.get("modifiers", [])) if ctrl.get("modifiers") else ""
                    hint = f"{action}" + (f"+{mods}" if mods else "")
                    try:
                        widget.setToolTip(f"Interactive: {hint}")
                    except Exception:
                        pass
            except Exception:
                pass

            # Container so we can show the widget and a right-hand hint label
            container_h = QWidget()
            hbox = QHBoxLayout(container_h)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.addWidget(widget)
            if hint:
                hint_label = QLabel(f"({hint})")
                # subtle visual style so it's unobtrusive but readable
                hint_label.setStyleSheet("color: gray; font-size: 11px;")
                hint_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                hint_label.setToolTip(f"Interactive control: {hint}")
                hbox.addWidget(hint_label)
            else:
                # keep layout stable when no hint exists
                hbox.addWidget(QLabel(""))

            new_form.addRow(name + ":", container_h)
            new_param_widgets[name] = widget

        # Replace the widget shown in the scroll area (frees old widgets)
        self.param_scroll.takeWidget()
        self.param_scroll.setWidget(new_widget)
        self.param_form_widget = new_widget
        self.param_form = new_form
        # Build control map for InputHandler based on specs' optional 'control' entries.
        control_map = {}
        try:
            for name, spec in specs.items():
                try:
                    if isinstance(spec, dict) and "control" in spec:
                        ctrl = spec.get("control") or {}
                        action = ctrl.get("action")
                        mods = tuple(sorted(ctrl.get("modifiers", []))) if ctrl.get("modifiers") is not None else tuple()
                        sensitivity = float(ctrl.get("sensitivity", 1.0))
                        key = (action, mods)
                        control_map.setdefault(key, []).append({"name": name, "sensitivity": sensitivity})
                except Exception:
                    continue
        except Exception:
            control_map = {}

        # Provide control map to input handler if available
        try:
            if hasattr(self, "input_handler") and self.input_handler is not None:
                self.input_handler.set_control_map(control_map)
        except Exception:
            pass

        self.param_widgets = new_param_widgets

        # No legacy fallbacks: the view exposes only the dynamic param_widgets mapping.
        # Consumers should read parameters from param_widgets or use the ViewModel API.

    # --------------------------
    # Config dialog (view-only)
    # --------------------------
    class _ConfigDialog(QDialog):
        def __init__(self, parent, cfg_dict):
            super().__init__(parent)
            self.setWindowTitle("Edit Configuration")
            form = QFormLayout(self)

            # default load folder
            self.load_edit = QLineEdit(self)
            self.load_edit.setText(str(cfg_dict.get("default_load_folder", "")))
            load_h = QHBoxLayout()
            load_h.addWidget(self.load_edit)
            browse_load = QPushButton("Browse")
            load_h.addWidget(browse_load)
            form.addRow("Default Load Folder:", load_h)
            browse_load.clicked.connect(self._browse_load)

            # default save folder
            self.save_edit = QLineEdit(self)
            self.save_edit.setText(str(cfg_dict.get("default_save_folder", "")))
            save_h = QHBoxLayout()
            save_h.addWidget(self.save_edit)
            browse_save = QPushButton("Browse")
            save_h.addWidget(browse_save)
            form.addRow("Default Save Folder:", save_h)
            browse_save.clicked.connect(self._browse_save)

            # buttons: Save, Cancel, and Reload (reload reads config from disk)
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            reload_btn = QPushButton("Reload")
            buttons.addButton(reload_btn, QDialogButtonBox.ActionRole)
            form.addRow(buttons)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            reload_btn.clicked.connect(self._on_reload_clicked)

        def _browse_load(self):
            d = QFileDialog.getExistingDirectory(self, "Select Default Load Folder", self.load_edit.text() or "")
            if d:
                self.load_edit.setText(d)

        def _browse_save(self):
            d = QFileDialog.getExistingDirectory(self, "Select Default Save Folder", self.save_edit.text() or "")
            if d:
                self.save_edit.setText(d)

        def _on_reload_clicked(self):
            """Reload configuration from disk via the parent ViewModel and refresh fields."""
            try:
                parent = self.parent()
                if parent and hasattr(parent, "viewmodel") and parent.viewmodel:
                    try:
                        parent.viewmodel.reload_config()
                    except Exception:
                        pass
                    try:
                        cfg = parent.viewmodel.get_config()
                        self.load_edit.setText(str(cfg.get("default_load_folder", "")))
                        self.save_edit.setText(str(cfg.get("default_save_folder", "")))
                        # also inform user in the main log
                        try:
                            parent.append_log("Configuration reloaded into dialog.")
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass

    def _on_edit_config_clicked(self):
        """Open the configuration editor dialog (view-only)."""
        if not self.viewmodel:
            self.append_log("No ViewModel available to edit configuration.")
            return

        # Ask ViewModel for current config values (ViewModel handles logic)
        try:
            cfg = self.viewmodel.get_config()
        except Exception as e:
            self.append_log(f"Failed to load configuration: {e}")
            return

        dlg = self._ConfigDialog(self, cfg)
        if dlg.exec() == QDialog.Accepted:
            # collect values and instruct ViewModel to save them
            new_load = dlg.load_edit.text().strip()
            new_save = dlg.save_edit.text().strip()
            try:
                ok = self.viewmodel.save_config(default_load_folder=new_load, default_save_folder=new_save)
                if ok:
                    self.append_log("Configuration saved.")
                    # reload ViewModel config and refresh UI as needed
                    if hasattr(self.viewmodel, "reload_config"):
                        try:
                            self.viewmodel.reload_config()
                        except Exception:
                            pass
                else:
                    self.append_log("Configuration save failed.")
            except Exception as e:
                self.append_log(f"Error saving configuration: {e}")

    def _on_curve_clicked(self, event, curve_id):
        """Handle mouse click on a curve."""
        if not self.viewmodel:
            return
        # Set this as the selected curve
        self.viewmodel.set_selected_curve(curve_id)

    def _on_curve_selected(self, curve_id):
        """Highlight the selected curve visually."""
        # Deselect previous
        if self.selected_curve_id and self.selected_curve_id in self.curves:
            curve = self.curves[self.selected_curve_id]
            curve.setPen(pg.mkPen(FIT_COLOR, width=2))
        self.selected_curve_id = curve_id

        # Highlight new
        if curve_id and curve_id in self.curves:
            curve = self.curves[curve_id]
            curve.setPen(pg.mkPen('red', width=4))
        self.append_log(f"Curve selection changed → {curve_id or 'none'}")


