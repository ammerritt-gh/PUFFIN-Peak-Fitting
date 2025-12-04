# view/main_window.py
# type: ignore
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout, QPushButton,
    QLabel, QTextEdit, QComboBox, QFormLayout, QDoubleSpinBox,
    QDialog, QLineEdit, QDialogButtonBox, QHBoxLayout, QFileDialog,
    QScrollArea, QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QGroupBox
)
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QBrush
import pyqtgraph as pg
import numpy as np
import math
import re
from functools import partial
import traceback

from view.view_box import CustomViewBox
from view.input_handler import InputHandler
from view.constants import CURVE_SELECT_TOL_PIXELS
from view.docks.controls_dock import ControlsDock
from view.docks.parameters_dock import ParametersDock
from view.docks.elements_dock import ElementsDock
from view.docks.log_dock import LogDock
from view.docks.resolution_dock import ResolutionDock
from models import CompositeModelSpec

# -- color palette (change these) --
PLOT_BG = "white"       # plot background
POINT_COLOR = "black"   # scatter points
ERROR_COLOR = "black"   # error bars (can match points)
FIT_COLOR = "purple"     # fit line
AXIS_COLOR = "black"    # axis and tick labels
GRID_ALPHA = 0.5

class MainWindow(QMainWindow):
    # Emitted whenever the parameter panel is rebuilt/refreshed. Listeners
    # can use this to re-wire per-widget signals (valueChanged, editingFinished, etc.).
    parameters_updated = Signal()
    def __init__(self, viewmodel=None):
        super().__init__()
        self.setWindowTitle("PUMA Peak Fitter")
        self.viewmodel = viewmodel
        self.param_widgets = {}   # name -> widget
        self._param_last_values = {}
        self._building_params = False
        self.curves = {}  # curve_id -> PlotDataItem
        self.selected_curve_id = None
        # internal: ensure we connect scene click handler only once
        self._scene_click_connected = False
        self.legend = None
        self._legend_entries = {}
        self._component_legend_order = []
        self._fit_label = "Total Fit"
        self._excluded_has_points = False

        # --- Central Plot ---
        # Initialize the plot widget inside _init_plot() to avoid duplicate
        # creation and ensure a single central widget is used.
        self._init_plot()

        # --- Docks ---
        self._init_docks()

        for dock in [self.controls_dock, self.parameters_dock, self.log_dock]:
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
            except Exception as e:
                try:
                    self._log_exception("Failed to set viewbox input backref", e)
                except Exception:
                    pass
        except Exception as e:
            try:
                self._log_exception("Failed to install viewbox event filter", e)
            except Exception:
                pass

        # connect ViewBox interaction signals → ViewModel
        if self.viewmodel:
            self.viewbox.peakSelected.connect(self._on_peak_selected)
            # use centralized handle_action when available for deselect
            try:
                if hasattr(self.viewmodel, "handle_action"):
                    self.viewbox.peakDeselected.connect(lambda: self.viewmodel.handle_action("set_selected_curve", curve_id=None))
                else:
                    self.viewbox.peakDeselected.connect(lambda: self.viewmodel.set_selected_curve(None))
            except Exception as e:
                try:
                    self._log_exception("Failed to connect peakDeselected handler", e)
                except Exception:
                    pass
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

        # legend showing data, fits, and components
        try:
            self.legend = self.plot_widget.addLegend(offset=(12, 12))
        except Exception:
            self.legend = None
        self._legend_entries = {}

    # Note: fit/component overlays are managed by update_plot_data via self.curves
    # so we avoid creating separate persistent PlotDataItems here and prevent
    # duplicate plot items.

        # axis colors
        for ax in ("left", "bottom", "right", "top"):
            try:
                axis = self.plot_widget.getAxis(ax)
                axis.setPen(pg.mkPen(AXIS_COLOR))
                axis.setTextPen(pg.mkPen(AXIS_COLOR))
            except Exception as e:
                try:
                    self._log_exception("Failed to set axis colors", e)
                except Exception:
                    pass

        self._update_legend()


    # --------------------------
    # Plot interaction callbacks
    # --------------------------
    def _on_peak_selected(self, x, y):
        # Require the click to land near an existing curve before responding.
        current_curve = self.selected_curve_id
        target_curve = None
        try:
            if self.input_handler is not None:
                target_curve = self.input_handler.detect_curve_at(self.plot_widget, x, y)
        except Exception:
            target_curve = None

        if target_curve is None:
            self.append_log(f"Peak click ignored — no peak near ({x:.3f}, {y:.3f}).")
            try:
                self.viewbox.clear_selection()
            except Exception as e:
                try:
                    self._log_exception("Failed to clear selection on viewbox", e)
                except Exception:
                    pass
            if current_curve is not None and self.viewmodel is not None:
                try:
                    self.viewmodel.set_selected_curve(current_curve)
                except Exception as e:
                    try:
                        self._log_exception("Failed to restore previous selected curve", e)
                    except Exception:
                        pass
            return

        if target_curve != self.selected_curve_id and self.viewmodel is not None:
            try:
                self.viewmodel.set_selected_curve(target_curve)
            except Exception as e:
                try:
                    self._log_exception("Failed to set selected curve", e)
                except Exception:
                    pass

        # Ensure we have a target curve before responding to the peak click so drag works.
        if not self._ensure_curve_selection_for_peaks():
            self.append_log("Peak clicked but no curve selected — ignored.")
            return

        if self.viewmodel and hasattr(self.viewmodel, "on_peak_selected"):
            self.viewmodel.on_peak_selected(x, y)
        self.append_log(f"Selected peak near ({x:.3f}, {y:.3f})")

    def _on_peak_moved(self, info):
        # Only forward peak movement when a curve is selected
        if not self._ensure_curve_selection_for_peaks():
            self.append_log("Peak moved but no curve selected — ignored.")
            return

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
    def _init_docks(self):
        """Initialize all dock widgets using the modular dock classes."""
        # Create the log dock first so logging is available immediately
        self.log_dock = LogDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        
        # Create controls dock (left side)
        self.controls_dock = ControlsDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_dock)
        
        # Create parameters dock (right side)
        self.parameters_dock = ParametersDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameters_dock)
        
        # Create elements dock (bottom-right, split with log)
        self.elements_dock = ElementsDock(self)
        try:
            self.addDockWidget(Qt.BottomDockWidgetArea, self.elements_dock)
            self.splitDockWidget(self.log_dock, self.elements_dock, Qt.Horizontal)
        except Exception:
            try:
                self.addDockWidget(Qt.RightDockWidgetArea, self.elements_dock)
                self.splitDockWidget(self.parameters_dock, self.elements_dock, Qt.Vertical)
            except Exception:
                pass
        
        try:
            self.elements_dock.setMinimumWidth(260)
        except Exception:
            pass
        
        # Create resolution dock (floating, initially hidden)
        self.resolution_dock = ResolutionDock(self)
        self.resolution_dock.hide()  # Start hidden; button reopens it
        
        # Set up backward compatibility aliases for old attribute names
        self.left_dock = self.controls_dock
        self.right_dock = self.parameters_dock
        self.bottom_dock = self.log_dock
        
        # Expose key widgets for backward compatibility
        self.log_text = self.log_dock.log_text
        self.file_list = self.controls_dock.file_list
        self.exclude_btn = self.controls_dock.exclude_btn
        self.include_all_btn = self.controls_dock.include_all_btn
        self.file_remove_btn = self.controls_dock.remove_btn
        self.file_clear_btn = self.controls_dock.clear_list_btn
        self.element_list = self.elements_dock.element_list
        self.element_add_btn = self.elements_dock.add_btn
        self.element_remove_btn = self.elements_dock.remove_btn
        self.model_selector = self.parameters_dock.model_selector
        self.chi_label = self.parameters_dock.chi_label
        self.param_scroll = self.parameters_dock.param_scroll
        self.param_form_widget = self.parameters_dock.param_form_widget
        self.param_form = self.parameters_dock.param_form
        self.param_widgets = self.parameters_dock.param_widgets
        self._param_last_values = self.parameters_dock._param_last_values
        
        # Wire up viewbox exclude mode to controls dock
        try:
            if hasattr(self, "viewbox") and self.viewbox is not None:
                self.controls_dock.exclude_btn.toggled.connect(self.viewbox.set_exclude_mode)
                if hasattr(self.viewbox, "excludeModeChanged"):
                    try:
                        self.viewbox.excludeModeChanged.connect(self.controls_dock.set_exclude_button_checked)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Connect dock signals to handlers
        self._wire_dock_signals()

    def _wire_dock_signals(self):
        """Wire up all dock signals to main window handlers."""
        if not self.viewmodel:
            self._update_file_action_state()
            return
        
        # Controls dock signals
        try:
            if hasattr(self.viewmodel, "handle_action"):
                self.controls_dock.load_data_clicked.connect(lambda: self.viewmodel.handle_action("load_data"))
                self.controls_dock.save_data_clicked.connect(lambda: self.viewmodel.handle_action("save_data"))
                self.controls_dock.run_fit_clicked.connect(lambda: self.viewmodel.handle_action("run_fit"))
                self.controls_dock.update_plot_clicked.connect(lambda: self.viewmodel.handle_action("update_plot"))
            else:
                self.controls_dock.load_data_clicked.connect(self.viewmodel.load_data)
                self.controls_dock.save_data_clicked.connect(self.viewmodel.save_data)
                self.controls_dock.run_fit_clicked.connect(self.viewmodel.run_fit)
                self.controls_dock.update_plot_clicked.connect(self.viewmodel.update_plot)
        except Exception:
            try:
                self.controls_dock.load_data_clicked.connect(self.viewmodel.load_data)
            except Exception:
                pass
            try:
                self.controls_dock.save_data_clicked.connect(self.viewmodel.save_data)
            except Exception:
                pass
            try:
                self.controls_dock.run_fit_clicked.connect(self.viewmodel.run_fit)
            except Exception:
                pass
            try:
                self.controls_dock.update_plot_clicked.connect(self.viewmodel.update_plot)
            except Exception:
                pass
        
        self.controls_dock.edit_config_clicked.connect(self._on_edit_config_clicked)
        self.controls_dock.exclude_toggled.connect(self._on_exclude_toggled)
        self.controls_dock.resolution_clicked.connect(self._on_resolution_clicked)
        
        try:
            if hasattr(self.viewmodel, "handle_action"):
                self.controls_dock.include_all_clicked.connect(lambda: self.viewmodel.handle_action("clear_exclusions"))
            else:
                self.controls_dock.include_all_clicked.connect(lambda: getattr(self.viewmodel, 'clear_exclusions', lambda: None)())
        except Exception:
            self.controls_dock.include_all_clicked.connect(lambda: getattr(self.viewmodel, 'clear_exclusions', lambda: None)())
        
        try:
            self.viewmodel.files_updated.connect(self.controls_dock.update_files)
        except Exception:
            pass
        
        self.controls_dock.file_selected.connect(self._on_file_selected)
        self.controls_dock.remove_file_clicked.connect(self._on_remove_file_clicked)
        self.controls_dock.clear_files_clicked.connect(self._on_clear_files_clicked)
        
        # Parameters dock signals
        self.parameters_dock.model_changed.connect(self._on_model_changed)
        self.parameters_dock.apply_clicked.connect(self._on_apply_clicked)
        self.parameters_dock.refresh_clicked.connect(self._refresh_parameters)
        self.parameters_dock.parameter_changed.connect(self._on_parameter_changed)
        
        # Set initial model selection
        try:
            initial = getattr(self.viewmodel.state, "model_name", None)
            if initial:
                self.parameters_dock.set_model_selector(initial)
        except Exception:
            pass
        
        # Initial parameter population
        self._refresh_parameters()
        
        # Update elements list whenever parameters change
        try:
            self.viewmodel.parameters_updated.connect(self._refresh_element_list)
        except Exception:
            pass
        
        # Elements dock signals
        self.elements_dock.element_selected.connect(self._on_element_selected)
        self.elements_dock.element_add_clicked.connect(self._on_element_added_clicked)
        self.elements_dock.element_remove_clicked.connect(self._on_element_remove_clicked)
        self.elements_dock.element_rows_moved.connect(self._on_element_rows_moved)
        
        # Resolution dock signals
        self.resolution_dock.resolution_model_changed.connect(self._on_resolution_model_changed)
        self.resolution_dock.resolution_parameter_changed.connect(self._on_resolution_parameter_changed)
        self.resolution_dock.resolution_apply_clicked.connect(self._on_resolution_apply_clicked)
        
        # Initial population
        try:
            self.viewmodel.notify_file_queue()
        except Exception:
            pass
        
        try:
            self._refresh_element_list()
        except Exception:
            pass
        
        self._update_file_action_state()

    def _on_exclude_toggled(self, checked):
        """Toggle exclude mode in the ViewBox and log the change."""
        try:
            if hasattr(self, 'viewbox') and self.viewbox is not None:
                self.viewbox.set_exclude_mode(bool(checked))
            
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

    def _on_model_changed(self, model_name):
        """Handle model selection change from parameters dock."""
        try:
            if self.viewmodel:
                if hasattr(self.viewmodel, "handle_action"):
                    try:
                        self.viewmodel.handle_action("set_model", model_name=model_name)
                    except Exception:
                        if hasattr(self.viewmodel, "set_model"):
                            try:
                                self.viewmodel.set_model(model_name)
                            except Exception:
                                raise
                else:
                    if hasattr(self.viewmodel, "set_model"):
                        self.viewmodel.set_model(model_name)
            
            self._refresh_parameters()
            
            try:
                self._refresh_element_list()
            except Exception:
                pass
        except Exception as e:
            self.append_log(f"Failed to switch model: {e}")

    def _on_parameter_changed(self, name, value):
        """Handle parameter change from parameters dock."""
        if not self.viewmodel:
            return
        
        try:
            # If this is a fixed-toggle checkbox, handle widget enabling
            if isinstance(name, str) and name.endswith("__fixed"):
                base = name[: -len("__fixed")]
                try:
                    val_widget = self.param_widgets.get(base)
                    if val_widget is not None:
                        try:
                            val_widget.setEnabled(not bool(value))
                        except Exception:
                            pass
                except Exception:
                    pass
            
            # Apply parameter via viewmodel
            if hasattr(self.viewmodel, "handle_action"):
                try:
                    self.viewmodel.handle_action("apply_parameters", params={name: value})
                except Exception:
                    try:
                        self.viewmodel.apply_parameters({name: value})
                    except Exception as exc:
                        self.append_log(f"Failed to update parameter '{name}': {exc}")
            else:
                try:
                    self.viewmodel.apply_parameters({name: value})
                except Exception as exc:
                    self.append_log(f"Failed to update parameter '{name}': {exc}")
        except Exception as exc:
            self.append_log(f"Failed to update parameter '{name}': {exc}")

    def _on_files_updated(self, files):
        # Delegate to controls dock (which is already connected in _wire_dock_signals)
        # This method kept for backward compatibility but is no longer needed
        pass

    def _on_file_selected(self, row):
        self._update_file_action_state()
        if not self.viewmodel or row is None or row < 0:
            return
        try:
            if hasattr(self.viewmodel, "handle_action"):
                try:
                    self.viewmodel.handle_action("activate_file", index=int(row))
                except Exception:
                    self.viewmodel.activate_file(row)
            else:
                self.viewmodel.activate_file(row)
        except Exception as e:
            self.append_log(f"Failed to load dataset: {e}")

    def _on_remove_file_clicked(self):
        if not self.viewmodel or not hasattr(self.viewmodel, "handle_action"):
            return
        row = self.file_list.currentRow() if hasattr(self, "file_list") else -1
        if row < 0:
            return
        try:
            if hasattr(self.viewmodel, "handle_action"):
                try:
                    self.viewmodel.handle_action("remove_file_at", index=int(row))
                except Exception:
                    self.viewmodel.remove_file_at(row)
            else:
                self.viewmodel.remove_file_at(row)
        except Exception as e:
            self.append_log(f"Failed to remove dataset: {e}")

    def _on_clear_files_clicked(self):
        if not self.viewmodel:
            return
        try:
            if hasattr(self.viewmodel, "handle_action"):
                try:
                    # prefer clear_plot which also clears config
                    self.viewmodel.handle_action("clear_plot")
                except Exception:
                    try:
                        getattr(self.viewmodel, "clear_loaded_files", lambda: None)()
                    except Exception:
                        pass
            else:
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

            elem_id = str(elem_id)
            curve_id = None
            if elem_id == 'model':
                if self._composite_has_components():
                    curve_id = None
                else:
                    curve_id = 'fit'
            else:
                curve_id = f"component:{elem_id}"

            # prefer ViewModel selection API when available
            if self.viewmodel and hasattr(self.viewmodel, 'set_selected_curve'):
                try:
                    self.viewmodel.set_selected_curve(curve_id)
                except Exception:
                    pass
            else:
                try:
                    # fallback to view-level highlighting
                    self._on_curve_selected(curve_id)
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
            if not self.viewmodel:
                return

            spec = getattr(self.viewmodel.state, "model_spec", None)
            if not isinstance(spec, CompositeModelSpec):
                self.append_log("Switch to the Custom model before adding elements.")
                return

            available = []
            try:
                if hasattr(self.viewmodel, "get_available_component_names"):
                    available = self.viewmodel.get_available_component_names()
            except Exception:
                available = []
            if not available:
                available = ["Gaussian", "Voigt", "Linear Background"]

            dialog = self._AddElementDialog(self, available)
            if dialog.exec() == QDialog.Accepted:
                component_name = dialog.selected_component
                initial_params = dialog.initial_params or {}
                if component_name:
                    added = False
                    try:
                        if hasattr(self.viewmodel, "handle_action"):
                            try:
                                res = self.viewmodel.handle_action("add_component_to_model", component_name=component_name, initial_params=initial_params)
                                added = bool(res)
                            except Exception:
                                # fallback to direct method
                                added = self.viewmodel.add_component_to_model(component_name, initial_params)
                        else:
                            added = self.viewmodel.add_component_to_model(component_name, initial_params)
                    except Exception as e:
                        self.append_log(f"Failed to add component: {e}")
                        added = False
                    if added:
                        try:
                            self._refresh_element_list()
                        except Exception:
                            pass
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

            # prefer ViewModel removal APIs (go through dispatcher when present)
            try:
                if self.viewmodel:
                    # try remove at index first
                    if hasattr(self.viewmodel, "handle_action"):
                        try:
                            res = self.viewmodel.handle_action("remove_component_at", index=int(row - 1))
                            if res:
                                self._refresh_element_list()
                                return
                        except Exception:
                            pass
                        try:
                            res = self.viewmodel.handle_action("remove_last_component")
                            if res:
                                self._refresh_element_list()
                                return
                        except Exception:
                            pass
                    else:
                        if hasattr(self.viewmodel, 'remove_component_at'):
                            try:
                                self.viewmodel.remove_component_at(row - 1)
                                self._refresh_element_list()
                                return
                            except Exception:
                                pass
                        if hasattr(self.viewmodel, 'remove_last_component'):
                            try:
                                self.viewmodel.remove_last_component()
                                self._refresh_element_list()
                                return
                            except Exception:
                                pass
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

    def _on_element_rows_moved(self, parent, start, end, dest_parent, dest_row):
        """Sync drag-and-drop ordering back to the ViewModel."""
        try:
            if not self.viewmodel:
                return
            if start <= 0:
                # keep the model entry fixed at the top
                self._refresh_element_list()
                return
            first = self.element_list.item(0) if hasattr(self, 'element_list') else None
            if first is None:
                return
            first_data = first.data(Qt.UserRole) if isinstance(first, QListWidgetItem) else None
            if not isinstance(first_data, dict) or first_data.get('id') != 'model':
                self._refresh_element_list()
                return
            prefixes = []
            count = self.element_list.count() if hasattr(self, 'element_list') else 0
            for idx in range(1, count):
                item = self.element_list.item(idx)
                if item is None:
                    continue
                data = item.data(Qt.UserRole)
                if isinstance(data, dict):
                    prefix = data.get('id')
                    if prefix and prefix != 'model':
                        prefixes.append(prefix)
            if not prefixes:
                return
            # Prefer routing reorders through the centralized dispatcher
            try:
                if hasattr(self.viewmodel, "handle_action"):
                    try:
                        res = self.viewmodel.handle_action("reorder_components_by_prefix", prefix_order=prefixes)
                        if not res and end == start:
                            # best-effort fallback: try single-item reorder
                            old_index = start - 1
                            new_index = max(0, dest_row - 1 if dest_row <= count else count - 1)
                            if new_index >= len(prefixes):
                                new_index = len(prefixes) - 1
                            if new_index != old_index:
                                self.viewmodel.handle_action("reorder_component", old_index=old_index, new_index=new_index)
                    except Exception:
                        # final fallback: try direct methods
                        if hasattr(self.viewmodel, 'reorder_components_by_prefix'):
                            try:
                                self.viewmodel.reorder_components_by_prefix(prefixes)
                            except Exception:
                                pass
                        elif hasattr(self.viewmodel, 'reorder_component') and end == start:
                            old_index = start - 1
                            new_index = max(0, dest_row - 1 if dest_row <= count else count - 1)
                            if new_index >= len(prefixes):
                                new_index = len(prefixes) - 1
                            if new_index != old_index:
                                try:
                                    self.viewmodel.reorder_component(old_index, new_index)
                                except Exception:
                                    pass
                else:
                    if hasattr(self.viewmodel, 'reorder_components_by_prefix'):
                        self.viewmodel.reorder_components_by_prefix(prefixes)
                    elif hasattr(self.viewmodel, 'reorder_component') and end == start:
                        # fallback best-effort: compute positions relative to list excluding model
                        old_index = start - 1
                        new_index = max(0, dest_row - 1 if dest_row <= count else count - 1)
                        if new_index >= len(prefixes):
                            new_index = len(prefixes) - 1
                        if new_index != old_index:
                            self.viewmodel.reorder_component(old_index, new_index)
                    else:
                        self._refresh_element_list()
            except Exception:
                self._refresh_element_list()
        except Exception as e:
            try:
                self.append_log(f"Failed to reorder elements: {e}")
            except Exception:
                pass

    def _refresh_element_list(self):
        """Populate the element list from the ViewModel/state when possible.
        This is tolerant of non-composite specs and will try to infer prefixes
        from parameter names as a fallback.
        """
        try:
            descriptors = []
            if self.viewmodel and hasattr(self.viewmodel, 'get_component_descriptors'):
                try:
                    descriptors = self.viewmodel.get_component_descriptors() or []
                except Exception:
                    descriptors = []
            
            # Add inferred prefixes if no descriptors
            if not descriptors and self.viewmodel and hasattr(self.viewmodel, 'state') and hasattr(self.viewmodel.state, 'model_spec'):
                spec = self.viewmodel.state.model_spec
                params = getattr(spec, 'params', {}) or {}
                prefixes = {}
                for k in params.keys():
                    if '_' in k:
                        p = k.split('_', 1)[0] + '_'
                        prefixes[p] = prefixes.get(p, 0) + 1
                for p in sorted(prefixes.keys()):
                    name = p.rstrip('_')
                    descriptors.append({'prefix': p, 'label': name})
            
            # Add model entry at the beginning
            model_label = 'Model'
            if self.viewmodel and hasattr(self.viewmodel, 'state'):
                model_label = getattr(self.viewmodel.state, 'model_name', None) or 'Model'
            
            model_desc = {'prefix': 'model', 'label': str(model_label)}
            all_descriptors = [model_desc] + descriptors
            
            # If still no elements, add single model entry with tooltip
            if len(all_descriptors) == 1:
                try:
                    mdl = getattr(self.viewmodel.state, 'model', None) if self.viewmodel and hasattr(self.viewmodel, 'state') else None
                    if mdl is not None:
                        center_val = None
                        if hasattr(mdl, 'center'):
                            center_val = getattr(mdl, 'center')
                        elif hasattr(mdl, 'Center'):
                            center_val = getattr(mdl, 'Center')
                        if center_val is not None:
                            model_desc['tooltip'] = f"center={center_val}"
                except Exception:
                    pass
            
            # Delegate to elements dock
            self.elements_dock.refresh_element_list(all_descriptors)
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

    def _log_exception(self, context: str, exc: Exception):
        """Emit a diagnostic message for an exception, preferring the ViewModel log.

        context: short description of where the exception occurred.
        exc: the caught exception instance.
        """
        try:
            tb = traceback.format_exc()
            msg = f"{context}: {exc}\n{tb}"
            if getattr(self, 'viewmodel', None) is not None and hasattr(self.viewmodel, 'log_message'):
                try:
                    self.viewmodel.log_message.emit(msg)
                except Exception:
                    print(msg)
            else:
                print(msg)
        except Exception:
            # best-effort: avoid raising while logging
            try:
                print(f"{context}: {exc}")
            except Exception:
                pass

    def _active_model_is_composite(self) -> bool:
        try:
            if not self.viewmodel:
                return False
            spec = getattr(self.viewmodel.state, "model_spec", None)
            return isinstance(spec, CompositeModelSpec)
        except Exception:
            return False

    def _composite_has_components(self) -> bool:
        if not self._active_model_is_composite():
            return False
        try:
            if self.viewmodel and hasattr(self.viewmodel, "get_component_descriptors"):
                descriptors = self.viewmodel.get_component_descriptors() or []
                return bool(descriptors)
        except Exception:
            return False
        return False

    def _ensure_scene_click_handler(self):
        if getattr(self, "_scene_click_connected", False):
            return
        try:
            scene = self.plot_widget.scene()
            scene.sigMouseClicked.connect(self._on_scene_clicked)
            self._scene_click_connected = True
        except Exception:
            pass

    def _auto_select_curve_for_peaks(self):
        """Return a sensible default curve id for peak interactions when none selected."""
        selectable_ids = []
        try:
            for cid, curve in self.curves.items():
                if getattr(curve, "_selectable", True):
                    selectable_ids.append(cid)
        except Exception:
            selectable_ids = []

        if not selectable_ids:
            return None

        if self._composite_has_components():
            component_ids = [cid for cid in selectable_ids if str(cid).startswith("component:")]
            if len(component_ids) == 1:
                return component_ids[0]
            return None

        # Non-composite: fall back to the fit curve when available
        if "fit" in selectable_ids:
            return "fit"
        return selectable_ids[0]

    def _ensure_curve_selection_for_peaks(self) -> bool:
        """Ensure a curve is selected before peak interactions; auto-select when safe."""
        try:
            current = getattr(self.input_handler, "selected_curve_id", None)
        except Exception:
            current = None

        if current:
            return True

        auto_id = self._auto_select_curve_for_peaks()
        if not auto_id:
            return False

        try:
            if hasattr(self.input_handler, "set_selected_curve"):
                self.input_handler.set_selected_curve(auto_id)
        except Exception:
            pass

        try:
            if self.viewmodel and hasattr(self.viewmodel, "set_selected_curve"):
                self.viewmodel.set_selected_curve(auto_id)
        except Exception:
            pass

        return True

    def _apply_curve_pen(self, curve: pg.PlotDataItem, selected: bool = False):
        if curve is None:
            return
        color = getattr(curve, "_base_color", FIT_COLOR) or FIT_COLOR
        is_component = bool(getattr(curve, "_is_component", False))
        if selected:
            width = 4 if is_component else 3
            style = Qt.SolidLine
        else:
            width = 2
            style = Qt.DashLine if is_component else Qt.SolidLine
        try:
            pen = pg.mkPen(color, width=width, style=style)
            curve.setPen(pen)
        except Exception:
            pass

    def _upsert_curve(self, curve_id: str, x_arr: np.ndarray, y_arr: np.ndarray,
                      color: str, selectable: bool, is_component: bool, label: str = None):
        if curve_id is None:
            return
        try:
            x_use = np.asarray(x_arr, dtype=float)
            y_use = np.asarray(y_arr, dtype=float)
        except Exception:
            return

        curve = self.curves.get(curve_id)
        if curve is None:
            try:
                curve = self.plot_widget.plot(
                    x_use,
                    y_use,
                    pen=pg.mkPen(color or FIT_COLOR, width=2),
                )
            except Exception:
                return
            curve.curve_id = curve_id
            self.curves[curve_id] = curve
            self._ensure_scene_click_handler()
        else:
            try:
                curve.setData(x_use, y_use)
            except Exception:
                pass

        setattr(curve, "_base_color", color or FIT_COLOR)
        setattr(curve, "_selectable", bool(selectable))
        setattr(curve, "_is_component", bool(is_component))
        setattr(curve, "_legend_label", label or curve_id)

        selected = (self.selected_curve_id == curve_id)
        self._apply_curve_pen(curve, selected=selected)

    def _update_legend(self):
        legend = getattr(self, "legend", None)
        if legend is None:
            return

        desired_entries = []
        if getattr(self, "scatter", None) is not None:
            desired_entries.append(("Data", self.scatter))
        if self._excluded_has_points and getattr(self, "excluded_scatter", None) is not None:
            desired_entries.append(("Excluded Data", self.excluded_scatter))

        fit_curve = self.curves.get("fit") if hasattr(self, "curves") else None
        if fit_curve is not None:
            label = getattr(fit_curve, "_legend_label", self._fit_label)
            desired_entries.append((label or self._fit_label, fit_curve))

        for label, cid in getattr(self, "_component_legend_order", []):
            curve = self.curves.get(cid)
            if curve is None:
                continue
            entry_label = getattr(curve, "_legend_label", label)
            desired_entries.append((entry_label or label, curve))

        # Ensure legend labels are unique by appending counters when necessary.
        label_counts = {}
        unique_entries = []
        for label, item in desired_entries:
            if item is None:
                continue
            text = label or ""
            count = label_counts.get(text, 0)
            label_counts[text] = count + 1
            if count == 0:
                unique_entries.append((text, item))
            else:
                unique_entries.append((f"{text} ({count + 1})", item))

        desired_map = {label: item for label, item in unique_entries}

        # Remove legend entries that are no longer desired or whose item changed.
        for label in list(self._legend_entries.keys()):
            item = self._legend_entries.get(label)
            if label not in desired_map or desired_map[label] is not item:
                try:
                    legend.removeItem(label)
                except Exception:
                    pass
                self._legend_entries.pop(label, None)

        # Add missing entries in order.
        for label, item in unique_entries:
            if self._legend_entries.get(label) is item:
                continue
            try:
                legend.addItem(item, label)
                self._legend_entries[label] = item
            except Exception:
                pass

    def update_plot_data(self, x, y_data, y_fit=None, y_err=None):
        if x is None or y_data is None:
            return

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)
        self._component_legend_order = []
        self._excluded_has_points = False

        excl_mask = None
        try:
            if self.viewmodel is not None and hasattr(self.viewmodel, "state") and hasattr(self.viewmodel.state, "excluded"):
                excl_mask = np.asarray(self.viewmodel.state.excluded, dtype=bool)
                if len(excl_mask) != len(x_arr):
                    try:
                        self.append_log(f"Exclusion mask length {len(excl_mask)} != x length {len(x_arr)} — ignoring exclusions")
                    except Exception:
                        pass
                    excl_mask = None
        except Exception:
            excl_mask = None

        y_err_arr = None
        if y_err is not None:
            try:
                y_err_arr = np.asarray(y_err, dtype=float)
            except Exception:
                y_err_arr = None

        if excl_mask is None:
            self.scatter.setData(x=x_arr, y=y_arr)
            if self.excluded_scatter is not None:
                try:
                    self.excluded_scatter.setData(x=np.array([], dtype=float), y=np.array([], dtype=float))
                except Exception:
                    pass
            self._excluded_has_points = False
            if y_err_arr is not None and len(y_err_arr) == len(y_arr):
                top = np.abs(y_err_arr)
                bottom = top
                self.err_item.setData(x=x_arr, y=y_arr, top=top, bottom=bottom)
            else:
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
                self._excluded_has_points = bool(len(x_ex))

            if y_err_arr is not None:
                try:
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

        components_meta = []
        fit_display_x = x_arr
        fit_display_y = None
        fit_stats_y = None

        if isinstance(y_fit, dict):
            try:
                components_meta = list(y_fit.get("components") or [])
            except Exception:
                components_meta = []
            total_payload = y_fit.get("total")
            if isinstance(total_payload, dict):
                raw_x = total_payload.get("x")
                raw_y = total_payload.get("y")
                raw_stats = total_payload.get("data_y")
                try:
                    if raw_x is not None:
                        fit_display_x = np.asarray(raw_x, dtype=float)
                except Exception:
                    fit_display_x = x_arr
                try:
                    if raw_y is not None:
                        fit_display_y = np.asarray(raw_y, dtype=float)
                except Exception:
                    fit_display_y = None
                try:
                    if raw_stats is not None:
                        fit_stats_y = np.asarray(raw_stats, dtype=float)
                except Exception:
                    fit_stats_y = None
                if fit_stats_y is None and fit_display_y is not None and len(fit_display_y) == len(x_arr):
                    fit_stats_y = fit_display_y
            elif total_payload is not None:
                try:
                    fit_display_y = np.asarray(total_payload, dtype=float)
                except Exception:
                    fit_display_y = None
                if fit_display_y is not None and len(fit_display_y) == len(x_arr):
                    fit_stats_y = fit_display_y
        elif y_fit is not None:
            try:
                fit_display_y = np.asarray(y_fit, dtype=float)
            except Exception:
                fit_display_y = None
            if fit_display_y is not None and len(fit_display_y) == len(x_arr):
                fit_stats_y = fit_display_y

        try:
            if excl_mask is None:
                xin = x_arr
                yin = y_arr
                if y_err_arr is not None and len(y_err_arr) == len(y_arr):
                    errin = y_err_arr
                else:
                    errin = None
                yfit_for_chi = fit_stats_y if fit_stats_y is not None else None
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
                if fit_stats_y is None:
                    yfit_for_chi = None
                else:
                    if len(fit_stats_y) == len(x_arr):
                        yfit_for_chi = fit_stats_y[~excl_mask]
                    elif len(fit_stats_y) == len(xin):
                        yfit_for_chi = fit_stats_y
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
                self.parameters_dock.update_chi_label(label_text)
            except Exception:
                pass
        except Exception:
            try:
                self.parameters_dock.update_chi_label("reduced χ²=N/A; Cash=N/A")
            except Exception:
                pass

        target_ids = set()
        is_composite_payload = bool(components_meta)
        for comp in components_meta:
            if not isinstance(comp, dict):
                continue
            prefix = str(comp.get("prefix") or "")
            if not prefix:
                continue
            comp_y = comp.get("y")
            if comp_y is None:
                continue
            try:
                comp_arr = np.asarray(comp_y, dtype=float)
            except Exception:
                continue
            comp_x = comp.get("x")
            if comp_x is not None:
                try:
                    comp_x_arr = np.asarray(comp_x, dtype=float)
                except Exception:
                    comp_x_arr = x_arr
            else:
                comp_x_arr = x_arr
            if comp_arr.size == 0 or comp_x_arr.size != comp_arr.size:
                continue
            color = comp.get("color") or FIT_COLOR
            curve_id = f"component:{prefix}"
            label = comp.get("label") or prefix.rstrip("_") or prefix
            self._upsert_curve(curve_id, comp_x_arr, comp_arr, color=color, selectable=True, is_component=True, label=label)
            target_ids.add(curve_id)
            self._component_legend_order.append((label, curve_id))

        if fit_display_y is not None and fit_display_x is not None and len(fit_display_y) == len(fit_display_x) and len(fit_display_y) > 0:
            selectable_total = not is_composite_payload
            self._upsert_curve(
                "fit",
                fit_display_x,
                fit_display_y,
                color=FIT_COLOR,
                selectable=selectable_total,
                is_component=False,
                label=self._fit_label,
            )
            target_ids.add("fit")
            if not selectable_total and self.selected_curve_id == "fit":
                self.selected_curve_id = None
                try:
                    if self.viewmodel:
                        self.viewmodel.clear_selected_curve()
                except Exception:
                    pass

        for cid in list(self.curves.keys()):
            if cid not in target_ids:
                curve = self.curves.pop(cid)
                try:
                    self.plot_widget.removeItem(curve)
                except Exception:
                    pass
                if self.selected_curve_id == cid:
                    self.selected_curve_id = None
                    try:
                        if self.viewmodel:
                            self.viewmodel.clear_selected_curve()
                    except Exception:
                        pass

        self._component_legend_order = [(label, cid) for label, cid in self._component_legend_order if cid in target_ids]
        self._update_legend()

        if not target_ids:
            self.selected_curve_id = None


    def _on_apply_clicked(self):
        if not self.viewmodel:
            return

        # Get parameter values from parameters dock
        params = self.parameters_dock.get_parameter_values()

        # Send params dict to ViewModel; ViewModel handles validation/usage
        try:
            # Prefer routing through the centralized dispatcher when available
            if hasattr(self.viewmodel, "handle_action"):
                try:
                    self.viewmodel.handle_action("apply_parameters", params=params)
                    self.append_log("Parameters applied.")
                except Exception:
                    # fallback to direct method when dispatcher fails
                    try:
                        self.viewmodel.apply_parameters(params)
                        self.append_log("Parameters applied.")
                    except Exception as e:
                        self.append_log(f"Failed to apply parameters: {e}")
            else:
                # direct call for older ViewModel implementations
                try:
                    self.viewmodel.apply_parameters(params)
                    self.append_log("Parameters applied.")
                except Exception as e:
                    self.append_log(f"Failed to apply parameters: {e}")
        except Exception as e:
            try:
                self.append_log(f"Failed to apply parameters: {e}")
            except Exception:
                pass

    def _refresh_parameters(self):
        """Ask the ViewModel for parameter specs and rebuild the form.

        If the user is currently interacting with a parameter widget (focus
        is inside one of our param widgets), defer the refresh to avoid
        rebuilding widgets under active interaction which steals focus and
        interrupts continuous adjustments (holding arrow keys or mouse).
        """
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

            # Delegate to parameters dock
            self.parameters_dock.populate_parameters(specs)
            
            # Sync InputHandler control map if needed
            try:
                if hasattr(self, "input_handler") and self.input_handler is not None:
                    control_map = self.parameters_dock.get_control_map()
                    self.input_handler.set_control_map(control_map)
            except Exception as e:
                print(f"[MainWindow] Failed to set control map: {e}")

            self.append_log("Parameter panel refreshed.")
        except Exception as e:
            self.append_log(f"Failed to refresh parameters: {e}")

    # --------------------------
    # Config dialog (view-only)
    # --------------------------
    class _AddElementDialog(QDialog):
        def __init__(self, parent, component_names):
            super().__init__(parent)
            self.setWindowTitle("Add Element")
            self._component_names = list(component_names or [])

            layout = QVBoxLayout(self)

            selector_form = QFormLayout()
            self.component_box = QComboBox()
            if self._component_names:
                self.component_box.addItems(self._component_names)
            selector_form.addRow("Component", self.component_box)
            layout.addLayout(selector_form)

            self._param_scroll = QScrollArea()
            self._param_scroll.setWidgetResizable(True)
            self._param_widget = QWidget()
            self._param_form = QFormLayout(self._param_widget)
            self._param_scroll.setWidget(self._param_widget)
            layout.addWidget(self._param_scroll)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

            self._param_inputs: dict = {}
            self._param_meta: dict = {}
            self.component_box.currentTextChanged.connect(self._on_component_changed)

            if self._component_names:
                self._on_component_changed(self._component_names[0])
            else:
                try:
                    buttons.button(QDialogButtonBox.Ok).setEnabled(False)
                except Exception:
                    # If disabling the OK button fails, ignore the error.
                    # This is non-critical: the dialog will still function, but may allow OK with no components.
                    pass

            self._selected_component = None
            self._initial_params = {}

        def _clear_params(self):
            while self._param_form.count():
                item = self._param_form.takeAt(0)
                if item is None:
                    continue
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _on_component_changed(self, name: str):
            self._clear_params()
            self._param_inputs = {}
            self._param_meta = {}
            params = {}
            try:
                from models import get_model_spec
                spec = get_model_spec(name)
                params = spec.get_parameters()
            except Exception:
                params = {}
            for param_name, meta in params.items():
                value = meta.get("value", "")
                editor = QLineEdit(str(value) if value is not None else "")
                self._param_inputs[param_name] = editor
                self._param_meta[param_name] = meta
                self._param_form.addRow(param_name, editor)

        def _coerce_value(self, param_name: str, text: str):
            meta = self._param_meta.get(param_name, {})
            ptype = str(meta.get("type", "")).lower()
            if ptype in ("float", "double"):
                try:
                    return float(text)
                except Exception:
                    return meta.get("value")
            if ptype == "int":
                try:
                    return int(float(text))
                except Exception:
                    return meta.get("value")
            if ptype == "bool":
                return text.strip().lower() in ("1", "true", "yes", "on")
            return text

        def get_result(self):
            component = self.component_box.currentText().strip()
            values = {}
            for name, widget in self._param_inputs.items():
                values[name] = self._coerce_value(name, widget.text())
            return component, values

        def accept(self):
            component, params = self.get_result()
            self._selected_component = component
            self._initial_params = params
            super().accept()

        @property
        def selected_component(self):
            return self._selected_component

        @property
        def initial_params(self):
            return self._initial_params

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

    # --------------------------
    # Resolution dock handlers
    # --------------------------
    def _on_resolution_clicked(self):
        """Open/show the resolution dock window."""
        try:
            if hasattr(self, "resolution_dock") and self.resolution_dock is not None:
                self.resolution_dock.show()
                self.resolution_dock.raise_()
                self.resolution_dock.activateWindow()
                self.append_log("Resolution settings window opened.")
                
                # Refresh parameters if viewmodel has resolution state
                self._refresh_resolution_parameters()
        except Exception as e:
            self.append_log(f"Failed to open resolution window: {e}")

    def _on_resolution_model_changed(self, model_name):
        """Handle resolution model selection change."""
        try:
            if self.viewmodel and hasattr(self.viewmodel, "set_resolution_model"):
                self.viewmodel.set_resolution_model(model_name)
                self.append_log(f"Resolution model changed to: {model_name}")
                self._refresh_resolution_parameters()
                try:
                    self.viewmodel.update_plot()
                except Exception:
                    pass
        except Exception as e:
            self.append_log(f"Failed to change resolution model: {e}")

    def _on_resolution_parameter_changed(self, name, value):
        """Handle resolution parameter change."""
        try:
            if self.viewmodel and hasattr(self.viewmodel, "apply_resolution_parameters"):
                self.viewmodel.apply_resolution_parameters({name: value})
                # Update preview
                self._update_resolution_preview()
                # Update main plot immediately (like other parameters)
                try:
                    self.viewmodel.update_plot()
                except Exception:
                    pass
        except Exception as e:
            self.append_log(f"Failed to update resolution parameter '{name}': {e}")

    def _on_resolution_apply_clicked(self):
        """Handle resolution apply button click."""
        try:
            if not self.viewmodel:
                return
            params = self.resolution_dock.get_parameter_values()
            if hasattr(self.viewmodel, "apply_resolution_parameters"):
                self.viewmodel.apply_resolution_parameters(params)
                self.append_log("Resolution parameters applied.")
                try:
                    self.viewmodel.update_plot()
                except Exception:
                    pass
        except Exception as e:
            self.append_log(f"Failed to apply resolution parameters: {e}")

    def _refresh_resolution_parameters(self):
        """Refresh the resolution dock parameter form."""
        try:
            if not self.viewmodel or not hasattr(self, "resolution_dock"):
                return
            
            if hasattr(self.viewmodel, "get_resolution_parameters"):
                specs = self.viewmodel.get_resolution_parameters()
                self.resolution_dock.populate_parameters(specs or {})
                self._update_resolution_preview()
        except Exception as e:
            self.append_log(f"Failed to refresh resolution parameters: {e}")

    def _update_resolution_preview(self):
        """Update the resolution preview plot."""
        try:
            if not self.viewmodel or not hasattr(self, "resolution_dock"):
                return
            
            if hasattr(self.viewmodel, "get_resolution_preview"):
                x, y = self.viewmodel.get_resolution_preview()
                if x is not None and y is not None:
                    self.resolution_dock.update_preview(x, y)
        except Exception:
            pass

    def _on_curve_clicked(self, event, curve_id):
        """Handle mouse click on a curve."""
        if not self.viewmodel:
            return
        curve = self.curves.get(curve_id)
        if curve is not None and getattr(curve, "_selectable", True) is False:
            return
        # Set this as the selected curve
        self.viewmodel.set_selected_curve(curve_id)

    def _on_curve_selected(self, curve_id):
        """Highlight the selected curve visually."""
        if curve_id == "fit" and self._composite_has_components():
            # keep composite total non-selectable
            try:
                if self.viewmodel:
                    self.viewmodel.clear_selected_curve()
            except Exception as e:
                self.append_log(f"Error clearing selected curve: {e}")
            curve_id = None

        # reset previous curve pen
        if self.selected_curve_id and self.selected_curve_id in self.curves:
            self._apply_curve_pen(self.curves[self.selected_curve_id], selected=False)

        self.selected_curve_id = curve_id if curve_id in self.curves else None

        if self.selected_curve_id and self.selected_curve_id in self.curves:
            curve = self.curves[self.selected_curve_id]
            if getattr(curve, "_selectable", True):
                self._apply_curve_pen(curve, selected=True)
            else:
                self.selected_curve_id = None

        # Sync element list selection when possible
        if hasattr(self, "element_list"):
            try:
                self.element_list.blockSignals(True)
                if self.selected_curve_id and self.selected_curve_id.startswith("component:"):
                    prefix = self.selected_curve_id.split(":", 1)[1]
                    target_row = -1
                    for row in range(self.element_list.count()):
                        item = self.element_list.item(row)
                        data = item.data(Qt.UserRole) if item is not None else None
                        if isinstance(data, dict) and data.get('id') == prefix:
                            target_row = row
                            break
                    if target_row >= 0:
                        self.element_list.setCurrentRow(target_row)
                    else:
                        self.element_list.clearSelection()
                elif self.selected_curve_id == "fit":
                    target_row = -1
                    for row in range(self.element_list.count()):
                        item = self.element_list.item(row)
                        data = item.data(Qt.UserRole) if item is not None else None
                        if isinstance(data, dict) and data.get('id') == 'model':
                            target_row = row
                            break
                    if target_row >= 0:
                        self.element_list.setCurrentRow(target_row)
                    else:
                        self.element_list.clearSelection()
                else:
                    self.element_list.clearSelection()
            finally:
                try:
                    self.element_list.blockSignals(False)
                except Exception as e:
                    # Non-critical: UI may remain in a blocked state, but app continues
                    self.append_log(f"Error unblocking element_list signals: {e}")

        self.append_log(f"Curve selection changed → {self.selected_curve_id or 'none'}")

    def _on_scene_clicked(self, ev):
        """Central handler for scene clicks. Map the click to data coords and
        test proximity to each plotted curve's data. Select the first curve
        whose visible line is within tol pixels of the click.
        """
        try:
            vb = self.plot_widget.getViewBox()
            sp = ev.scenePos()
            pv = vb.mapSceneToView(sp)
            x_click, y_click = float(pv.x()), float(pv.y())
            tol_pixels = float(CURVE_SELECT_TOL_PIXELS)
            t2 = tol_pixels * tol_pixels
            from PySide6.QtCore import QPointF

            # Iterate curves and test a small neighborhood around the nearest
            # x-value to avoid mapping every point for large datasets.
            for cid, curve in list(self.curves.items()):
                if getattr(curve, "_selectable", True) is False:
                    continue
                try:
                    data = curve.getData()
                    if not data:
                        continue
                    cx, cy = data
                    if cx is None or len(cx) == 0:
                        continue
                    cx_arr = np.asarray(cx)
                    cy_arr = np.asarray(cy)
                    # find index nearest to clicked x in data-space
                    idx = int(np.argmin(np.abs(cx_arr - x_click)))
                    lo = max(0, idx - 5)
                    hi = min(len(cx_arr), idx + 6)
                    for xi, yi in zip(cx_arr[lo:hi], cy_arr[lo:hi]):
                        pt = vb.mapViewToScene(QPointF(float(xi), float(yi)))
                        dx = float(pt.x()) - float(sp.x())
                        dy = float(pt.y()) - float(sp.y())
                        if (dx * dx + dy * dy) <= t2:
                            # Found a hit — notify ViewModel (if present) or
                            # fall back to local selection handling.
                            try:
                                if self.viewmodel and hasattr(self.viewmodel, 'set_selected_curve'):
                                    self.viewmodel.set_selected_curve(cid)
                                else:
                                    self._on_curve_selected(cid)
                            except Exception:
                                try:
                                    self._on_curve_selected(cid)
                                except Exception:
                                    pass
                            ev.accept()
                            return
                except Exception:
                    continue
        except Exception:
            pass


