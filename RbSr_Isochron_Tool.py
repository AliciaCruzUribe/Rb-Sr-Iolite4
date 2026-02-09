#/ Type: UI
#/ Name: Rb-Sr Isochron Tool
#/ Authors: Cici Cruz-Uribe
#/ Description: Interactive isochron plotting with York regression for Rb-Sr geochronology
#/ References:
#/ Version: 1.1
#/ Contact:

"""
Flexible isochron plotting tool for Rb-Sr geochronology with:
- Dynamic channel selection for X and Y axes
- Models 1, 2, and 3 with optional fixed intercepts
- Error ellipse visualization with rho correlation
- Inverse isochrons
- Results display (age, initial ratio, MSWD)
"""

import numpy as np
import time
from scipy import stats
from iolite import QtGui, QtCore
from iolite.QtGui import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QCheckBox, QLineEdit, QGroupBox, QListWidget, QListWidgetItem, QSplitter, QDoubleSpinBox, QFileDialog, QMessageBox, QAction, QTableWidget, QTableWidgetItem
from iolite.QtCore import Qt

# Matplotlib imports - proper backend for iolite
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


class RbSrIsochronWidget(QWidget):
    """Main widget for Rb-Sr isochron plotting and regression"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Rb-Sr Isochron Tool')
        self.resize(1200, 800)

        # Store settings
        self.settings = QtCore.QSettings('Iolite', 'RbSr_Isochron_Tool')

        # Initialize data storage
        self.current_x_data = None
        self.current_y_data = None
        self.current_x_err = None
        self.current_y_err = None
        self.current_rho = None
        self.current_groups = None
        self.group_colors = {}

        # Store actual channel names used (may differ from dropdown if using UNC channels)
        self.actual_x_err_channel = None
        self.actual_y_err_channel = None

        # Store regression results
        self.last_slope = None
        self.last_intercept = None
        self.last_slope_err = None
        self.last_intercept_err = None
        self.last_mswd = None
        self.last_prob = None
        self.last_age = None
        self.last_age_err = None

        # Store per-group fixed intercepts
        self.group_intercepts = {}  # {group_name: (use_fixed, intercept, intercept_err)}

        # Flag for individual integrations
        self.use_individual_integrations = False

        print("Setting up UI...")
        self.setup_ui()
        print("UI setup complete")

        print("Loading settings...")
        # Temporarily disabled to prevent crash
        # self.load_settings()
        print("Settings loading skipped")

        print("Populating channel lists...")
        try:
            self.populate_channel_lists()
            print("Channel lists populated")
        except Exception as e:
            print(f"Warning: Could not populate channel lists: {e}")
            # Continue anyway - channels can be populated later

    def setup_ui(self):
        """Create the user interface"""
        main_layout = QVBoxLayout(self)

        # Create splitter for left (controls) and right (plot)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Channel Selection Group
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()

        # X-axis (Rb/Sr) selection
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X-axis (Rb/Sr):"))
        self.x_channel_combo = QComboBox()
        self.x_channel_combo.setMinimumWidth(200)
        self.x_channel_combo.currentTextChanged.connect(self.on_channel_changed)
        x_layout.addWidget(self.x_channel_combo)
        channel_layout.addLayout(x_layout)

        # X uncertainty selection
        x_err_layout = QHBoxLayout()
        x_err_layout.addWidget(QLabel("X uncertainty:"))
        self.x_err_combo = QComboBox()
        self.x_err_combo.setMinimumWidth(200)
        self.x_err_combo.currentTextChanged.connect(self.on_channel_changed)
        x_err_layout.addWidget(self.x_err_combo)
        channel_layout.addLayout(x_err_layout)

        # Y-axis (Sr/Sr) selection
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y-axis (Sr/Sr):"))
        self.y_channel_combo = QComboBox()
        self.y_channel_combo.setMinimumWidth(200)
        self.y_channel_combo.currentTextChanged.connect(self.on_channel_changed)
        y_layout.addWidget(self.y_channel_combo)
        channel_layout.addLayout(y_layout)

        # Y uncertainty selection
        y_err_layout = QHBoxLayout()
        y_err_layout.addWidget(QLabel("Y uncertainty:"))
        self.y_err_combo = QComboBox()
        self.y_err_combo.setMinimumWidth(200)
        self.y_err_combo.currentTextChanged.connect(self.on_channel_changed)
        y_err_layout.addWidget(self.y_err_combo)
        channel_layout.addLayout(y_err_layout)

        # Rho correlation selection
        rho_layout = QHBoxLayout()
        self.use_rho_check = QCheckBox("Use rho correlation:")
        self.use_rho_check.setChecked(False)  # Default: unchecked
        self.use_rho_check.stateChanged.connect(self.on_use_rho_changed)
        # Don't auto-update plot - user must click Update Plot button
        rho_layout.addWidget(self.use_rho_check)
        self.rho_combo = QComboBox()
        self.rho_combo.setMinimumWidth(200)
        self.rho_combo.setEnabled(self.use_rho_check.isChecked())
        self.rho_combo.currentTextChanged.connect(self.on_channel_changed)
        rho_layout.addWidget(self.rho_combo)
        channel_layout.addLayout(rho_layout)

        channel_group.setLayout(channel_layout)
        left_layout.addWidget(channel_group)

        # Regression Settings Group
        regression_group = QGroupBox("Regression Settings")
        regression_layout = QVBoxLayout()

        # Regression Model Selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Regression Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Model 1: York (MSWD scaling)",
            "Model 2: Total Least Squares",
            "Model 3: York + Overdispersion"
        ])
        self.model_combo.setCurrentIndex(0)  # Default to Model 1
        self.model_combo.setToolTip(
            "Model 1: Maximum likelihood with analytical uncertainties (scales by √MSWD if overdispersed)\n"
            "Model 2: Total Least Squares - ignores analytical uncertainties\n"
            "Model 3: York regression with geological overdispersion parameter"
        )
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        regression_layout.addLayout(model_layout)

        # Age Dispersion checkbox (only for Model 3)
        self.age_dispersion_check = QCheckBox("Age Dispersion (wtype='b')")
        self.age_dispersion_check.setChecked(False)  # Default: unchecked (wtype='a')
        self.age_dispersion_check.setToolTip(
            "Unchecked (default): Constant dispersion on intercept (initial ratio scatter)\n"
            "Checked: Proportional dispersion on age (samples closed at different times)"
        )
        self.age_dispersion_check.setVisible(False)  # Hidden until Model 3 is selected
        regression_layout.addWidget(self.age_dispersion_check)

        # Table for per-group fixed intercepts
        table_label = QLabel("Per-group regression settings:")
        regression_layout.addWidget(table_label)

        self.intercept_table = QTableWidget()
        self.intercept_table.setColumnCount(4)
        self.intercept_table.setHorizontalHeaderLabels(['Group', 'Fixed Int.', 'Intercept', '2SE'])
        self.intercept_table.setColumnWidth(0, 100)
        self.intercept_table.setColumnWidth(1, 60)
        self.intercept_table.setColumnWidth(2, 80)
        self.intercept_table.setColumnWidth(3, 80)
        self.intercept_table.setMaximumHeight(200)
        # Connect signal to update Age Dispersion checkbox when table is modified
        self.intercept_table.itemChanged.connect(self.on_intercept_table_changed)
        regression_layout.addWidget(self.intercept_table)

        # Button to populate table with default values
        populate_defaults_btn = QPushButton("Set Default Values (0.7100)")
        populate_defaults_btn.clicked.connect(self.populate_default_intercepts)
        regression_layout.addWidget(populate_defaults_btn)

        # Checkbox for using individual integrations
        self.use_individual_check = QCheckBox("Use Individual Integrations (automatically uses individual uncertainties)")
        self.use_individual_check.setChecked(False)
        self.use_individual_check.setToolTip(
            "Use individual integration data points instead of selection group means.\n"
            "This automatically uses UNC channels for uncertainties (per-integration 2SE\n"
            "based on counting statistics and detector noise)."
        )
        self.use_individual_check.stateChanged.connect(self.on_individual_integrations_changed)
        regression_layout.addWidget(self.use_individual_check)

        # Checkbox for inverse isochron
        self.inverse_isochron_check = QCheckBox("Inverse Isochron")
        self.inverse_isochron_check.setChecked(False)
        self.inverse_isochron_check.setToolTip(
            "Plot inverse isochron: ⁸⁶Sr/⁸⁷Sr vs ⁸⁷Rb/⁸⁷Sr\n"
            "instead of conventional: ⁸⁷Sr/⁸⁶Sr vs ⁸⁷Rb/⁸⁶Sr\n\n"
            "The y-intercept gives (⁸⁶Sr/⁸⁷Sr)ᵢ, which is inverted\n"
            "to obtain the initial ⁸⁷Sr/⁸⁶Sr ratio."
        )
        regression_layout.addWidget(self.inverse_isochron_check)

        regression_group.setLayout(regression_layout)
        left_layout.addWidget(regression_group)

        # Selection Groups
        groups_group = QGroupBox("Selection Groups")
        groups_layout = QVBoxLayout()

        # Multi-panel checkbox
        self.multi_panel_check = QCheckBox("Enable multi-panel (2x2)")
        self.multi_panel_check.setChecked(False)
        self.multi_panel_check.setToolTip("Create a 2x2 grid of plots with independent axis scaling")
        self.multi_panel_check.stateChanged.connect(self.on_multi_panel_changed)
        groups_layout.addWidget(self.multi_panel_check)

        # Groups table with panel assignments
        self.groups_table = QTableWidget()
        self.groups_table.setColumnCount(5)
        self.groups_table.setHorizontalHeaderLabels(['Group', 'P1', 'P2', 'P3', 'P4'])
        self.groups_table.horizontalHeader().setStretchLastSection(False)
        self.groups_table.setColumnWidth(0, 120)
        self.groups_table.setColumnWidth(1, 30)
        self.groups_table.setColumnWidth(2, 30)
        self.groups_table.setColumnWidth(3, 30)
        self.groups_table.setColumnWidth(4, 30)
        groups_layout.addWidget(self.groups_table)

        # Button row: Refresh Groups (left) and Plot/Update Plot (right)
        button_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Groups")
        refresh_btn.clicked.connect(self.populate_groups)
        button_row.addWidget(refresh_btn)

        update_btn = QPushButton("Plot/Update Plot")
        update_btn.clicked.connect(self.update_plot)
        button_row.addWidget(update_btn)

        groups_layout.addLayout(button_row)

        groups_group.setLayout(groups_layout)
        left_layout.addWidget(groups_group)

        # Plot Display Options
        display_group = QGroupBox("Plot Display Options")
        display_layout = QVBoxLayout()

        # First row: Legend, Results box, 95% CI envelope
        display_row1 = QHBoxLayout()
        self.show_legend_check = QCheckBox("Legend")
        self.show_legend_check.setChecked(True)  # Default: show legend
        display_row1.addWidget(self.show_legend_check)

        self.show_results_check = QCheckBox("Results box")
        self.show_results_check.setChecked(True)  # Default: show results
        display_row1.addWidget(self.show_results_check)

        self.show_envelope_check = QCheckBox("95% CI envelope")
        self.show_envelope_check.setChecked(True)  # Default: show envelope (like IsoplotR)
        self.show_envelope_check.setToolTip("Show 95% confidence envelope around regression line (like IsoplotR)")
        display_row1.addWidget(self.show_envelope_check)
        display_row1.addStretch()
        display_layout.addLayout(display_row1)

        # Second row: Color by chemistry checkbox with Channel and Colormap dropdowns to the right
        display_row2 = QHBoxLayout()
        self.use_gradient_check = QCheckBox("Color by chemistry")
        self.use_gradient_check.setChecked(False)
        self.use_gradient_check.setToolTip("Color data points by a chemical parameter instead of by group")
        self.use_gradient_check.stateChanged.connect(self.on_gradient_changed)
        display_row2.addWidget(self.use_gradient_check)

        display_row2.addWidget(QLabel("Channel:"))
        self.gradient_channel_combo = QComboBox()
        self.gradient_channel_combo.setMinimumWidth(100)
        self.gradient_channel_combo.setEnabled(False)  # Disabled until checkbox is checked
        display_row2.addWidget(self.gradient_channel_combo)

        display_row2.addWidget(QLabel("Color:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['viridis', 'plasma', 'magma', 'inferno', 'cividis',
                                   'coolwarm', 'RdYlBu', 'RdBu', 'Spectral',
                                   'Blues', 'Reds', 'Greens', 'Purples', 'Oranges'])
        self.cmap_combo.setCurrentText('viridis')
        self.cmap_combo.setEnabled(False)  # Disabled until checkbox is checked
        display_row2.addWidget(self.cmap_combo)
        display_layout.addLayout(display_row2)

        display_group.setLayout(display_layout)
        left_layout.addWidget(display_group)

        # Export buttons (side by side)
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout()

        save_plot_btn = QPushButton("Save Plot as PDF")
        save_plot_btn.clicked.connect(self.save_plot_pdf)
        export_layout.addWidget(save_plot_btn)

        export_results_btn = QPushButton("Export to Excel")
        export_results_btn.clicked.connect(self.export_results_excel)
        export_layout.addWidget(export_results_btn)

        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)

        left_layout.addStretch()

        # Right panel - Plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        # Enable interactive mode for basic zoom/pan
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Enable hover functionality for data point labels
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

        # Initialize hover annotation (will be created per-axis when plotting)
        self.hover_annotation = None
        self.scatter_data = []  # Store scatter plot data for hover detection

        right_layout.addWidget(self.canvas)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)  # Left panel won't stretch
        splitter.setStretchFactor(1, 1)  # Right panel takes extra space
        left_panel.adjustSize()  # Calculate minimum size needed
        splitter.setSizes([left_panel.sizeHint.width(), 800])  # Use calculated minimum

        main_layout.addWidget(splitter)

    def populate_channel_lists(self):
        """Populate channel combo boxes with available time series channels"""
        if not data:
            print("No data available yet")
            return

        # Get all time series channels
        try:
            all_channels = data.timeSeriesNames()
            print(f"Found {len(all_channels)} total channels")
            print(f"All channels: {all_channels}")
        except Exception as e:
            print(f"Error getting channel names: {e}")
            return

        # Clear existing items
        self.x_channel_combo.clear()
        self.y_channel_combo.clear()
        self.x_err_combo.clear()
        self.y_err_combo.clear()
        self.rho_combo.clear()
        self.gradient_channel_combo.clear()

        # Add empty option for uncertainties
        self.x_err_combo.addItem("")
        self.y_err_combo.addItem("")
        self.rho_combo.addItem("")
        self.gradient_channel_combo.addItem("")  # Empty option for no gradient

        # Filter channels - only show DRS output channels (not raw masses or CPS)
        rb_sr_channels = []
        sr_sr_channels = []
        err_channels = []
        rho_channels = []
        gradient_channels = []  # For chemistry gradient coloring - only CPS channels

        # List of raw channels to exclude (non-CPS versions)
        raw_channels = ['Li7', 'Mg24', 'Si29', 'Ca43', 'Fe57', 'Rb85', 'Sr86', 'Sr105',
                       'Rb87', 'Sr87', 'Sr88', 'Y89', 'Cs133', 'Ba137', 'TotalBeam', 'mask']

        for channel in all_channels:
            channel_lower = channel.lower()

            # CPS channels go to gradient list only
            if channel.endswith('_CPS'):
                gradient_channels.append(channel)
                continue

            # Skip raw mass channels (non-CPS)
            if channel in raw_channels:
                continue

            # Check for rho channels FIRST (before rb/sr checks)
            if 'rho' in channel_lower:
                rho_channels.append(channel)
            # Error channels
            elif '2se' in channel_lower or 'unc' in channel_lower:
                err_channels.append(channel)
            # Rb/Sr ratios - anything with both rb and sr (but not error channels or rho)
            elif 'rb' in channel_lower and 'sr' in channel_lower:
                rb_sr_channels.append(channel)
            # Sr/Sr ratios - anything with sr (but not rb, not error channels, not rho)
            elif 'sr' in channel_lower and 'rb' not in channel_lower:
                sr_sr_channels.append(channel)

        print(f"Found {len(rb_sr_channels)} Rb/Sr channels: {rb_sr_channels}")
        print(f"Found {len(sr_sr_channels)} Sr/Sr channels: {sr_sr_channels}")
        print(f"Found {len(err_channels)} error channels: {err_channels}")
        print(f"Found {len(rho_channels)} rho channels: {rho_channels}")
        print(f"Found {len(gradient_channels)} CPS channels for gradient: {gradient_channels}")

        # Populate X-axis (Rb/Sr) - show all Rb/Sr output ratios
        self.x_channel_combo.addItems(sorted(rb_sr_channels))

        # Populate Y-axis (Sr/Sr) - show all Sr output ratios
        self.y_channel_combo.addItems(sorted(sr_sr_channels))

        # Populate uncertainties - show all error channels
        self.x_err_combo.addItems(sorted(err_channels))
        self.y_err_combo.addItems(sorted(err_channels))

        # Populate rho
        self.rho_combo.addItems(sorted(rho_channels))

        # Populate gradient channels (for chemistry coloring)
        self.gradient_channel_combo.addItems(sorted(gradient_channels))

        # Set defaults
        default_x = 'AgeCorr_Rb87_Sr86_MBC'
        default_y = 'StdCorr_Sr87_Sr86_MBC'
        default_x_err = 'AgeCorr_Rb87_Sr86_2SE'
        default_y_err = 'StdCorr_Sr87_Sr86_2SE'

        x_idx = self.x_channel_combo.findText(default_x)
        if x_idx >= 0:
            self.x_channel_combo.setCurrentIndex(x_idx)
            # Set default uncertainty channel
            x_err_idx = self.x_err_combo.findText(default_x_err)
            if x_err_idx >= 0:
                self.x_err_combo.setCurrentIndex(x_err_idx)

        y_idx = self.y_channel_combo.findText(default_y)
        if y_idx >= 0:
            self.y_channel_combo.setCurrentIndex(y_idx)
            # Set default uncertainty channel
            y_err_idx = self.y_err_combo.findText(default_y_err)
            if y_err_idx >= 0:
                self.y_err_combo.setCurrentIndex(y_err_idx)

        # Try to find Rho_Sr87Sr86_Rb87Sr86 as default
        rho_idx = self.rho_combo.findText('Rho_Sr87Sr86_Rb87Sr86')
        if rho_idx >= 0:
            self.rho_combo.setCurrentIndex(rho_idx)

        # Populate groups
        self.populate_groups()

    def populate_groups(self):
        """Populate the selection groups table with panel checkboxes"""
        if not data:
            return

        self.groups_table.setRowCount(0)

        try:
            # Get both samples and reference materials
            sample_groups = data.selectionGroupNames(data.Sample)
            rm_groups = data.selectionGroupNames(data.ReferenceMaterial)

            # Combine and sort
            all_groups = list(sample_groups) + list(rm_groups)
            all_groups = sorted(set(all_groups))  # Remove duplicates and sort

        except Exception as e:
            print(f"Error getting selection groups: {e}")
            return

        self.groups_table.setRowCount(len(all_groups))

        for i, group in enumerate(all_groups):
            # Group name (read-only)
            group_item = QTableWidgetItem(group)
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)
            self.groups_table.setItem(i, 0, group_item)

            # Panel checkboxes (P1, P2, P3, P4)
            for col in range(1, 5):
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox_item.setCheckState(Qt.Unchecked)
                self.groups_table.setItem(i, col, checkbox_item)

    def on_channel_changed(self):
        """Handle channel selection changes"""
        pass  # save_settings disabled
        # Don't auto-update plot - user must click Update Plot button

    def on_use_rho_changed(self, state):
        """Enable/disable rho dropdown based on checkbox state"""
        enabled = (state == Qt.Checked)
        self.rho_combo.setEnabled(enabled)
        # Don't auto-update plot - user must click Update Plot button

    def on_gradient_changed(self, state):
        """Enable/disable gradient controls based on checkbox state"""
        enabled = (state == Qt.Checked)
        self.gradient_channel_combo.setEnabled(enabled)
        self.cmap_combo.setEnabled(enabled)
        # Don't auto-update plot - user must click Update Plot button

    def on_multi_panel_changed(self, state):
        """Handle multi-panel mode toggle"""
        # Update column visibility or styling if needed
        # For now, just don't auto-update - user must click Update Plot button
        pass

    def populate_intercept_table(self):
        """Populate the intercept table with current groups"""
        selected_groups = self.get_selected_groups()

        self.intercept_table.setRowCount(len(selected_groups))

        for i, group in enumerate(selected_groups):
            # Group name (read-only)
            group_item = QTableWidgetItem(group)
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)
            self.intercept_table.setItem(i, 0, group_item)

            # Get saved values or use defaults
            if group in self.group_intercepts:
                use_fixed, intercept_val, intercept_err = self.group_intercepts[group]
            else:
                use_fixed = False  # Default to free intercept
                # Try to get initial Sr ratio from reference material file
                intercept_val, intercept_err = self.get_reference_material_initial_sr(group)
                if intercept_val is None:
                    # Default fallback value
                    intercept_val = 0.7100
                    intercept_err = 0.0002

            # Fixed intercept checkbox
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.Checked if use_fixed else Qt.Unchecked)
            self.intercept_table.setItem(i, 1, checkbox_item)

            # Intercept value (editable)
            intercept_item = QTableWidgetItem(f"{intercept_val:.6f}")
            self.intercept_table.setItem(i, 2, intercept_item)

            # Intercept error (editable)
            err_item = QTableWidgetItem(f"{intercept_err:.6f}")
            self.intercept_table.setItem(i, 3, err_item)

    def get_reference_material_initial_sr(self, group_name):
        """
        Try to get the initial ⁸⁷Sr/⁸⁶Sr ratio from the reference material file.
        Returns (value, uncertainty) or (None, None) if not available.
        """
        try:
            # Try to access the reference material database through iolite
            rm_data = data.referenceMaterialData(group_name)
            if rm_data is not None and "87Sr/86Sr initial" in rm_data:
                val = float(rm_data["87Sr/86Sr initial"].value())
                err_val = float(rm_data["87Sr/86Sr initial"].uncertainty())
                print(f"  Found initial Sr ratio for {group_name} in reference material: {val} ± {err_val}")
                return val, err_val
        except Exception as e:
            # Reference material data not available, use default
            pass
        return None, None

    def populate_default_intercepts(self):
        """Fill table with default intercept values, but preserve reference material values if available"""
        for row in range(self.intercept_table.rowCount):
            # Check the "Fixed Int." checkbox
            checkbox_item = self.intercept_table.item(row, 1)
            if checkbox_item:
                checkbox_item.setCheckState(Qt.Checked)

            # Get the group name for this row
            group_item = self.intercept_table.item(row, 0)
            if group_item:
                group_name = group_item.text()
                # Try to get value from reference material first
                rm_val, rm_err = self.get_reference_material_initial_sr(group_name)
                if rm_val is not None:
                    # Use reference material value
                    self.intercept_table.setItem(row, 2, QTableWidgetItem(f"{rm_val:.6f}"))
                    self.intercept_table.setItem(row, 3, QTableWidgetItem(f"{rm_err:.6f}"))
                else:
                    # Use default values
                    self.intercept_table.setItem(row, 2, QTableWidgetItem("0.710000"))
                    self.intercept_table.setItem(row, 3, QTableWidgetItem("0.000200"))

    def get_group_intercepts(self):
        """Read intercept values from table and store them"""
        self.group_intercepts.clear()

        for row in range(self.intercept_table.rowCount):
            group_item = self.intercept_table.item(row, 0)
            checkbox_item = self.intercept_table.item(row, 1)
            intercept_item = self.intercept_table.item(row, 2)
            err_item = self.intercept_table.item(row, 3)

            if group_item and checkbox_item and intercept_item and err_item:
                try:
                    group_name = group_item.text()
                    use_fixed = (checkbox_item.checkState() == Qt.Checked)
                    intercept = float(intercept_item.text())
                    intercept_err = float(err_item.text())
                    self.group_intercepts[group_name] = (use_fixed, intercept, intercept_err)
                except ValueError:
                    print(f"Warning: Invalid intercept value for {group_name}, using defaults")
                    self.group_intercepts[group_name] = (False, 0.7100, 0.0002)

        # Update Age Dispersion checkbox availability based on fixed intercepts
        self.update_age_dispersion_availability()

    def update_age_dispersion_availability(self):
        """Enable/disable Age Dispersion checkbox based on whether any group has fixed intercept"""
        # Only relevant for Model 3
        if self.model_combo.currentIndex != 2:
            return

        # Check if any group has a fixed intercept
        has_fixed_intercept = any(use_fixed for use_fixed, _, _ in self.group_intercepts.values())

        if has_fixed_intercept:
            # Disable and uncheck the Age Dispersion checkbox
            self.age_dispersion_check.setEnabled(False)
            self.age_dispersion_check.setChecked(False)
            self.age_dispersion_check.setToolTip(
                "Age Dispersion (wtype='b') is not available when using fixed intercepts.\n"
                "Fixed intercept mode only supports constant dispersion (wtype='a')."
            )
        else:
            # Enable the Age Dispersion checkbox
            self.age_dispersion_check.setEnabled(True)
            self.age_dispersion_check.setToolTip(
                "Unchecked (default): Constant dispersion on intercept (initial ratio scatter)\n"
                "Checked: Proportional dispersion on age (samples closed at different times)"
            )

    def on_intercept_table_changed(self, item):
        """Handle changes to the intercept table"""
        # Check if a Fixed Int. checkbox was changed (column 1)
        if item and item.column() == 1:
            # Read current table state and update Age Dispersion availability
            self.get_group_intercepts()


    def get_selected_groups(self):
        """Get list of all groups that have any panel checkbox checked"""
        selected = []
        print(f"Checking {self.groups_table.rowCount} groups...")
        for i in range(self.groups_table.rowCount):
            group_item = self.groups_table.item(i, 0)
            if group_item is None:
                continue
            group_text = group_item.text()

            # Check if any panel checkbox is checked for this group
            any_checked = False
            for col in range(1, 5):
                checkbox_item = self.groups_table.item(i, col)
                if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                    any_checked = True
                    break

            if any_checked:
                selected.append(group_text)
                print(f"  Group '{group_text}': selected")
        print(f"Selected groups: {selected}")
        return selected

    def get_groups_for_panel(self, panel_num):
        """Get list of groups assigned to a specific panel (1-4)"""
        groups = []
        col = panel_num  # Column 1 = P1, Column 2 = P2, etc.
        for i in range(self.groups_table.rowCount):
            group_item = self.groups_table.item(i, 0)
            checkbox_item = self.groups_table.item(i, col)
            if group_item and checkbox_item and checkbox_item.checkState() == Qt.Checked:
                groups.append(group_item.text())
        return groups

    def on_individual_integrations_changed(self, state):
        """Enable/disable uncertainty dropdowns based on individual integrations checkbox"""
        # When individual integrations is checked, disable (grey out) uncertainty dropdowns
        # since UNC channels are automatically selected
        is_checked = (state == Qt.Checked)

        # Disable/enable the uncertainty dropdown menus
        self.x_err_combo.setEnabled(not is_checked)
        self.y_err_combo.setEnabled(not is_checked)

        # Update tooltip to explain why they're disabled
        if is_checked:
            tooltip_text = "Disabled - UNC channels automatically selected for individual integrations"
            self.x_err_combo.setToolTip(tooltip_text)
            self.y_err_combo.setToolTip(tooltip_text)
        else:
            self.x_err_combo.setToolTip("")
            self.y_err_combo.setToolTip("")

    def on_model_changed(self, index):
        """Handle regression model selection changes"""
        # Model 1 = index 0: York with MSWD scaling (current implementation)
        # Model 2 = index 1: Total Least Squares (to be implemented)
        # Model 3 = index 2: York + Overdispersion (to be implemented)

        # Show/hide Age Dispersion checkbox based on model selection
        if index == 2:  # Model 3
            self.age_dispersion_check.setVisible(True)
            # Check if Age Dispersion should be disabled due to fixed intercepts
            self.update_age_dispersion_availability()
            print("Model 3 (York + Overdispersion) selected")
        else:
            self.age_dispersion_check.setVisible(False)
            if index == 1:
                print("Model 2 (Total Least Squares) selected")
            else:
                print("Model 1 (York with MSWD scaling) selected")

    def load_data_for_groups(self):
        """Load data from selected channels for checked groups"""
        if not data:
            print("ERROR: No data object available")
            return None

        x_channel_name = self.x_channel_combo.currentText
        y_channel_name = self.y_channel_combo.currentText
        x_err_name = self.x_err_combo.currentText
        y_err_name = self.y_err_combo.currentText
        rho_name = self.rho_combo.currentText

        print(f"Loading data for channels:")
        print(f"  X: {x_channel_name}")
        print(f"  Y: {y_channel_name}")
        print(f"  X err: {x_err_name}")
        print(f"  Y err: {y_err_name}")
        print(f"  Rho: {rho_name}")

        if not x_channel_name or not y_channel_name:
            print("ERROR: X or Y channel name is empty")
            return None

        selected_groups = self.get_selected_groups()
        if not selected_groups:
            print("ERROR: No groups selected")
            return None

        print(f"Selected groups: {selected_groups}")

        # Get channel data
        try:
            x_channel = data.timeSeries(x_channel_name)
            y_channel = data.timeSeries(y_channel_name)
            print(f"Got X channel: {x_channel is not None}")
            print(f"Got Y channel: {y_channel is not None}")
        except Exception as e:
            print(f"ERROR getting channels: {e}")
            return None

        if x_channel is None or y_channel is None:
            print("ERROR: One or both channels are None")
            return None

        # Check if using individual integrations
        use_individual = self.use_individual_check.isChecked()

        # Determine which uncertainty channels to use
        if use_individual:
            # For individual integrations, look for UNC channels
            # Handle different possible naming patterns
            if "_2SE" in x_err_name:
                x_err_name_indiv = x_err_name.replace("_2SE", "_UNC")
            elif "2SE" in x_err_name:
                x_err_name_indiv = x_err_name.replace("2SE", "UNC")
            else:
                # If no 2SE in name, try appending _UNC
                x_err_name_indiv = x_err_name.rsplit('_', 1)[0] + "_UNC" if x_err_name else ""

            if "_2SE" in y_err_name:
                y_err_name_indiv = y_err_name.replace("_2SE", "_UNC")
            elif "2SE" in y_err_name:
                y_err_name_indiv = y_err_name.replace("2SE", "UNC")
            else:
                # If no 2SE in name, try appending _UNC
                y_err_name_indiv = y_err_name.rsplit('_', 1)[0] + "_UNC" if y_err_name else ""

            print(f"\n=== Individual Integrations Mode ===")
            print(f"Looking for UNC channels...")
            print(f"  X 2SE channel: {x_err_name}")
            print(f"  X UNC channel: {x_err_name_indiv}")
            print(f"  Y 2SE channel: {y_err_name}")
            print(f"  Y UNC channel: {y_err_name_indiv}")

            # Try to load UNC channels
            x_err_channel = None
            y_err_channel = None
            x_unc_found = False
            y_unc_found = False

            try:
                if x_err_name_indiv:
                    x_err_channel = data.timeSeries(x_err_name_indiv)
                    x_unc_found = True
                    self.actual_x_err_channel = x_err_name_indiv  # Store actual channel used
                    print(f"  SUCCESS: Successfully loaded X UNC channel: {x_err_name_indiv}")
            except Exception as e:
                print(f"  FAILED: Could not find X UNC channel '{x_err_name_indiv}'")
                print(f"    Error: {e}")
                print(f"  -> Falling back to X 2SE channel: {x_err_name}")
                try:
                    x_err_channel = data.timeSeries(x_err_name) if x_err_name else None
                    self.actual_x_err_channel = x_err_name  # Store actual channel used
                except:
                    pass

            try:
                if y_err_name_indiv:
                    y_err_channel = data.timeSeries(y_err_name_indiv)
                    y_unc_found = True
                    self.actual_y_err_channel = y_err_name_indiv  # Store actual channel used
                    print(f"  SUCCESS: Successfully loaded Y UNC channel: {y_err_name_indiv}")
            except Exception as e:
                print(f"  FAILED: Could not find Y UNC channel '{y_err_name_indiv}'")
                print(f"    Error: {e}")
                print(f"  -> Falling back to Y 2SE channel: {y_err_name}")
                try:
                    y_err_channel = data.timeSeries(y_err_name) if y_err_name else None
                    self.actual_y_err_channel = y_err_name  # Store actual channel used
                except:
                    pass

            if not x_unc_found or not y_unc_found:
                print("\n  WARNING: Not all UNC channels found!")
                print("  Make sure your DRS creates the UNC channels for individual integrations.")
        else:
            # Use standard uncertainty channels for selection means
            x_err_channel = data.timeSeries(x_err_name) if x_err_name else None
            y_err_channel = data.timeSeries(y_err_name) if y_err_name else None
            self.actual_x_err_channel = x_err_name  # Store actual channel used
            self.actual_y_err_channel = y_err_name  # Store actual channel used

        rho_channel = data.timeSeries(rho_name) if rho_name and self.use_rho_check.isChecked() else None

        # Get gradient channel if gradient coloring is enabled
        gradient_channel = None
        gradient_name = self.gradient_channel_combo.currentText if self.use_gradient_check.isChecked() else ""
        if gradient_name:
            try:
                gradient_channel = data.timeSeries(gradient_name)
                print(f"  Gradient channel loaded: {gradient_name}")
            except Exception as e:
                print(f"  WARNING: Could not load gradient channel '{gradient_name}': {e}")

        print(f"\nFinal channel status:")
        print(f"  X err channel loaded: {x_err_channel is not None}")
        print(f"  Y err channel loaded: {y_err_channel is not None}")
        print(f"  Rho channel loaded: {rho_channel is not None}")
        print(f"  Gradient channel loaded: {gradient_channel is not None}")

        # Collect data by group
        all_x = []
        all_y = []
        all_x_err = []
        all_y_err = []
        all_rho = []
        all_groups = []
        all_gradient = []  # For gradient coloring
        all_labels = []  # For hover labels (selection names)

        for group_name in selected_groups:
            print(f"Processing group: {group_name}")
            group = data.selectionGroup(group_name)
            if group is None:
                print(f"  WARNING: Could not get group {group_name}")
                continue

            selections = group.selections()
            print(f"  Found {len(selections)} selections")

            for sel in selections:
                if use_individual:
                    # Get individual integration data points (all points in selection)
                    try:
                        x_data = x_channel.dataForSelection(sel)
                        y_data = y_channel.dataForSelection(sel)
                        x_err_data = x_err_channel.dataForSelection(sel) if x_err_channel else np.zeros_like(x_data)
                        y_err_data = y_err_channel.dataForSelection(sel) if y_err_channel else np.zeros_like(y_data)

                        # Filter out NaN and invalid values
                        valid_mask = (np.isfinite(x_data) & np.isfinite(y_data) &
                                    np.isfinite(x_err_data) & np.isfinite(y_err_data) &
                                    (x_data > 0) & (y_data > 0))

                        x_valid = x_data[valid_mask]
                        y_valid = y_data[valid_mask]
                        x_err_valid = x_err_data[valid_mask]
                        y_err_valid = y_err_data[valid_mask]

                        if len(x_valid) == 0:
                            print(f"  Skipping {sel.name} - No valid data points")
                            continue

                        # Add all valid points from this selection
                        all_x.extend(x_valid)
                        all_y.extend(y_valid)
                        all_x_err.extend(x_err_valid)
                        all_y_err.extend(y_err_valid)

                        # Show uncertainty statistics for first selection to verify correct channel
                        if len(all_x) == len(x_valid):  # First selection added
                            print(f"  {sel.name}: Added {len(x_valid)} individual data points")
                            print(f"    X uncertainty range: {x_err_valid.min():.6f} to {x_err_valid.max():.6f}")
                            print(f"    Y uncertainty range: {y_err_valid.min():.6f} to {y_err_valid.max():.6f}")
                        else:
                            print(f"  {sel.name}: Added {len(x_valid)} individual data points")

                        # Handle rho for individual integrations
                        if rho_channel is not None:
                            rho_data = rho_channel.dataForSelection(sel)
                            rho_valid = rho_data[valid_mask]
                            all_rho.extend(rho_valid)
                        else:
                            all_rho.extend([0.0] * len(x_valid))

                        # Handle gradient channel for individual integrations
                        if gradient_channel is not None:
                            gradient_data = gradient_channel.dataForSelection(sel)
                            gradient_valid = gradient_data[valid_mask]
                            all_gradient.extend(gradient_valid)
                        else:
                            all_gradient.extend([np.nan] * len(x_valid))

                        # Add group name for each point
                        all_groups.extend([group_name] * len(x_valid))

                        # Add selection name as label for each point
                        all_labels.extend([sel.name] * len(x_valid))

                    except Exception as e:
                        print(f"  Error getting individual integration data for {sel.name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                else:
                    # Use selection mean (original behavior)
                    try:
                        x_result = data.result(sel, x_channel)
                        y_result = data.result(sel, y_channel)

                        x_mean = x_result.value()
                        y_mean = y_result.value()

                        if x_mean is None or y_mean is None or np.isnan(x_mean) or np.isnan(y_mean):
                            print(f"  Skipping selection - NaN or None values")
                            continue

                        print(f"  Got data point: x={x_mean:.4f}, y={y_mean:.6f}")

                        all_x.append(x_mean)
                        all_y.append(y_mean)
                        all_groups.append(group_name)
                        all_labels.append(sel.name)  # Add selection name as label

                        # Get uncertainties
                        if x_err_channel:
                            try:
                                x_err_result = data.result(sel, x_err_channel)
                                x_err_mean = x_err_result.value()
                                all_x_err.append(x_err_mean if x_err_mean is not None else 0.0)
                            except:
                                all_x_err.append(0.0)
                        else:
                            all_x_err.append(0.0)

                        if y_err_channel:
                            try:
                                y_err_result = data.result(sel, y_err_channel)
                                y_err_mean = y_err_result.value()
                                all_y_err.append(y_err_mean if y_err_mean is not None else 0.0)
                            except:
                                all_y_err.append(0.0)
                        else:
                            all_y_err.append(0.0)

                        # Get rho - either from channel or calculate it
                        if rho_channel and self.use_rho_check.isChecked():
                            try:
                                rho_result = data.result(sel, rho_channel)
                                rho_mean = rho_result.value()
                                all_rho.append(rho_mean if rho_mean is not None else 0.0)
                            except:
                                all_rho.append(0.0)
                        elif self.use_rho_check.isChecked():
                            # Default to 0 (uncorrelated errors)
                            all_rho.append(0.0)
                        else:
                            all_rho.append(0.0)

                        # Get gradient value for coloring
                        if gradient_channel is not None:
                            try:
                                gradient_result = data.result(sel, gradient_channel)
                                gradient_mean = gradient_result.value()
                                all_gradient.append(gradient_mean if gradient_mean is not None else np.nan)
                            except:
                                all_gradient.append(np.nan)
                        else:
                            all_gradient.append(np.nan)

                    except Exception as e:
                        print(f"  Error getting data for selection: {e}")
                        continue

        print(f"\n=== Data Collection Summary ===")
        print(f"Total data points collected: {len(all_x)}")
        print(f"Data mode: {'Individual integrations' if use_individual else 'Selection means'}")
        if use_individual and len(all_x) > 0:
            # Show uncertainty statistics to verify correct channels were used
            print(f"\nUncertainty statistics (verifies UNC vs 2SE channels):")
            print(f"  X uncertainty - min: {np.min(all_x_err):.6f}, max: {np.max(all_x_err):.6f}, mean: {np.mean(all_x_err):.6f}")
            print(f"  Y uncertainty - min: {np.min(all_y_err):.6f}, max: {np.max(all_y_err):.6f}, mean: {np.mean(all_y_err):.6f}")
            print(f"\nNote: UNC channel uncertainties should vary per-integration.")
            print(f"      If all uncertainties are identical, you may be using 2SE channels instead.")

        if len(all_x) == 0:
            print("ERROR: No data points collected")
            return None

        return {
            'x': np.array(all_x),
            'y': np.array(all_y),
            'x_err': np.array(all_x_err),
            'y_err': np.array(all_y_err),
            'rho': np.array(all_rho),
            'groups': all_groups,
            'gradient': np.array(all_gradient),
            'labels': all_labels
        }


    def york_regression(self, x, y, x_err, y_err, rho=None, fix_intercept=False, fixed_int=None, fixed_int_err=None):
        """
        York et al. (2004) regression with anchored intercept support

        Parameters:
        x, y: arrays of data points
        x_err, y_err: arrays of 2SE uncertainties
        rho: array of error correlations
        fix_intercept: boolean, whether to anchor the intercept
        fixed_int: anchored intercept value (added as a data point at x=0)
        fixed_int_err: anchored intercept uncertainty (2SE)

        Returns:
        slope, intercept, slope_err (2SE), intercept_err (2SE), mswd, probability, n, dof
        where n = total points used (including synthetic if anchored)
              dof = degrees of freedom
        """
        from scipy import stats

        n_original = len(x)

        # Handle anchored intercept by adding a synthetic data point
        if fix_intercept and fixed_int is not None:
            print(f"  Anchored intercept mode: adding synthetic point at (0, {fixed_int}) with uncertainty {fixed_int_err}")

            # Add synthetic data point at x=0, y=fixed_int
            x = np.append(x, 0.0)
            y = np.append(y, fixed_int)

            # Add uncertainties for the synthetic point
            # x uncertainty at intercept is essentially zero (very small)
            x_err = np.append(x_err, 1e-10)  # Very small x uncertainty

            # TESTING: If IsoplotR treats anchored intercept uncertainty as 1SE (not 2SE),
            # we need to double it to match the 2SE format of our other data
            # This way when York divides by 2, we get back to the original 1SE value
            y_err_synthetic = (fixed_int_err * 2.0) if fixed_int_err is not None else 1e-10
            y_err = np.append(y_err, y_err_synthetic)

            print(f"  TESTING: Treating fixed_int_err as 1SE, converting to 2SE for consistency")
            print(f"  Input uncertainty: {fixed_int_err} (assumed 1SE)")
            print(f"  Synthetic point y-uncertainty: {y_err_synthetic} (converted to 2SE)")

            # Add rho for synthetic point (zero correlation)
            if rho is None:
                rho = np.zeros(n_original + 1)
            else:
                rho = np.append(rho, 0.0)

            # Update n to include synthetic point
            n = len(x)
            print(f"  Total points including synthetic: {n}")

            # Set flag to adjust DOF later (we didn't "lose" a parameter for intercept)
            using_anchored_intercept = True
        else:
            n = len(x)
            using_anchored_intercept = False

            if rho is None:
                rho = np.zeros(n)

        # Convert 2SE to 1SE for the regression
        sigma_x = x_err / 2.0
        sigma_y = y_err / 2.0

        print(f"York regression inputs:")
        print(f"  n = {n} points (including {1 if using_anchored_intercept else 0} synthetic)")
        print(f"  x range: {x.min():.2f} to {x.max():.2f}")
        print(f"  y range: {y.min():.6f} to {y.max():.6f}")
        print(f"  sigma_x range: {sigma_x.min():.4f} to {sigma_x.max():.4f}")
        print(f"  sigma_y range: {sigma_y.min():.8f} to {sigma_y.max():.8f}")
        print(f"  rho range: {rho.min():.4f} to {rho.max():.4f}")
        print(f"  rho values: {rho}")

        # Weights (standard York approach - no special treatment)
        omega_x = 1.0 / (sigma_x**2)
        omega_y = 1.0 / (sigma_y**2)

        # Initial guess for slope (weighted least squares)
        w_init = omega_y
        x_mean = np.average(x, weights=w_init)
        y_mean = np.average(y, weights=w_init)
        b = np.sum(w_init * (x - x_mean) * (y - y_mean)) / np.sum(w_init * (x - x_mean)**2)

        print(f"  Initial slope guess (WLS): {b:.6e}")

        # Iterative solution (standard York algorithm)
        max_iterations = 10000
        tolerance = 1e-15

        iteration_count = 0
        for iteration in range(max_iterations):
            iteration_count = iteration + 1
            b_old = b

            # York 2004, eq. 13 - Effective weights
            alpha_term = rho * np.sqrt(omega_x * omega_y)
            denom_W = omega_x + b**2 * omega_y - 2.0 * b * alpha_term

            # Check for problematic denominators
            if iteration == 0:
                print(f"  First iteration diagnostics:")
                print(f"    alpha_term range: {alpha_term.min():.4e} to {alpha_term.max():.4e}")
                print(f"    W denominator range: {denom_W.min():.4e} to {denom_W.max():.4e}")
                if np.any(denom_W <= 0):
                    print(f"    WARNING: {np.sum(denom_W <= 0)} points have non-positive W denominator!")
                    print(f"    This can happen with high rho values. Consider disabling rho.")

            # Protect against division by zero or negative values
            # If denominator is too small or negative, use uncorrelated weights
            bad_denom = np.abs(denom_W) < 1e-10
            if np.any(bad_denom):
                # Fall back to uncorrelated weights for problematic points
                denom_W_safe = np.where(bad_denom, omega_x + b**2 * omega_y, denom_W)
                W = (omega_x * omega_y) / denom_W_safe
            else:
                W = (omega_x * omega_y) / denom_W

            W = np.abs(W)

            # Weighted means
            sum_W = np.sum(W)
            x_bar = np.sum(W * x) / sum_W
            y_bar = np.sum(W * y) / sum_W

            U = x - x_bar
            V = y - y_bar

            # Beta values (York 2004)
            beta = W * (U / omega_y + b * V / omega_x - (b * U + V) * alpha_term / (omega_x * omega_y))

            # New slope (York 2004, eq. 14)
            denom = np.sum(W * beta * U)
            if abs(denom) > 1e-30:
                b = np.sum(W * beta * V) / denom
            else:
                print(f"  Convergence stopped: denominator near zero at iteration {iteration_count}")
                print(f"    sum(W*beta*U) = {denom}")
                print(f"    This usually indicates highly correlated errors (high rho)")
                break

            # Check convergence with relative tolerance
            if abs(b - b_old) < tolerance * max(abs(b), 1.0):
                print(f"  Converged at iteration {iteration_count}, delta = {abs(b - b_old):.6e}")
                break

        if iteration_count >= max_iterations:
            print(f"  Warning: Max iterations ({max_iterations}) reached without full convergence")

        print(f"  Final slope after {iteration_count} iterations: {b:.10e}")

        # Final calculations
        a = y_bar - b * x_bar
        x_adj = x_bar + beta
        sigma_b = np.sqrt(1.0 / np.sum(W * (x_adj - x_bar)**2))
        sigma_a = np.sqrt(1.0 / sum_W + x_bar**2 / np.sum(W * (x_adj - x_bar)**2))

        # Goodness of fit
        y_fit = a + b * x
        S = np.sum(W * (y - y_fit)**2)

        # Adjust DOF for anchored intercept
        if using_anchored_intercept:
            # With anchored intercept, we have n points and fit 2 parameters
            # The synthetic point doesn't count as "real" DOF loss
            dof = n - 2
        else:
            # Normal case
            dof = n - 2

        if dof > 0:
            mswd = S / dof
            prob = 1.0 - stats.chi2.cdf(S, dof)
        else:
            mswd = 0.0
            prob = 0.0

        # Calculate covariance between slope and intercept
        # From York regression: cov(a, b) = -x_bar * var(b)
        # This is because a = y_bar - b * x_bar
        cov_ab = -x_bar * sigma_b**2

        print(f"York regression results:")
        print(f"  slope (b) = {b:.6e}")
        print(f"  intercept (a) = {a:.6f}")
        print(f"  sigma_b (1SE) = {sigma_b:.6e}")
        print(f"  sigma_a (1SE) = {sigma_a:.6e}")
        print(f"  cov(a,b) (1SE) = {cov_ab:.6e}")
        print(f"  MSWD = {mswd:.6f}")
        print(f"  DOF = {dof}")
        print(f"  n (total points used) = {n}")

        # Convert uncertainties back to 2SE
        sigma_b_2se = sigma_b * 2.0
        sigma_a_2se = sigma_a * 2.0
        # Covariance scales by 4 (2^2) when converting from 1SE to 2SE
        cov_ab_2se = cov_ab * 4.0

        return b, a, sigma_b_2se, sigma_a_2se, mswd, prob, n, dof, cov_ab_2se

    def calculate_age(self, slope, slope_err, mswd, prob, n, dof_override=None, is_model3=False, is_inverse=False,
                       intercept=None, intercept_err=None, cov_ab=None):
        """
        Calculate age from slope using Rb-87 decay constant

        Model 1 (York with MSWD):
            - Uses normal distribution (z) when MSWD ≈ 1 (p-value >= 0.05)
            - Uses t-distribution with sqrt(MSWD) scaling when overdispersed (p-value < 0.05)
            - This matches IsoplotR and DRS behavior

        Model 2 (TLS): Uses normal distribution (no analytical uncertainties to scale)

        Model 3 (York + Overdispersion): Uses t-distribution WITHOUT MSWD scaling
            - Dispersion parameter is already incorporated in the Hessian-derived uncertainties
            - Just use t-distribution with appropriate DOF

        Parameters:
        slope: regression slope (b)
        slope_err: slope uncertainty (2SE)
        mswd: mean square of weighted deviates
        prob: probability (p-value)
        n: number of data points (original, not including synthetic)
        dof_override: if provided, use this DOF instead of calculating from n
        is_model3: boolean indicating if this is Model 3 (York + overdispersion)
        is_inverse: boolean indicating if this is an inverse isochron
        intercept: regression intercept (a) - required for inverse isochron
        intercept_err: intercept uncertainty (2SE) - required for inverse isochron
        cov_ab: covariance between intercept and slope (2SE basis) - used for inverse isochron error propagation

        For conventional isochron: slope = e^λt - 1, so t = ln(1 + slope) / λ
        For inverse isochron (following IsoplotR):
            - D/P ratio = -a/b (negative intercept/slope)
            - t = ln(1 + D/P) / λ = ln(1 - a/b) / λ
        """
        from scipy import stats

        lambda_rb87 = 1.3972e-11  # yr^-1 (Villa et al., 2015)

        print(f"Calculate age: slope = {slope}, slope_err = {slope_err}, is_model3 = {is_model3}, is_inverse = {is_inverse}")

        if is_inverse:
            # For inverse isochron, following IsoplotR's ab2y0t.PD:
            # IsoplotR uses: DP <- quotient(X=fit$a, Y=fit$b) then DP[1] <- -DP[1]
            # quotient(X,Y) returns Y/X, so DP = -b/a = -slope/intercept
            # Age = ln(1 + DP) / λ = ln(1 - b/a) / λ

            if intercept is None or slope is None:
                print(f"  Error: inverse isochron requires both intercept and slope")
                return 0.0, 0.0

            if intercept == 0:
                print(f"  Warning: inverse isochron intercept = 0, returning age = 0")
                return 0.0, 0.0

            # Calculate D/P ratio = -b/a (IsoplotR formula)
            a = intercept
            b = slope
            DP = -b / a

            print(f"  Inverse isochron: a (intercept) = {a:.6e}, b (slope) = {b:.6e}")
            print(f"  D/P ratio = -b/a = {DP:.6f}")

            # Check validity
            arg = 1.0 + DP  # = 1 - b/a
            if arg <= 0:
                print(f"  Warning: invalid argument for log (1 + DP = {arg}), returning age = 0")
                return 0.0, 0.0

            age = np.log(arg) / lambda_rb87 / 1e6  # Convert to Ma

            # Error propagation for DP = -b/a
            # ∂(DP)/∂a = b/a²
            # ∂(DP)/∂b = -1/a
            # var(DP) = (b/a²)² * var(a) + (1/a)² * var(b) + 2*(b/a²)*(-1/a)*cov(a,b)
            #         = b²*var(a)/a⁴ + var(b)/a² - 2*b*cov(a,b)/a³

            # Convert 2SE to variance (1SE basis)
            var_a = (intercept_err / 2.0)**2 if intercept_err is not None else 0
            var_b = (slope_err / 2.0)**2
            cov_ab_1se = (cov_ab / 4.0) if cov_ab is not None else 0  # Convert from 2SE to 1SE covariance

            var_DP = b**2 * var_a / a**4 + var_b / a**2 - 2 * b * cov_ab_1se / a**3

            # Ensure non-negative variance
            if var_DP < 0:
                print(f"  Warning: negative variance in DP calculation, using uncorrelated formula")
                var_DP = b**2 * var_a / a**4 + var_b / a**2

            sigma_DP = np.sqrt(max(var_DP, 0))

            print(f"  var(a) = {var_a:.6e}, var(b) = {var_b:.6e}, cov(a,b) = {cov_ab_1se:.6e}")
            print(f"  var(DP) = {var_DP:.6e}, sigma(DP) = {sigma_DP:.6e}")

            # Age derivative: dt/d(DP) = 1/(λ * (1 + DP))
            dage_dDP = 1.0 / (lambda_rb87 * arg) / 1e6

            # For the rest of the function, we'll use sigma_DP instead of slope_err
            # We need to set up the variables for the common uncertainty calculation code
            dage_dslope = dage_dDP  # Using DP derivative instead of slope derivative
            slope_err = sigma_DP * 2.0  # Convert back to 2SE for consistency with rest of code

            print(f"  Age = ln({arg:.6f}) / λ = {age:.2f} Ma")
            print(f"  Age derivative: dage/d(DP) = {dage_dDP:.6e}")
        else:
            # Conventional isochron
            if slope <= 0:
                print(f"  Warning: slope <= 0, returning age = 0")
                return 0.0, 0.0

            # Calculate age
            age = np.log(1 + slope) / lambda_rb87 / 1e6  # Convert to Ma

            # Derivative for error propagation: dt/d(slope) = 1/(λ * (1 + slope))
            dage_dslope = 1.0 / (lambda_rb87 * (1 + slope)) / 1e6

        # Age uncertainty
        alpha = 0.05

        # Calculate degrees of freedom
        if dof_override is not None:
            dof = dof_override
            print(f"  Using DOF override: dof={dof} (instead of n-2={n-2})")
        else:
            dof = n - 2  # For free intercept

        # Handle different regression models
        print(f"  DIAGNOSTIC: n={n}, dof={dof}, mswd={mswd}, prob={prob}")

        if is_model3:
            # Model 3: IsoplotR simply multiplies 1SE by 2 for 95% CI display
            # (not 1.96 or t-distribution)
            slope_err_1se = slope_err / 2.0
            print(f"  Model 3 path: Using 1SE × 2 for 95% CI (matching IsoplotR)")
            s_slope_for_age = slope_err  # slope_err is already 2SE
            print(f"  Using slope_err (1SE) = {slope_err_1se:.6e}")
            print(f"  Using slope_err (2SE) = {slope_err:.6e}")
            # Calculate age uncertainty using derivative
            age_err = np.abs(dage_dslope) * s_slope_for_age
            t_value = 2.0  # For display purposes only
            print(f"  FINAL: t_value = {t_value:.6f} (display only), s_slope_for_age = {s_slope_for_age:.6e}")
            print(f"  FINAL: age = {age:.2f} ± {age_err:.2f} Ma")
        elif mswd is not None and prob is not None:
            # Model 1: York with MSWD/prob
            # slope_err is 2SE, convert to 1SE
            s_slope = slope_err / 2.0
            print(f"  Model 1 path (has MSWD and prob)")
            if prob < alpha:
                # MSWD significantly > 1: apply overdispersion correction
                t_value = stats.t.ppf(1 - alpha/2, dof)
                s_slope_for_age = s_slope * np.sqrt(mswd)
                print(f"  Note: p-value ({prob:.4f}) < {alpha}, applying overdispersion correction")
                print(f"  MSWD = {mswd:.4f}, multiplier = {np.sqrt(mswd):.4f}")
                print(f"  Using t-distribution: t_value = {t_value:.6f}")
            else:
                # MSWD ≈ 1 or scatter consistent with analytical uncertainties
                # Use NORMAL distribution (matches IsoplotR and DRS)
                t_value = stats.norm.ppf(1 - alpha/2)
                s_slope_for_age = s_slope
                print(f"  p-value ({prob:.4f}) >= {alpha}, NO overdispersion correction")
                print(f"  Using NORMAL distribution: z_value = {t_value:.6f}")
            age_err = t_value * np.abs(dage_dslope) * s_slope_for_age
            print(f"  FINAL: t_value = {t_value:.6f}, s_slope_for_age = {s_slope_for_age:.6e}")
            print(f"  FINAL: age = {age:.2f} ± {age_err:.2f} Ma")
        else:
            # Model 2: Use NORMAL distribution (matches IsoplotR)
            # slope_err is 2SE, convert to 1SE
            s_slope = slope_err / 2.0
            t_value = stats.norm.ppf(1 - alpha/2)
            s_slope_for_age = s_slope
            print(f"  Model 2 path (no MSWD/prob)")
            print(f"  Using NORMAL distribution: z_value = {t_value:.6f}")
            age_err = t_value * np.abs(dage_dslope) * s_slope_for_age
            print(f"  FINAL: t_value = {t_value:.6f}, s_slope_for_age = {s_slope_for_age:.6e}")
            print(f"  FINAL: age = {age:.2f} ± {age_err:.2f} Ma")

        return age, age_err

    def tls_regression(self, x, y, fix_intercept=False, fixed_int=None, fixed_int_err=None):
        """
        Total Least Squares regression (Model 2)
        Uses PCA to find best-fit line, ignoring analytical uncertainties
        Jackknife resampling for uncertainty estimation

        Based on IsoplotR's tls() function

        Parameters:
        x, y: arrays of data points
        fix_intercept: boolean, whether to fix the intercept (truly fixed, no uncertainty)
        fixed_int: fixed intercept value
        fixed_int_err: not used for TLS (intercept is truly fixed)

        Returns:
        slope, intercept, slope_err (2SE), intercept_err (2SE), mswd, probability, n, dof

        Note: MSWD and probability are None for Model 2 as it doesn't use analytical uncertainties
        """
        from sklearn.decomposition import PCA
        import numpy as np
        from scipy import stats

        n = len(x)

        if fix_intercept and fixed_int is not None:
            # Truly fixed intercept TLS - minimize orthogonal distances
            print(f"TLS with fixed intercept = {fixed_int} (no uncertainty)")

            # For TLS with fixed intercept, we need to minimize orthogonal distances
            # to a line y = fixed_int + slope * x
            # This is different from PCA which goes through the centroid

            def orthogonal_regression_fixed_intercept(x_data, y_data, intercept):
                """Find slope that minimizes orthogonal distance to y = intercept + slope*x"""
                # Shift data
                y_shifted = y_data - intercept

                # For orthogonal regression through origin:
                # Minimize sum of (y_shifted - slope*x)^2 / (1 + slope^2)
                # This has analytical solution

                # Calculate sums needed for solution
                sum_x2 = np.sum(x_data**2)
                sum_y2 = np.sum(y_shifted**2)
                sum_xy = np.sum(x_data * y_shifted)

                # Analytical solution for orthogonal regression through origin
                # slope = (sum_y2 - sum_x2 + sqrt((sum_y2-sum_x2)^2 + 4*sum_xy^2)) / (2*sum_xy)
                delta = sum_y2 - sum_x2
                discriminant = delta**2 + 4 * sum_xy**2

                if sum_xy > 0:
                    slope = (delta + np.sqrt(discriminant)) / (2 * sum_xy)
                else:
                    slope = (delta - np.sqrt(discriminant)) / (2 * sum_xy)

                return slope

            # Fit slope
            slope = orthogonal_regression_fixed_intercept(x, y, fixed_int)
            intercept = fixed_int

            print(f"  Initial fit: slope = {slope:.6e}")

            # Calculate uncertainty using IsoplotR's exact method
            # Based on anchored.deming function in tls.R

            y_shifted = y - fixed_int
            # Predicted y values
            y_pred = slope * x
            # Orthogonal residuals (same as IsoplotR: deming_residuals)
            residuals = (y_shifted - y_pred) / np.sqrt(1 + slope**2)

            # Variance of residuals (R's var() uses n-1 denominator)
            ve = np.var(residuals, ddof=1)

            # Calculate Hessian (second derivative of misfit function)
            # The misfit function is sum(residuals^2)
            # For orthogonal regression, we need the second derivative w.r.t. slope

            # Numerical Hessian using finite differences
            from scipy.optimize import approx_fprime

            def misfit_for_slope(b):
                """Misfit function for slope (intercept fixed at fixed_int)"""
                # b might be array or scalar, handle both
                if isinstance(b, np.ndarray):
                    b = b[0]
                resid = (y - fixed_int - b * x) / np.sqrt(1 + b**2)
                return np.sum(resid**2)

            eps = np.sqrt(np.finfo(float).eps)

            # Calculate second derivative using finite differences manually
            # H = d²f/db² ≈ (f(b+h) - 2*f(b) + f(b-h)) / h²
            h = eps * max(abs(slope), 1.0)
            f_plus = misfit_for_slope(slope + h)
            f_center = misfit_for_slope(slope)
            f_minus = misfit_for_slope(slope - h)
            H = (f_plus - 2*f_center + f_minus) / (h**2)

            # Slope variance following IsoplotR: inverthess(H) * ve
            # inverthess(H) is likely just 1/H
            if abs(H) > 1e-10:
                slope_var = ve / H
            else:
                slope_var = ve / 1e-10  # Fallback

            slope_err_1se = np.sqrt(abs(slope_var))  # Take abs in case of numerical issues
            slope_err = 2.0 * slope_err_1se  # Convert to 2SE

            intercept_err = 0.0  # Truly fixed, no uncertainty

            print(f"  IsoplotR method (anchored.deming):")
            print(f"  Variance of residuals (ve): {ve:.6e}")
            print(f"  Hessian (H): {H:.6e}")
            print(f"  Slope variance (ve/H): {slope_var:.6e}")
            print(f"  Slope 1SE: {slope_err_1se:.6e}")
            print(f"  Slope 2SE: {slope_err:.6e}")
            print(f"  Final: slope = {slope:.6e} ± {slope_err:.6e} (2SE)")

            # For fixed intercept, covariance is 0 since intercept has no uncertainty
            cov_ab_2se = 0.0

        else:
            # Free intercept using standard PCA
            print("TLS with free intercept (PCA)")

            # Combine x and y into data matrix
            data = np.column_stack([x, y])

            # Perform PCA
            pca = PCA(n_components=1)
            pca.fit(data)

            # First principal component gives the direction of the line
            pc1 = pca.components_[0]
            slope = pc1[1] / pc1[0]

            # Intercept: line passes through centroid
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            intercept = y_mean - slope * x_mean

            # Jackknife uncertainty estimation
            jack_slopes = np.zeros(n)
            jack_intercepts = np.zeros(n)

            for i in range(n):
                # Leave out point i
                x_jack = np.delete(x, i)
                y_jack = np.delete(y, i)

                # Fit PCA on remaining points
                data_jack = np.column_stack([x_jack, y_jack])
                pca_jack = PCA(n_components=1)
                pca_jack.fit(data_jack)

                pc1_jack = pca_jack.components_[0]
                slope_jack = pc1_jack[1] / pc1_jack[0]

                x_mean_jack = np.mean(x_jack)
                y_mean_jack = np.mean(y_jack)
                intercept_jack = y_mean_jack - slope_jack * x_mean_jack

                jack_slopes[i] = slope_jack
                jack_intercepts[i] = intercept_jack

            # Calculate jackknife standard errors
            slope_mean = np.mean(jack_slopes)
            intercept_mean = np.mean(jack_intercepts)

            slope_var = ((n - 1) / n) * np.sum((jack_slopes - slope_mean)**2)
            intercept_var = ((n - 1) / n) * np.sum((jack_intercepts - intercept_mean)**2)

            # Calculate covariance between slope and intercept from jackknife samples
            cov_ab_1se = ((n - 1) / n) * np.sum((jack_intercepts - intercept_mean) * (jack_slopes - slope_mean))

            slope_err = 2.0 * np.sqrt(slope_var)  # 2SE
            intercept_err = 2.0 * np.sqrt(intercept_var)  # 2SE
            cov_ab_2se = cov_ab_1se * 4.0  # Convert to 2SE scale

        # Model 2 doesn't calculate MSWD/probability (no analytical uncertainties)
        mswd = None
        prob = None
        dof = n - 2  # Standard DOF for linear regression

        print(f"TLS Regression: slope = {slope:.6f} ± {slope_err:.6f}, intercept = {intercept:.6f} ± {intercept_err:.6f}")

        return slope, intercept, slope_err, intercept_err, mswd, prob, n, dof, cov_ab_2se

    def york_overdispersion_regression(self, x, y, x_err, y_err, rho=None,
                                       fix_intercept=False, fixed_int=None, fixed_int_err=None, wtype='a'):
        """
        York regression with overdispersion parameter (Model 3)
        Adds geological dispersion variance to the regression

        Based on IsoplotR's MLyork() function with model=3

        For Model 3 with anchored intercept (IsoplotR behavior):
        - The intercept is TRULY FIXED at fixed_int
        - The anchor uncertainty (fixed_int_err) IS the dispersion parameter σₐ
        - Only the slope is optimized
        - This matches IsoplotR's LL_3a(b,x|a̅,σ_a̅,X,Y,Σ) formulation

        Parameters:
        x, y: arrays of data points
        x_err, y_err: arrays of 2SE uncertainties
        rho: array of error correlations
        fix_intercept: boolean, whether to fix the intercept
        fixed_int: fixed intercept value
        fixed_int_err: fixed intercept uncertainty (1SE) - THIS IS THE DISPERSION for Model 3
        wtype: 'a' for constant dispersion (initial ratio scatter, default)
               'b' for proportional dispersion (age dispersion)

        Returns:
        slope, intercept, slope_err (2SE), intercept_err (2SE), mswd, probability, dispersion, dispersion_err
        """
        print("*** VERSION CHECK: Model 3 - Dispersion = Anchor Uncertainty (IsoplotR style) - 2026-01-02 ***")
        from scipy.optimize import minimize
        from scipy import stats
        import numpy as np

        n = len(x)

        # Convert 2SE to 1SE for calculations
        sigma_x = x_err / 2.0
        sigma_y = y_err / 2.0

        # Use rho if provided, otherwise assume 0
        if rho is None:
            rho = np.zeros(n)

        wtype_desc = "constant dispersion (initial ratio)" if wtype == 'a' else "proportional dispersion (age)"
        print(f"York + Overdispersion regression (Model 3) with wtype='{wtype}' ({wtype_desc})")

        # For Model 3 with fixed intercept:
        # - The intercept is truly fixed at fixed_int
        # - The anchor uncertainty (fixed_int_err) IS the dispersion parameter
        # - We only optimize the slope
        if fix_intercept and fixed_int is not None:
            using_fixed_intercept = True
            fixed_intercept_value = fixed_int

            # The anchor uncertainty IS the dispersion (IsoplotR Model 3 behavior)
            # fixed_int_err is expected to be 1SE
            if fixed_int_err is not None and fixed_int_err > 0:
                dispersion = fixed_int_err  # Dispersion = anchor uncertainty (1SE)
            else:
                dispersion = 0.0

            print(f"  Model 3 with FIXED intercept = {fixed_intercept_value}")
            print(f"  Anchor uncertainty (1SE) = {fixed_int_err} -> THIS IS the dispersion parameter")
            print(f"  Only optimizing slope (dispersion is fixed at anchor uncertainty)")

            # Get initial slope guess from York regression
            slope_init, intercept_init, slope_err_init, intercept_err_init, mswd_init, prob_init, n_york, dof_york, _ = self.york_regression(
                x, y, x_err, y_err, rho=rho,
                fix_intercept=True, fixed_int=fixed_int, fixed_int_err=fixed_int_err
            )
            print(f"  Initial York regression: slope = {slope_init:.6f}, MSWD = {mswd_init:.4f}")

        else:
            # Free intercept case - optimize intercept, slope, and dispersion
            using_fixed_intercept = False
            fixed_intercept_value = None
            dispersion = None  # Will be optimized

            # Get initial guess from standard York regression
            slope_init, intercept_init, slope_err_init, intercept_err_init, mswd_init, prob_init, n_york, dof_york, _ = self.york_regression(
                x, y, x_err, y_err, rho=rho,
                fix_intercept=False, fixed_int=None, fixed_int_err=None
            )
            print(f"  Initial York regression: slope = {slope_init:.6f}, intercept = {intercept_init:.6f}, MSWD = {mswd_init:.4f}")

        # Check if data is underdispersed (MSWD < 1)
        if mswd_init < 1.0:
            print(f"  Note: MSWD < 1 indicates underdispersion (scatter less than analytical uncertainties)")

        if using_fixed_intercept:
            # =========================================================
            # FIXED INTERCEPT CASE (Model 3 anchored)
            # Dispersion = anchor uncertainty, only optimize slope
            # =========================================================

            # Define negative log-likelihood with fixed intercept and fixed dispersion
            def neg_log_likelihood_fixed(params):
                """
                Negative log-likelihood for Model 3 with fixed intercept
                params = [slope] only
                Dispersion is fixed at the anchor uncertainty
                """
                b = params[0]  # slope
                a = fixed_intercept_value  # intercept (fixed)
                w = dispersion  # dispersion (fixed at anchor uncertainty)

                nll = 0.0

                for i in range(n):
                    var_x = sigma_x[i]**2

                    # Add dispersion based on wtype
                    if wtype == 'a':
                        var_y = sigma_y[i]**2 + w**2
                    else:  # wtype == 'b'
                        var_y = sigma_y[i]**2 + (w * x[i])**2

                    cov_xy = rho[i] * sigma_x[i] * sigma_y[i]

                    det_E = var_x * var_y - cov_xy**2
                    if det_E <= 0:
                        return 1e10

                    omega_xx = var_y / det_E
                    omega_yy = var_x / det_E
                    omega_xy = -cov_xy / det_E

                    alpha = omega_xx + omega_yy * b**2 + 2 * omega_xy * b
                    beta = (y[i] - a - b * x[i]) * (omega_xy + b * omega_yy)

                    if alpha <= 0:
                        return 1e10

                    x_opt = x[i] + beta / alpha

                    dx = x[i] - x_opt
                    dy = y[i] - a - b * x_opt

                    maha = (omega_xx * dx**2 + 2 * omega_xy * dx * dy + omega_yy * dy**2)
                    nll += 0.5 * (np.log(det_E) + maha)

                return nll

            # Optimize slope only
            print(f"  Optimizing slope only (intercept and dispersion fixed)")
            print(f"  Initial slope: b={slope_init:.6e}")

            result = minimize(neg_log_likelihood_fixed, [slope_init], method='BFGS',
                             options={'gtol': 1e-8, 'disp': False})

            slope = result.x[0]
            intercept = fixed_intercept_value

            print(f"  BFGS optimization: success={result.success}, nfev={result.nfev}")
            print(f"  Optimized slope: b={slope:.6e}")
            print(f"  Intercept FIXED at: a={intercept:.6f}")
            print(f"  Dispersion FIXED at: w={dispersion:.6e} (= anchor uncertainty)")

            # Calculate slope uncertainty from Hessian (1x1 matrix for slope only)
            try:
                eps = 0.0031 * max(abs(slope), 1.0)
                f0 = neg_log_likelihood_fixed([slope])
                f_plus = neg_log_likelihood_fixed([slope + eps])
                f_minus = neg_log_likelihood_fixed([slope - eps])
                hessian_slope = (f_plus - 2*f0 + f_minus) / (eps**2)

                print(f"  Hessian for slope: {hessian_slope:.6e}")

                if hessian_slope > 0:
                    slope_err_1se = 1.0 / np.sqrt(hessian_slope)
                    print(f"  Slope uncertainty from Hessian (1SE): {slope_err_1se:.6e}")
                else:
                    # Fallback to York uncertainty
                    slope_err_1se = slope_err_init / 2.0
                    print(f"  Hessian invalid, using York uncertainty (1SE): {slope_err_1se:.6e}")
            except Exception as e:
                print(f"  Hessian calculation failed: {e}")
                slope_err_1se = slope_err_init / 2.0

            slope_err = 2.0 * slope_err_1se
            intercept_err = 0.0  # Fixed intercept has no uncertainty
            dispersion_err = 0.0  # Dispersion is fixed at anchor uncertainty

            # Calculate MSWD with the fixed dispersion
            if wtype == 'a':
                var_y_aug = sigma_y**2 + dispersion**2
            else:
                var_y_aug = sigma_y**2 + (dispersion * x)**2

            chi_sq = 0.0
            for i in range(n):
                resid = y[i] - intercept - slope * x[i]
                chi_sq += resid**2 / var_y_aug[i]

            # DOF = n - 1 (only slope is fitted)
            dof = n - 1
            mswd_final = chi_sq / dof
            prob_final = 1.0 - stats.chi2.cdf(chi_sq, dof)

            print(f"  Final: slope = {slope:.6f} ± {slope_err:.6f}")
            print(f"         intercept = {intercept:.6f} (FIXED)")
            print(f"         dispersion = {dispersion:.6e} (= anchor uncertainty)")
            print(f"         MSWD = {mswd_final:.4f}, p-value = {prob_final:.4f}")

            return slope, intercept, slope_err, intercept_err, mswd_final, prob_final, dispersion, dispersion_err

        else:
            # =========================================================
            # FREE INTERCEPT CASE - optimize intercept, slope, and dispersion
            # =========================================================

            # Initial guess for log(dispersion)
            lw_init = 0.0

            # Define negative log-likelihood function for free intercept case
            def neg_log_likelihood(params):
                """
                Negative log-likelihood for York regression with overdispersion
                params = [intercept, slope, log(w)]
                """
                a = params[0]  # intercept
                b = params[1]  # slope
                lw = params[2]  # log(dispersion)
                w = np.exp(lw)  # dispersion parameter

                nll = 0.0  # negative log-likelihood

                for i in range(n):
                    # Variance-covariance matrix for this point
                    var_x = sigma_x[i]**2

                    # Add dispersion based on wtype
                    if wtype == 'a':
                        # Constant dispersion on Y (initial ratio scatter)
                        var_y = sigma_y[i]**2 + w**2
                    else:  # wtype == 'b'
                        # Proportional dispersion on Y (age dispersion)
                        var_y = sigma_y[i]**2 + (w * x[i])**2

                    cov_xy = rho[i] * sigma_x[i] * sigma_y[i]

                    # Determinant of covariance matrix
                    det_E = var_x * var_y - cov_xy**2

                    if det_E <= 0:
                        return 1e10  # Invalid covariance matrix

                    # Inverse of covariance matrix
                    omega_xx = var_y / det_E
                    omega_yy = var_x / det_E
                    omega_xy = -cov_xy / det_E

                    # Find optimal x_i (point on the line closest to measured point)
                    alpha = omega_xx + omega_yy * b**2 + 2 * omega_xy * b
                    beta = (y[i] - a - b * x[i]) * (omega_xy + b * omega_yy)

                    if alpha <= 0:
                        return 1e10  # Invalid

                    x_opt = x[i] + beta / alpha

                    # Mahalanobis distance (chi-square contribution)
                    dx = x[i] - x_opt
                    dy = y[i] - a - b * x_opt

                    maha = (omega_xx * dx**2 + 2 * omega_xy * dx * dy + omega_yy * dy**2)

                    # Log-likelihood contribution
                    nll += 0.5 * (np.log(det_E) + maha)

                return nll

            from scipy.optimize import minimize

            # Optimize all 3 parameters: intercept, slope, and log(dispersion)
            print(f"  Optimizing intercept, slope, and dispersion")
            print(f"  Initial params: a={intercept_init:.6f}, b={slope_init:.6e}, lw={lw_init:.6f}")
            init_params = [intercept_init, slope_init, lw_init]

            # Use BFGS which computes inverse Hessian approximation
            result = minimize(neg_log_likelihood, init_params, method='BFGS',
                             options={'gtol': 1e-8, 'disp': False})

            print(f"  BFGS optimization: success={result.success}, nfev={result.nfev}")
            print(f"  Final params: a={result.x[0]:.6f}, b={result.x[1]:.6e}, lw={result.x[2]:.6f}")
            print(f"  Final dispersion w = exp(lw) = {np.exp(result.x[2]):.6e}")
            print(f"  Final NLL = {result.fun:.6f}")

            # Extract results
            intercept = result.x[0]
            slope = result.x[1]
            lw = result.x[2]
            dispersion = np.exp(lw)

            print(f"  Optimized: slope = {slope:.6f}, intercept = {intercept:.6f}, dispersion = {dispersion:.6f}")

            # Calculate uncertainties from 3x3 Hessian matrix
            print("  Calculating uncertainties from Hessian matrix...")

            try:
                n_params = 3
                hessian = np.zeros((n_params, n_params))
                params_opt = np.array([intercept, slope, lw])

                # Step size for numerical Hessian computation
                ndeps = 0.0031
                eps_vec = np.array([ndeps * max(abs(p), 1.0) for p in params_opt])

                print(f"  R-style eps per parameter: [{eps_vec[0]:.6e}, {eps_vec[1]:.6e}, {eps_vec[2]:.6e}]")

                # Central difference Hessian with parameter-scaled step sizes
                f0 = neg_log_likelihood(params_opt)
                for i in range(n_params):
                    for j in range(n_params):
                        eps_i = eps_vec[i]
                        eps_j = eps_vec[j]

                        if i == j:
                            p_plus = params_opt.copy()
                            p_plus[i] += eps_i
                            p_minus = params_opt.copy()
                            p_minus[i] -= eps_i
                            hessian[i, i] = (neg_log_likelihood(p_plus) - 2*f0 + neg_log_likelihood(p_minus)) / (eps_i**2)
                        else:
                            p_pp = params_opt.copy()
                            p_pp[i] += eps_i
                            p_pp[j] += eps_j
                            p_pm = params_opt.copy()
                            p_pm[i] += eps_i
                            p_pm[j] -= eps_j
                            p_mp = params_opt.copy()
                            p_mp[i] -= eps_i
                            p_mp[j] += eps_j
                            p_mm = params_opt.copy()
                            p_mm[i] -= eps_i
                            p_mm[j] -= eps_j
                            hessian[i, j] = (neg_log_likelihood(p_pp) - neg_log_likelihood(p_pm) - neg_log_likelihood(p_mp) + neg_log_likelihood(p_mm)) / (4 * eps_i * eps_j)

                print(f"  Using R-style Hessian (3x3)")
                print(f"  Hessian matrix:")
                print(f"    [{hessian[0,0]:.6e}, {hessian[0,1]:.6e}, {hessian[0,2]:.6e}]")
                print(f"    [{hessian[1,0]:.6e}, {hessian[1,1]:.6e}, {hessian[1,2]:.6e}]")
                print(f"    [{hessian[2,0]:.6e}, {hessian[2,1]:.6e}, {hessian[2,2]:.6e}]")

                # Invert Hessian to get covariance matrix
                cov_matrix = np.linalg.inv(hessian)

                print(f"  Covariance matrix (inverted Hessian):")
                print(f"    [{cov_matrix[0,0]:.6e}, {cov_matrix[0,1]:.6e}, {cov_matrix[0,2]:.6e}]")
                print(f"    [{cov_matrix[1,0]:.6e}, {cov_matrix[1,1]:.6e}, {cov_matrix[1,2]:.6e}]")
                print(f"    [{cov_matrix[2,0]:.6e}, {cov_matrix[2,1]:.6e}, {cov_matrix[2,2]:.6e}]")

                # Extract uncertainties: [0] = intercept, [1] = slope, [2] = log(dispersion)
                intercept_err_1se_hess = np.sqrt(abs(cov_matrix[0, 0]))
                slope_err_1se_hess = np.sqrt(abs(cov_matrix[1, 1]))
                dispersion_err_1se_log = np.sqrt(abs(cov_matrix[2, 2]))

                # Transform dispersion uncertainty from log-space to linear space
                # se(w) ≈ w × se(log w)
                dispersion_err_1se = dispersion * dispersion_err_1se_log
                dispersion_err = 2.0 * dispersion_err_1se

                # For Model 3, use numerical Hessian with calibrated step size
                print(f"  Model 3: Using numerical Hessian for uncertainties (calibrated to match IsoplotR)")
                slope_err_1se = slope_err_1se_hess
                intercept_err_1se = intercept_err_1se_hess

                slope_err = 2.0 * slope_err_1se
                intercept_err = 2.0 * intercept_err_1se

                print(f"  Final uncertainties (1SE): slope = {slope_err_1se:.6e}, intercept = {intercept_err_1se:.6e}")
                print(f"  York uncertainties (1SE): slope = {slope_err_init/2:.6e}, intercept = {intercept_err_init/2:.6e}")
                print(f"  Numerical Hessian uncertainties (1SE): slope = {slope_err_1se_hess:.6e}, intercept = {intercept_err_1se_hess:.6e}")

                print(f"  Hessian-derived uncertainties (1SE): slope = {slope_err_1se:.6e}, intercept = {intercept_err_1se:.6e}, dispersion = {dispersion_err_1se:.6e}")

                # Calculate MSWD and probability with augmented uncertainties
                # Add dispersion to y uncertainties and recalculate
                if wtype == 'a':
                    var_y_aug = sigma_y**2 + dispersion**2
                else:  # wtype == 'b'
                    var_y_aug = sigma_y**2 + (dispersion * x)**2

                # Calculate chi-square with augmented uncertainties
                chi_sq = 0.0
                for i in range(n):
                    resid = y[i] - intercept - slope * x[i]
                    chi_sq += resid**2 / var_y_aug[i]

                # DOF for Model 3 with free intercept: n - 2 (two parameters: intercept, slope)
                # Dispersion is not counted as it's a variance component, not a location parameter
                dof = n - 2

                mswd_final = chi_sq / dof
                prob_final = 1.0 - stats.chi2.cdf(chi_sq, dof)

            except (np.linalg.LinAlgError, ValueError, Exception) as e:
                print(f"  Warning: Hessian calculation failed ({e}), using York regression uncertainties")
                # Fallback: use initial York regression uncertainties
                slope_err = slope_err_init
                intercept_err = intercept_err_init
                slope_err_1se = slope_err / 2.0
                intercept_err_1se = intercept_err / 2.0
                dispersion_err = 0.0
                dispersion_err_1se = 0.0
                mswd_final = mswd_init
                prob_final = prob_init

            # Print final results
            if dispersion >= 1e-10:
                print(f"  Final uncertainties: slope = {slope:.6f} ± {slope_err:.6f} (2SE)")

            print(f"  Final: slope = {slope:.6f} ± {slope_err:.6f}")
            print(f"         intercept = {intercept:.6f} ± {intercept_err:.6f}")
            print(f"         dispersion = {dispersion:.8e} ± {dispersion_err:.8e}")
            print(f"         MSWD = {mswd_final:.4f}, p-value = {prob_final:.4f}")

            return slope, intercept, slope_err, intercept_err, mswd_final, prob_final, dispersion, dispersion_err

    def update_plot(self):
        """Update the isochron plot"""
        # Validate inputs
        if not self.x_channel_combo.currentText:
            QMessageBox.warning(self, "No X Channel", "Please select an X-axis channel (Rb/Sr ratio).")
            return

        if not self.y_channel_combo.currentText:
            QMessageBox.warning(self, "No Y Channel", "Please select a Y-axis channel (Sr/Sr ratio).")
            return

        selected_groups = self.get_selected_groups()
        if not selected_groups:
            QMessageBox.warning(self, "No Groups Selected", "Please check at least one selection group to plot.")
            return

        # Load data
        plot_data = self.load_data_for_groups()

        if plot_data is None:
            QMessageBox.warning(self, "No Data", "No data found for the selected channels and groups.")
            self.figure.clear()
            self.canvas.draw()
            return

        x = plot_data['x']
        y = plot_data['y']
        x_err = plot_data['x_err']
        y_err = plot_data['y_err']
        rho = plot_data['rho']
        groups = plot_data['groups']
        gradient = plot_data['gradient']
        labels = plot_data.get('labels', [])  # Get labels for hover functionality

        # Apply inverse isochron transformation if enabled
        if self.inverse_isochron_check.isChecked():
            print("\n=== Applying Inverse Isochron Transformation ===")
            print(f"Original: X = ⁸⁷Rb/⁸⁶Sr, Y = ⁸⁷Sr/⁸⁶Sr")
            print(f"Inverse:  X = ⁸⁷Rb/⁸⁷Sr, Y = ⁸⁶Sr/⁸⁷Sr")

            # Check if data is suitable for inverse isochron
            # Very high 87Sr/86Sr (>10) indicates highly radiogenic samples where inverse isochron
            # may not be meaningful
            if np.max(y) > 10:
                print(f"\n  WARNING: Very high ⁸⁷Sr/⁸⁶Sr ratios detected (max = {np.max(y):.1f})")
                print(f"  Inverse isochron may not be appropriate for highly radiogenic samples.")
                print(f"  Consider using conventional isochron instead.")

            # Store original data for reference
            x_orig, y_orig = x.copy(), y.copy()
            x_err_orig, y_err_orig = x_err.copy(), y_err.copy()
            rho_orig = rho.copy()

            # Transform to inverse isochron coordinates
            # X_inv = X / Y = (⁸⁷Rb/⁸⁶Sr) / (⁸⁷Sr/⁸⁶Sr) = ⁸⁷Rb/⁸⁷Sr
            # Y_inv = 1 / Y = 1 / (⁸⁷Sr/⁸⁶Sr) = ⁸⁶Sr/⁸⁷Sr
            x_inv = x_orig / y_orig
            y_inv = 1.0 / y_orig

            # Error propagation for inverse coordinates
            # Convert 2SE to 1SE for calculations
            sigma_x = x_err_orig / 2.0
            sigma_y = y_err_orig / 2.0
            cov_xy = rho_orig * sigma_x * sigma_y

            # For X_inv = X/Y, using partial derivatives:
            # ∂(X/Y)/∂X = 1/Y
            # ∂(X/Y)/∂Y = -X/Y²
            # var(X_inv) = (1/Y)²*var(X) + (X/Y²)²*var(Y) + 2*(1/Y)*(-X/Y²)*cov(X,Y)
            #            = var(X)/Y² + X²*var(Y)/Y⁴ - 2*X*cov(X,Y)/Y³

            # Calculate each term separately for debugging
            term1 = sigma_x**2 / y_orig**2
            term2 = x_orig**2 * sigma_y**2 / y_orig**4
            term3 = 2 * x_orig * cov_xy / y_orig**3

            var_x_inv = term1 + term2 - term3

            # Ensure non-negative variance - if negative, use simpler formula without correlation
            negative_var_mask = var_x_inv < 0
            if np.any(negative_var_mask):
                print(f"  Note: {np.sum(negative_var_mask)} points have negative variance (strong correlation)")
                print(f"  Using uncorrelated error propagation for these points")
                # For these points, use the uncorrelated formula (ignoring rho)
                var_x_inv_uncorr = term1 + term2
                var_x_inv = np.where(negative_var_mask, var_x_inv_uncorr, var_x_inv)

            var_x_inv = np.maximum(var_x_inv, 1e-30)  # Ensure positive
            sigma_x_inv = np.sqrt(var_x_inv)
            x_inv_err = sigma_x_inv * 2.0  # Convert back to 2SE

            # For Y_inv = 1/Y:
            # ∂(1/Y)/∂Y = -1/Y²
            # var(Y_inv) = (1/Y²)²*var(Y) = var(Y)/Y⁴
            var_y_inv = sigma_y**2 / y_orig**4
            sigma_y_inv = np.sqrt(np.maximum(var_y_inv, 1e-30))
            y_inv_err = sigma_y_inv * 2.0  # Convert back to 2SE

            # Rho transformation for inverse coordinates
            # cov(X_inv, Y_inv) = X*var(Y)/Y⁴ - cov(X,Y)/Y³
            cov_inv = x_orig * sigma_y**2 / y_orig**4 - cov_xy / y_orig**3

            # Convert to correlation coefficient
            denom = sigma_x_inv * sigma_y_inv
            # Use safe division
            rho_inv = np.divide(cov_inv, denom, out=np.zeros_like(cov_inv), where=denom > 1e-30)

            # Clip rho to valid range [-1, 1]
            rho_inv = np.clip(rho_inv, -1.0, 1.0)

            # Final check for NaN/Inf values
            nan_mask_x = ~np.isfinite(x_inv_err)
            nan_mask_y = ~np.isfinite(y_inv_err)
            nan_mask_rho = ~np.isfinite(rho_inv)

            if np.any(nan_mask_x):
                print(f"  WARNING: {np.sum(nan_mask_x)} NaN/Inf in X uncertainties, using fallback")
                rel_err = np.sqrt((x_err_orig/x_orig)**2 + (y_err_orig/y_orig)**2)
                x_inv_err = np.where(nan_mask_x, rel_err * np.abs(x_inv), x_inv_err)

            if np.any(nan_mask_y):
                print(f"  WARNING: {np.sum(nan_mask_y)} NaN/Inf in Y uncertainties, using fallback")
                y_inv_err = np.where(nan_mask_y, (y_err_orig/y_orig**2), y_inv_err)

            if np.any(nan_mask_rho):
                print(f"  WARNING: {np.sum(nan_mask_rho)} NaN/Inf in rho, setting to 0")
                rho_inv = np.where(nan_mask_rho, 0.0, rho_inv)

            # Update the data arrays
            x = x_inv
            y = y_inv
            x_err = x_inv_err
            y_err = y_inv_err
            rho = rho_inv

            # Update plot_data dict as well
            plot_data['x'] = x
            plot_data['y'] = y
            plot_data['x_err'] = x_err
            plot_data['y_err'] = y_err
            plot_data['rho'] = rho

            print(f"Transformation complete:")
            print(f"  X range: {x.min():.4f} to {x.max():.4f}")
            print(f"  Y range: {y.min():.6f} to {y.max():.6f}")
            print(f"  X_err range: {x_err.min():.6f} to {x_err.max():.6f}")
            print(f"  Y_err range: {y_err.min():.6e} to {y_err.max():.6e}")
            print(f"  Rho range: {rho.min():.4f} to {rho.max():.4f}")

        if len(x) < 2:
            QMessageBox.warning(self, "Insufficient Data",
                f"Need at least 2 data points for regression. Found {len(x)} point(s).")
            self.figure.clear()
            self.canvas.draw()
            return

        # Store for access
        self.current_x_data = x
        self.current_y_data = y
        self.current_x_err = x_err
        self.current_y_err = y_err
        self.current_rho = rho
        self.current_groups = groups

        # Check if groups have changed - only repopulate intercept table if needed
        current_groups = set(selected_groups)
        table_groups = set()
        for row in range(self.intercept_table.rowCount):
            item = self.intercept_table.item(row, 0)
            if item:
                table_groups.add(item.text())

        # Only populate table if groups changed
        if current_groups != table_groups:
            print(f"Groups changed, repopulating table")
            self.populate_intercept_table()

        # Always read current values from table
        self.get_group_intercepts()

        # Create plot - check for multi-panel mode
        self.figure.clear()

        # Clear hover annotation and scatter data for new plot
        self.hover_annotation = None
        self.scatter_data = []

        # Store all results across panels
        self.all_results = []

        if self.multi_panel_check.isChecked():
            # Multi-panel mode: 2x2 grid
            axes = []
            for i in range(4):
                ax = self.figure.add_subplot(2, 2, i + 1)
                axes.append(ax)

            # Plot each panel with its assigned groups
            for panel_idx in range(4):
                panel_groups = self.get_groups_for_panel(panel_idx + 1)
                if panel_groups:
                    panel_results = self.plot_panel(axes[panel_idx], panel_groups, plot_data)
                    self.all_results.extend(panel_results)
                else:
                    # Empty panel - minimal labeling
                    axes[panel_idx].set_xlabel('⁸⁷Rb/⁸⁶Sr', fontsize=10)
                    axes[panel_idx].set_ylabel('⁸⁷Sr/⁸⁶Sr', fontsize=10)
                    axes[panel_idx].text(0.5, 0.5, "No groups selected",
                                         ha='center', va='center', transform=axes[panel_idx].transAxes,
                                         fontsize=10, color='gray')
        else:
            # Single panel mode
            ax = self.figure.add_subplot(111)
            panel_results = self.plot_panel(ax, selected_groups, plot_data)
            self.all_results = panel_results

        # Store last result for backward compatibility
        if self.all_results:
            last_result = self.all_results[-1]
            self.last_slope = last_result['slope']
            self.last_intercept = last_result['intercept']
            self.last_slope_err = last_result['slope_err']
            self.last_intercept_err = last_result['intercept_err']
            self.last_mswd = last_result['mswd']
            self.last_prob = last_result['prob']
            self.last_age = last_result['age']
            self.last_age_err = last_result['age_err']

        self.figure.tight_layout()
        self.canvas.draw()

        # Print completion message
        if self.all_results:
            last_result = self.all_results[-1]
            mswd_str = f"{last_result['mswd']:.2f}" if last_result['mswd'] is not None else "N/A"
            print(f"Regression complete: Age = {last_result['age']:.2f} ± {last_result['age_err']:.2f} Ma, MSWD = {mswd_str}, n = {last_result['n']}")

    def plot_panel(self, ax, groups_to_plot, plot_data):
        """Plot a single panel with the specified groups

        Returns list of regression results for the plotted groups
        """
        x = plot_data['x']
        y = plot_data['y']
        x_err = plot_data['x_err']
        y_err = plot_data['y_err']
        rho = plot_data['rho']
        groups = plot_data['groups']
        gradient = plot_data['gradient']
        labels = plot_data.get('labels', [])  # Get labels for hover

        # Filter to only the groups for this panel
        unique_groups = [g for g in groups_to_plot if g in set(groups)]
        if not unique_groups:
            return []

        # Set up colors for groups
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', 'd', '|', '_']
        group_color_map = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        group_marker_map = {group: markers[i % len(markers)] for i, group in enumerate(unique_groups)}

        # Get regression settings
        use_rho = self.use_rho_check.isChecked()

        # Set up gradient coloring if enabled
        # Filter gradient to only include data for groups in this panel
        use_gradient = self.use_gradient_check.isChecked()
        gradient_cmap = None
        gradient_norm = None
        if use_gradient:
            # Get gradient values only for groups in this panel
            panel_mask = np.array([g in unique_groups for g in groups])
            panel_gradient = gradient[panel_mask]

            if np.any(np.isfinite(panel_gradient)):
                cmap_name = self.cmap_combo.currentText
                gradient_cmap = plt.cm.get_cmap(cmap_name)
                # Normalize gradient values to [0, 1] based on this panel's data only
                valid_gradient = panel_gradient[np.isfinite(panel_gradient)]
                if len(valid_gradient) > 0:
                    gradient_min = np.min(valid_gradient)
                    gradient_max = np.max(valid_gradient)
                    if gradient_max > gradient_min:
                        gradient_norm = plt.Normalize(vmin=gradient_min, vmax=gradient_max)
                    else:
                        gradient_norm = plt.Normalize(vmin=gradient_min - 1, vmax=gradient_max + 1)
                else:
                    use_gradient = False
            else:
                use_gradient = False

        # Store results for this panel
        all_results = []

        # Track data extents including errors (for axis limits)
        all_x_min = []
        all_x_max_with_err = []
        all_y_min = []
        all_y_max_with_err = []

        # Store group data and regression results for two-pass plotting
        group_plot_data = []

        # =====================================================================
        # FIRST PASS: Gather data, run regressions, calculate extents
        # =====================================================================
        for group_name in unique_groups:
            # Get data for this group only
            group_mask = np.array([g == group_name for g in groups])
            x_group = x[group_mask]
            y_group = y[group_mask]
            x_err_group = x_err[group_mask]
            y_err_group = y_err[group_mask]
            rho_group = rho[group_mask] if use_rho else None
            gradient_group = gradient[group_mask] if use_gradient else None
            # Get labels for this group (for hover functionality)
            labels_group = [labels[i] for i in range(len(labels)) if group_mask[i]] if labels else []

            color = group_color_map[group_name]
            marker = group_marker_map[group_name]

            if len(x_group) == 0:
                continue

            # Track data extents (including errors for both bounds)
            all_x_min.append((x_group - x_err_group).min())
            all_x_max_with_err.append((x_group + x_err_group).max())
            all_y_min.append((y_group - y_err_group).min())
            all_y_max_with_err.append((y_group + y_err_group).max())

            if len(x_group) < 2:
                print(f"Skipping regression for {group_name}: only {len(x_group)} point(s)")
                group_plot_data.append({
                    'group_name': group_name,
                    'x_group': x_group,
                    'y_group': y_group,
                    'x_err_group': x_err_group,
                    'y_err_group': y_err_group,
                    'rho_group': rho_group,
                    'gradient_group': gradient_group,
                    'labels_group': labels_group,
                    'color': color,
                    'marker': marker,
                    'has_regression': False
                })
                continue

            # Perform regression for this group
            try:
                dof = None
                cov_ab = None

                if group_name in self.group_intercepts:
                    use_fixed_group, fixed_int_group, fixed_int_err_group = self.group_intercepts[group_name]
                else:
                    use_fixed_group = False
                    fixed_int_group = None
                    fixed_int_err_group = None

                # For inverse isochron with fixed intercept, transform the user-input value
                # User inputs (⁸⁷Sr/⁸⁶Sr)ᵢ but inverse isochron intercept is (⁸⁶Sr/⁸⁷Sr)ᵢ
                # So we need: fixed_int_inverse = 1 / fixed_int_conventional
                if self.inverse_isochron_check.isChecked() and use_fixed_group and fixed_int_group is not None:
                    if fixed_int_group != 0:
                        fixed_int_original = fixed_int_group
                        fixed_int_err_original = fixed_int_err_group

                        # Transform: (⁸⁶Sr/⁸⁷Sr)ᵢ = 1 / (⁸⁷Sr/⁸⁶Sr)ᵢ
                        fixed_int_group = 1.0 / fixed_int_original

                        # Error propagation: σ(1/x) = σ(x) / x²
                        if fixed_int_err_original is not None and fixed_int_err_original > 0:
                            fixed_int_err_group = fixed_int_err_original / (fixed_int_original ** 2)

                        print(f"  Inverse isochron: transforming fixed intercept")
                        print(f"    Input (⁸⁷Sr/⁸⁶Sr)ᵢ = {fixed_int_original:.6f} ± {fixed_int_err_original:.6f}")
                        print(f"    Transformed (⁸⁶Sr/⁸⁷Sr)ᵢ = {fixed_int_group:.6f} ± {fixed_int_err_group:.6f}")
                    else:
                        print(f"  Warning: Cannot transform fixed intercept of 0 for inverse isochron")
                        use_fixed_group = False

                model_index = self.model_combo.currentIndex

                if model_index == 0:
                    slope, intercept, slope_err, intercept_err, mswd, prob, n_points, dof, cov_ab = self.york_regression(
                        x_group, y_group, x_err_group, y_err_group, rho=rho_group,
                        fix_intercept=use_fixed_group, fixed_int=fixed_int_group, fixed_int_err=fixed_int_err_group
                    )
                    # If regression failed (NaN results), retry without rho
                    if np.isnan(intercept) and rho_group is not None and np.any(rho_group != 0):
                        print(f"  WARNING: York regression failed with rho, retrying without rho correlation")
                        slope, intercept, slope_err, intercept_err, mswd, prob, n_points, dof, cov_ab = self.york_regression(
                            x_group, y_group, x_err_group, y_err_group, rho=None,
                            fix_intercept=use_fixed_group, fixed_int=fixed_int_group, fixed_int_err=fixed_int_err_group
                        )
                        # Clear rho for ellipse plotting too
                        rho_group = np.zeros_like(rho_group)
                    dispersion = None

                elif model_index == 1:
                    slope, intercept, slope_err, intercept_err, mswd, prob, n_points, dof, cov_ab = self.tls_regression(
                        x_group, y_group,
                        fix_intercept=use_fixed_group, fixed_int=fixed_int_group, fixed_int_err=fixed_int_err_group
                    )
                    dispersion = None

                elif model_index == 2:
                    if use_fixed_group:
                        wtype = 'a'
                    else:
                        wtype = 'b' if self.age_dispersion_check.isChecked() else 'a'

                    slope, intercept, slope_err, intercept_err, mswd, prob, dispersion, dispersion_err = self.york_overdispersion_regression(
                        x_group, y_group, x_err_group, y_err_group, rho=rho_group,
                        fix_intercept=use_fixed_group, fixed_int=fixed_int_group, fixed_int_err=fixed_int_err_group,
                        wtype=wtype
                    )
                    # If regression failed (NaN results), retry without rho
                    if np.isnan(intercept) and rho_group is not None and np.any(rho_group != 0):
                        print(f"  WARNING: York overdispersion regression failed with rho, retrying without rho correlation")
                        slope, intercept, slope_err, intercept_err, mswd, prob, dispersion, dispersion_err = self.york_overdispersion_regression(
                            x_group, y_group, x_err_group, y_err_group, rho=None,
                            fix_intercept=use_fixed_group, fixed_int=fixed_int_group, fixed_int_err=fixed_int_err_group,
                            wtype=wtype
                        )
                        # Clear rho for ellipse plotting too
                        rho_group = np.zeros_like(rho_group)
                    x_bar = np.mean(x_group)
                    cov_ab = -x_bar * (slope_err / 2.0)**2 * 4.0

                if model_index in [0, 1]:
                    age, age_err = self.calculate_age(
                        slope, slope_err, mswd, prob, len(x_group), dof_override=dof,
                        is_model3=False, is_inverse=self.inverse_isochron_check.isChecked(),
                        intercept=intercept, intercept_err=intercept_err, cov_ab=cov_ab
                    )
                else:
                    if use_fixed_group:
                        dof_model3 = len(x_group) - 1
                    else:
                        dof_model3 = len(x_group) - 2
                    dof = dof_model3
                    age, age_err = self.calculate_age(
                        slope, slope_err, mswd, prob, len(x_group), dof_override=dof_model3,
                        is_model3=True, is_inverse=self.inverse_isochron_check.isChecked(),
                        intercept=intercept, intercept_err=intercept_err, cov_ab=cov_ab
                    )

                # For inverse isochron, calculate the conventional initial ratio from the intercept
                # Intercept = (⁸⁶Sr/⁸⁷Sr)ᵢ, so (⁸⁷Sr/⁸⁶Sr)ᵢ = 1/intercept
                if self.inverse_isochron_check.isChecked():
                    initial_ratio = 1.0 / intercept
                    # Error propagation: σ(1/x) = σ(x) / x²
                    initial_ratio_err = intercept_err / (intercept ** 2)
                else:
                    initial_ratio = intercept
                    initial_ratio_err = intercept_err

                result = {
                    'group': group_name,
                    'slope': slope,
                    'intercept': intercept,
                    'slope_err': slope_err,
                    'intercept_err': intercept_err,
                    'initial_ratio': initial_ratio,
                    'initial_ratio_err': initial_ratio_err,
                    'cov_ab': cov_ab,
                    'mswd': mswd,
                    'prob': prob,
                    'age': age,
                    'age_err': age_err,
                    'n': len(x_group),
                    'dof': dof,
                    'model': model_index,
                    'is_inverse': self.inverse_isochron_check.isChecked()
                }

                if model_index == 2:
                    result['dispersion'] = dispersion
                    result['dispersion_err'] = dispersion_err
                    result['wtype'] = 'b' if self.age_dispersion_check.isChecked() else 'a'

                all_results.append(result)

                group_plot_data.append({
                    'group_name': group_name,
                    'x_group': x_group,
                    'y_group': y_group,
                    'x_err_group': x_err_group,
                    'y_err_group': y_err_group,
                    'rho_group': rho_group,
                    'gradient_group': gradient_group,
                    'labels_group': labels_group,
                    'color': color,
                    'marker': marker,
                    'has_regression': True,
                    'slope': slope,
                    'intercept': intercept,
                    'slope_err': slope_err,
                    'intercept_err': intercept_err,
                    'cov_ab': cov_ab,
                    'dof': dof,
                    'mswd': mswd,
                    'prob': prob,
                    'model_index': model_index
                })

            except Exception as e:
                print(f"ERROR in regression for {group_name}: {e}")
                import traceback
                traceback.print_exc()
                group_plot_data.append({
                    'group_name': group_name,
                    'x_group': x_group,
                    'y_group': y_group,
                    'x_err_group': x_err_group,
                    'y_err_group': y_err_group,
                    'rho_group': rho_group,
                    'gradient_group': gradient_group,
                    'labels_group': labels_group,
                    'color': color,
                    'marker': marker,
                    'has_regression': False
                })
                continue

        # =====================================================================
        # CALCULATE FINAL AXIS LIMITS
        # =====================================================================
        x_plot_min, x_plot_max, y_plot_min, y_plot_max = None, None, None, None
        if all_x_min and all_x_max_with_err and all_y_min and all_y_max_with_err:
            x_data_min = min(all_x_min)
            x_data_max = max(all_x_max_with_err)
            y_data_max = max(all_y_max_with_err)

            # For y_data_min, consider both:
            # 1. The regression line y-value at x_data_min (to ensure line is visible)
            # 2. The actual data points minus their errors (to ensure ellipses are visible)
            y_at_xmin = []
            for gpd in group_plot_data:
                if gpd['has_regression']:
                    y_at_xmin.append(gpd['intercept'] + gpd['slope'] * x_data_min)

            # Take minimum of regression-based and data-based lower bounds
            y_data_min_from_data = min(all_y_min)
            if y_at_xmin:
                y_data_min_from_regression = min(y_at_xmin)
                y_data_min = min(y_data_min_from_data, y_data_min_from_regression)
            else:
                y_data_min = y_data_min_from_data

            x_range = x_data_max - x_data_min
            y_range = y_data_max - y_data_min

            x_plot_min = x_data_min - 0.05 * x_range
            x_plot_max = x_data_max + 0.10 * x_range
            y_plot_min = y_data_min - 0.05 * y_range
            y_plot_max = y_data_max + 0.10 * y_range

            ax.set_xlim(x_plot_min, x_plot_max)
            ax.set_ylim(y_plot_min, y_plot_max)

        # =====================================================================
        # SECOND PASS: Plot everything using final axis limits
        # =====================================================================
        for gpd in group_plot_data:
            group_name = gpd['group_name']
            x_group = gpd['x_group']
            y_group = gpd['y_group']
            x_err_group = gpd['x_err_group']
            y_err_group = gpd['y_err_group']
            rho_group = gpd['rho_group']
            gradient_group = gpd.get('gradient_group', None)
            labels_group = gpd.get('labels_group', [])
            color = gpd['color']
            marker = gpd['marker']

            # Plot error ellipses
            for i in range(len(x_group)):
                if x_err_group[i] > 0 and y_err_group[i] > 0:
                    if use_gradient and gradient_group is not None and np.isfinite(gradient_group[i]):
                        ellipse_color = gradient_cmap(gradient_norm(gradient_group[i]))
                    else:
                        ellipse_color = color

                    self.plot_error_ellipse(ax, x_group[i], y_group[i], x_err_group[i], y_err_group[i],
                                          rho_group[i] if rho_group is not None else 0.0,
                                          color=ellipse_color, alpha=0.4)

            # Store scatter data for hover functionality
            if len(x_group) > 0 and len(labels_group) > 0:
                self.scatter_data.append({
                    'ax': ax,
                    'x': np.array(x_group),
                    'y': np.array(y_group),
                    'labels': labels_group
                })

            if not gpd['has_regression']:
                continue

            slope = gpd['slope']
            intercept = gpd['intercept']
            slope_err = gpd['slope_err']
            intercept_err = gpd['intercept_err']
            cov_ab = gpd['cov_ab']
            dof = gpd['dof']
            mswd = gpd['mswd']
            prob = gpd['prob']
            model_index = gpd['model_index']

            # Plot confidence envelope
            if self.show_envelope_check.isChecked() and x_plot_min is not None:
                try:
                    x_for_envelope = np.array([x_plot_min, x_plot_max])
                    self.plot_confidence_envelope(
                        ax, x_for_envelope, slope, intercept, slope_err, intercept_err,
                        cov_ab, len(x_group), dof, mswd, prob,
                        color=color, alpha=0.15, model_index=model_index,
                        extend_low=0.0, extend_high=0.0
                    )
                except Exception as e:
                    print(f"ERROR plotting envelope for {group_name}: {e}")

            # Plot regression line
            if x_plot_min is not None and x_plot_max is not None:
                x_line = np.array([x_plot_min, x_plot_max])
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, '-', color=color, linewidth=1.0, alpha=0.7, label=group_name)

        # Add results text box (if enabled)
        if all_results and self.show_results_check.isChecked():
            results_lines = []
            for res in all_results:
                mswd_str = f"{res['mswd']:.2f}" if res['mswd'] is not None else "N/A"

                # Handle case where age is 0 or invalid
                if res['age'] == 0 or not np.isfinite(res['age']):
                    age_str = "Invalid"
                else:
                    age_str = f"{res['age']:.2f} ± {res['age_err']:.2f} Ma"

                if self.multi_panel_check.isChecked():
                    # Compact 3-line format for multi-panel mode
                    # Line 1: Age ± uncertainty
                    results_lines.append(f"{res['group']}: {age_str}")
                    # Line 2: Initial ratio with subscript i (using unicode subscript)
                    # Always display as (⁸⁷Sr/⁸⁶Sr)ᵢ regardless of isochron type
                    if np.isfinite(res['initial_ratio']):
                        results_lines.append(f"  (⁸⁷Sr/⁸⁶Sr)ᵢ: {res['initial_ratio']:.6f} ± {res['initial_ratio_err']:.6f}")
                    else:
                        results_lines.append(f"  (⁸⁷Sr/⁸⁶Sr)ᵢ: Invalid")
                    # Line 3: MSWD and n
                    results_lines.append(f"  MSWD={mswd_str}, n={res['n']}")
                else:
                    # Original format for single panel mode
                    results_lines.append(f"{res['group']}: {age_str} (MSWD={mswd_str}, n={res['n']})")
                    # Always display as (⁸⁷Sr/⁸⁶Sr)ᵢ regardless of isochron type
                    if np.isfinite(res['initial_ratio']):
                        results_lines.append(f"  (⁸⁷Sr/⁸⁶Sr)ᵢ: {res['initial_ratio']:.6f} ± {res['initial_ratio_err']:.6f}")
                    else:
                        results_lines.append(f"  (⁸⁷Sr/⁸⁶Sr)ᵢ: Invalid")
                    if 'dispersion' in res:
                        results_lines.append(f"  Dispersion: {res['dispersion']:.6f} ± {res['dispersion_err']:.6f}")
            results_text = '\n'.join(results_lines)

            # Smaller font for multi-panel mode
            font_size = 8 if self.multi_panel_check.isChecked() else 9
            ax.text(0.02, 0.98, results_text, transform=ax.transAxes,
                   fontsize=font_size, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.8', linewidth=0.5, alpha=0.8))

        # Labels and formatting
        font_size = 10 if self.multi_panel_check.isChecked() else 12
        if self.inverse_isochron_check.isChecked():
            ax.set_xlabel('⁸⁷Rb/⁸⁷Sr', fontsize=font_size)
            ax.set_ylabel('⁸⁶Sr/⁸⁷Sr', fontsize=font_size)
        else:
            ax.set_xlabel('⁸⁷Rb/⁸⁶Sr', fontsize=font_size)
            ax.set_ylabel('⁸⁷Sr/⁸⁶Sr', fontsize=font_size)

        # Only show title for single panel mode
        if not self.multi_panel_check.isChecked():
            if self.inverse_isochron_check.isChecked():
                ax.set_title('Rb-Sr Inverse Isochron', fontsize=14, fontweight='bold')
            else:
                ax.set_title('Rb-Sr Isochron', fontsize=14, fontweight='bold')

        # Limit number of ticks in multi-panel mode to avoid crowding
        if self.multi_panel_check.isChecked():
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

        # Add colorbar if gradient coloring is enabled
        if use_gradient and gradient_norm is not None:
            import matplotlib.cm as cm
            from matplotlib.ticker import ScalarFormatter
            sm = cm.ScalarMappable(cmap=gradient_cmap, norm=gradient_norm)
            sm.set_array([])
            if self.multi_panel_check.isChecked():
                # Smaller colorbar for multi-panel mode
                cbar = self.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
                cbar.set_label(self.gradient_channel_combo.currentText, fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                exponent_fontsize = 7
            else:
                cbar = self.figure.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label(self.gradient_channel_combo.currentText, fontsize=10)
                exponent_fontsize = 10

            # Use scientific notation with offset at top (e.g., "1e5" at top, single digits on ticks)
            cbar.formatter = ScalarFormatter(useMathText=True)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))  # Always use scientific notation
            cbar.update_ticks()

            # Left-align the exponent text with the colorbar and match font size
            cbar.ax.yaxis.get_offset_text().set_horizontalalignment('left')
            cbar.ax.yaxis.get_offset_text().set_x(0)
            cbar.ax.yaxis.get_offset_text().set_fontsize(exponent_fontsize)

        # Legend
        if len(unique_groups) > 0 and self.show_legend_check.isChecked():
            legend_fontsize = 7 if self.multi_panel_check.isChecked() else 9
            legend = ax.legend(loc='lower right', fontsize=legend_fontsize)
            legend.get_frame().set_linewidth(0.5)

        return all_results

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes is None:
            return

        ax = event.inaxes
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Zoom factor
        scale_factor = 1.1 if event.button == 'down' else 0.9

        # Zoom around mouse position
        x_data = event.xdata
        y_data = event.ydata

        new_x_min = x_data - (x_data - x_min) * scale_factor
        new_x_max = x_data + (x_max - x_data) * scale_factor
        new_y_min = y_data - (y_data - y_min) * scale_factor
        new_y_max = y_data + (y_max - y_data) * scale_factor

        ax.set_xlim([new_x_min, new_x_max])
        ax.set_ylim([new_y_min, new_y_max])
        self.canvas.draw()

    def on_hover(self, event):
        """Handle mouse hover to show data point labels"""
        if event.inaxes is None:
            # Mouse is outside axes, hide annotation
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        # Check if we have any scatter artists to check
        if not hasattr(self, 'scatter_data') or not self.scatter_data:
            return

        # Find the closest point across all scatter plots
        min_dist = float('inf')
        closest_label = None
        closest_x = None
        closest_y = None

        for scatter_info in self.scatter_data:
            ax = scatter_info['ax']
            if ax != event.inaxes:
                continue

            x_data = scatter_info['x']
            y_data = scatter_info['y']
            labels = scatter_info['labels']

            # Get axis limits to normalize distances
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            if x_range == 0 or y_range == 0:
                continue

            # Calculate normalized distances to all points
            for i in range(len(x_data)):
                dx = (x_data[i] - event.xdata) / x_range
                dy = (y_data[i] - event.ydata) / y_range
                dist = np.sqrt(dx**2 + dy**2)

                if dist < min_dist:
                    min_dist = dist
                    closest_label = labels[i]
                    closest_x = x_data[i]
                    closest_y = y_data[i]
                    closest_ax = ax

        # Show annotation if close enough to a point (within 5% of plot range)
        if min_dist < 0.05 and closest_label is not None:
            # Create or update annotation
            if self.hover_annotation is None or self.hover_annotation.axes != closest_ax:
                # Need to create new annotation for this axes
                if self.hover_annotation is not None:
                    self.hover_annotation.set_visible(False)
                self.hover_annotation = closest_ax.annotate(
                    closest_label,
                    xy=(closest_x, closest_y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E4DF', alpha=0.9, edgecolor='#B0A899'),
                    fontsize=8,
                    zorder=1000
                )
            else:
                self.hover_annotation.xy = (closest_x, closest_y)
                self.hover_annotation.set_text(closest_label)

            self.hover_annotation.set_visible(True)
            self.canvas.draw_idle()
        else:
            # Hide annotation if not close to any point
            if self.hover_annotation is not None and self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas.draw_idle()

    def plot_error_ellipse(self, ax, x, y, x_err, y_err, rho, color='blue', alpha=0.3):
        """
        Plot 2-sigma error ellipse

        Parameters:
        x, y: data point coordinates
        x_err, y_err: 2SE uncertainties from DRS
        rho: error correlation coefficient
        """
        # Convert 2SE to 1-sigma
        sigma_x = x_err / 2.0
        sigma_y = y_err / 2.0

        # Covariance matrix
        cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                       [rho * sigma_x * sigma_y, sigma_y**2]])

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Calculate ellipse parameters for 2-sigma (multiply by 2)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * 2 * np.sqrt(eigenvalues[0])   # 2 * 2-sigma
        height = 2 * 2 * np.sqrt(eigenvalues[1])  # 2 * 2-sigma

        # Create ellipse
        ellipse = Ellipse((x, y), width, height, angle=angle,
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=1)
        ax.add_patch(ellipse)

    def plot_confidence_envelope(self, ax, x_data, slope, intercept, slope_err, intercept_err,
                                  cov_ab, n, dof, mswd, prob, color='gray', alpha=0.2,
                                  model_index=0, extend_low=0.05, extend_high=0.10):
        """
        Plot 95% confidence envelope around regression line (like IsoplotR)

        The confidence envelope accounts for uncertainties in both slope and intercept,
        as well as their covariance. For Model 1 with overdispersion (MSWD > 1, p < 0.05),
        the envelope is scaled by sqrt(MSWD) to match IsoplotR behavior.

        Parameters:
        ax: matplotlib axis
        x_data: array of x data points (used to determine plot range)
        slope, intercept: regression line parameters
        slope_err, intercept_err: 2SE uncertainties
        cov_ab: covariance between intercept and slope (in 2SE scale, i.e. 4*cov_1se)
        n: number of data points
        dof: degrees of freedom
        mswd: Mean Square of Weighted Deviates
        prob: p-value for chi-squared test
        color: fill color for envelope
        alpha: transparency
        model_index: 0=Model 1 (York), 1=Model 2 (TLS), 2=Model 3 (overdispersion)
        extend_low: how much to extend x range below data (fraction)
        extend_high: how much to extend x range above data (fraction)
        """
        from scipy import stats

        # Extend x range: 5% on low side, 10% on high side
        x_min = x_data.min()
        x_max = x_data.max()
        x_range = x_max - x_min
        x_plot_min = x_min - extend_low * x_range
        x_plot_max = x_max + extend_high * x_range

        # Create dense x array for smooth envelope
        x_envelope = np.linspace(x_plot_min, x_plot_max, 200)

        # Convert 2SE to 1SE for calculations
        sigma_a = intercept_err / 2.0  # 1SE of intercept
        sigma_b = slope_err / 2.0      # 1SE of slope
        cov_ab_1se = cov_ab / 4.0      # Convert from 2SE scale to 1SE scale

        # Calculate variance of y at each x point
        # Var(y) = Var(a) + x²*Var(b) + 2*x*Cov(a,b)
        var_y = sigma_a**2 + x_envelope**2 * sigma_b**2 + 2 * x_envelope * cov_ab_1se

        # Standard error of y at each point
        se_y = np.sqrt(np.maximum(var_y, 0))  # Ensure non-negative

        # For Model 1: Apply MSWD scaling if overdispersed (p < 0.05)
        # This matches IsoplotR behavior
        if model_index == 0 and prob is not None and prob < 0.05 and mswd > 1:
            # Scale uncertainties by sqrt(MSWD)
            se_y = se_y * np.sqrt(mswd)
            print(f"  Envelope: Scaling by sqrt(MSWD)={np.sqrt(mswd):.3f} (overdispersed)")

        # Get t-value for 95% confidence interval
        # Use t-distribution with dof degrees of freedom
        if dof > 0:
            t_crit = stats.t.ppf(0.975, dof)  # 95% CI, two-tailed
        else:
            t_crit = 1.96  # Fallback to z-score

        # Calculate envelope bounds
        y_line = intercept + slope * x_envelope
        y_upper = y_line + t_crit * se_y
        y_lower = y_line - t_crit * se_y

        # Plot filled envelope
        ax.fill_between(x_envelope, y_lower, y_upper,
                        color=color, alpha=alpha,
                        linewidth=0, zorder=1)

        print(f"  Envelope: dof={dof}, t_crit={t_crit:.3f}, SE range=[{se_y.min():.6f}, {se_y.max():.6f}]")

    def save_plot_pdf(self):
        """Save the current plot as a PDF file"""
        if self.current_x_data is None:
            QtGui.QMessageBox.warning(self, "No Plot", "Please create a plot first before saving.")
            return

        # Open file dialog - iolite returns just filename, not tuple
        filename = QtGui.QFileDialog.getSaveFileName(
            self, "Save Plot as PDF", "", "PDF Files (*.pdf)"
        )

        if filename:
            if not filename.endswith('.pdf'):
                filename += '.pdf'

            try:
                self.figure.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
                QtGui.QMessageBox.information(self, "Success", f"Plot saved to:\n{filename}")
            except Exception as e:
                QtGui.QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_results_excel(self):
        """Export regression results and data to Excel file"""
        if self.current_x_data is None or self.last_age is None:
            QtGui.QMessageBox.warning(self, "No Results", "Please perform a regression first before exporting.")
            return

        # Open file dialog - iolite returns just filename, not tuple
        filename = QtGui.QFileDialog.getSaveFileName(
            self, "Export Results to Excel", "", "Excel Files (*.xlsx)"
        )

        if filename:
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'

            try:
                import pandas as pd

                # Create Excel writer
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:

                    # Sheet 1: Summary Results - one column per sample
                    if hasattr(self, 'all_results') and self.all_results:
                        # Create parameter list (without Slope rows)
                        param_list = [
                            'Age (Ma)',
                            'Age 2SE (Ma)',
                            'Initial ⁸⁷Sr/⁸⁶Sr',
                            'Initial ⁸⁷Sr/⁸⁶Sr 2SE',
                            'MSWD',
                            'Probability',
                            'Number of Points'
                        ]

                        # Check if any result has Model 3 (need to add dispersion rows)
                        has_model3 = any(result.get('model', 0) == 2 for result in self.all_results)
                        if has_model3:
                            param_list.extend([
                                'Dispersion',
                                'Dispersion 2SE',
                                'Dispersion Type'
                            ])

                        # Add model and intercept type rows
                        param_list.extend([
                            'Regression Model',
                            'Intercept Type'
                        ])

                        summary_data = {'Parameter': param_list}

                        # Add a column for each group
                        for result in self.all_results:
                            group_name = result['group']

                            # Get model type
                            model_index = result.get('model', 0)
                            model_names = {0: 'Model 1', 1: 'Model 2', 2: 'Model 3'}
                            model_name = model_names.get(model_index, 'Model 1')

                            # Determine intercept type for this group
                            if group_name in self.group_intercepts:
                                use_fixed, _, _ = self.group_intercepts[group_name]
                                intercept_type = 'Fixed' if use_fixed else 'Free'
                            else:
                                intercept_type = 'Free'

                            # Build column data (without slope)
                            column_data = [
                                result['age'],
                                result['age_err'],
                                result['initial_ratio'],
                                result['initial_ratio_err'],
                                result.get('mswd', 'N/A'),
                                result.get('prob', 'N/A'),
                                result['n']
                            ]

                            # Add dispersion data if Model 3 is used anywhere
                            if has_model3:
                                if model_index == 2:
                                    # This result has dispersion
                                    column_data.extend([
                                        result.get('dispersion', ''),
                                        result.get('dispersion_err', ''),
                                        f"wtype='{result.get('wtype', 'a')}'"
                                    ])
                                else:
                                    # This result doesn't have dispersion (Model 1 or 2)
                                    column_data.extend(['N/A', 'N/A', 'N/A'])

                            # Add model and intercept type
                            column_data.extend([
                                model_name,
                                intercept_type
                            ])

                            summary_data[group_name] = column_data
                    else:
                        # Fallback if no results (shouldn't happen)
                        summary_data = {
                            'Parameter': ['Age (Ma)', 'Age 2SE (Ma)', 'Initial ⁸⁷Sr/⁸⁶Sr', 'Initial ⁸⁷Sr/⁸⁶Sr 2SE',
                                        'MSWD', 'Probability', 'Number of Points', 'Regression Model', 'Intercept Type'],
                            'Value': [self.last_age, self.last_age_err, self.last_intercept, self.last_intercept_err,
                                    self.last_mswd, self.last_prob, len(self.current_x_data),
                                    f'Model {self.model_combo.currentIndex + 1}', 'Free']
                        }

                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)

                    # Sheet 2: Data Points
                    # Use appropriate column labels based on isochron type
                    if self.inverse_isochron_check.isChecked():
                        data_dict = {
                            'Group': self.current_groups,
                            '⁸⁷Rb/⁸⁷Sr': self.current_x_data,
                            '⁸⁷Rb/⁸⁷Sr 2SE': self.current_x_err,
                            '⁸⁶Sr/⁸⁷Sr': self.current_y_data,
                            '⁸⁶Sr/⁸⁷Sr 2SE': self.current_y_err,
                        }
                    else:
                        data_dict = {
                            'Group': self.current_groups,
                            '⁸⁷Rb/⁸⁶Sr': self.current_x_data,
                            '⁸⁷Rb/⁸⁶Sr 2SE': self.current_x_err,
                            '⁸⁷Sr/⁸⁶Sr': self.current_y_data,
                            '⁸⁷Sr/⁸⁶Sr 2SE': self.current_y_err,
                        }

                    if self.use_rho_check.isChecked() and self.current_rho is not None:
                        data_dict['Rho'] = self.current_rho

                    df_data = pd.DataFrame(data_dict)
                    df_data.to_excel(writer, sheet_name='Data', index=False)

                    # Sheet 3: Per-Group Intercept Settings
                    intercept_data = []
                    for group_name, (use_fixed, intercept, intercept_err) in self.group_intercepts.items():
                        intercept_data.append({
                            'Group': group_name,
                            'Use Fixed Intercept': 'Yes' if use_fixed else 'No',
                            'Intercept Value': intercept if use_fixed else 'N/A',
                            'Intercept 2SE': intercept_err if use_fixed else 'N/A'
                        })

                    if intercept_data:
                        df_intercepts = pd.DataFrame(intercept_data)
                        df_intercepts.to_excel(writer, sheet_name='Intercept Settings', index=False)

                    # Sheet 4: General Settings
                    # Adjust uncertainty channel display based on mode
                    use_individual = self.use_individual_check.isChecked()

                    if use_individual:
                        # For individual integrations, only show actual channels used (no dropdown selection)
                        settings_data = {
                            'Setting': [
                                'X-axis Channel',
                                'X-axis Error Channel (Actual)',
                                'Y-axis Channel',
                                'Y-axis Error Channel (Actual)',
                                'Rho Channel',
                                'Use Rho',
                                'Data Mode',
                                'Inverse Isochron',
                                'Selected Groups'
                            ],
                            'Value': [
                                self.x_channel_combo.currentText,
                                self.actual_x_err_channel if self.actual_x_err_channel else 'N/A',
                                self.y_channel_combo.currentText,
                                self.actual_y_err_channel if self.actual_y_err_channel else 'N/A',
                                self.rho_combo.currentText,
                                'Yes' if self.use_rho_check.isChecked() else 'No',
                                'Individual Integrations',
                                'Yes' if self.inverse_isochron_check.isChecked() else 'No',
                                ', '.join(self.get_selected_groups())
                            ]
                        }
                    else:
                        # For selection means, show dropdown selections
                        settings_data = {
                            'Setting': [
                                'X-axis Channel',
                                'X-axis Error Channel',
                                'Y-axis Channel',
                                'Y-axis Error Channel',
                                'Rho Channel',
                                'Use Rho',
                                'Data Mode',
                                'Inverse Isochron',
                                'Selected Groups'
                            ],
                            'Value': [
                                self.x_channel_combo.currentText,
                                self.x_err_combo.currentText,
                                self.y_channel_combo.currentText,
                                self.y_err_combo.currentText,
                                self.rho_combo.currentText,
                                'Yes' if self.use_rho_check.isChecked() else 'No',
                                'Selection Means',
                                'Yes' if self.inverse_isochron_check.isChecked() else 'No',
                                ', '.join(self.get_selected_groups())
                            ]
                        }

                    df_settings = pd.DataFrame(settings_data)
                    df_settings.to_excel(writer, sheet_name='Settings', index=False)

                QtGui.QMessageBox.information(self, "Success", f"Results exported to:\n{filename}")

            except ImportError:
                QtGui.QMessageBox.critical(self, "Error",
                    "pandas and openpyxl are required for Excel export.\n"
                    "Please install them: pip install pandas openpyxl")
            except Exception as e:
                QtGui.QMessageBox.critical(self, "Error", f"Failed to export results:\n{str(e)}")

    def save_settings_DISABLED(self):
        """Save current settings"""
        self.settings.setValue('x_channel', self.x_channel_combo.currentText)
        self.settings.setValue('y_channel', self.y_channel_combo.currentText)
        self.settings.setValue('x_err_channel', self.x_err_combo.currentText)
        self.settings.setValue('y_err_channel', self.y_err_combo.currentText)
        self.settings.setValue('rho_channel', self.rho_combo.currentText)
        self.settings.setValue('use_rho', self.use_rho_check.isChecked())

    def load_settings(self):
        """Load saved settings"""
        if self.settings.contains('use_rho'):
            self.use_rho_check.setChecked(self.settings.value('use_rho') == 'true')


# Global widget instance
widget = None


def createUIElements():
    """
    Creates user interface elements in the main window of iolite.
    Sets menu item Tools --> Rb-Sr Isochron Tool
    """
    action = QAction("Rb-Sr Isochron Tool", None)
    action.triggered.connect(createRbSrIsochron)
    ui.setAction(action)
    ui.setMenuName(['Tools', 'Rb-Sr Isochron Tool'])


def createRbSrIsochron():
    """Create and show the Rb-Sr Isochron widget"""
    global widget
    try:
        print("Creating Rb-Sr Isochron widget...")
        widget = RbSrIsochronWidget()
        widget.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        print("Widget created, showing...")
        widget.show()
        print("Widget shown successfully")
    except Exception as e:
        print(f"Error creating Rb-Sr Isochron widget: {e}")
        import traceback
        traceback.print_exc()
