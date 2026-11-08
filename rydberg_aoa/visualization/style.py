"""
Centralized plotting style constants for Rydberg AoA visualizations.

This module contains all styling parameters used across the visualization
functions to ensure consistency and easy maintenance.
"""

# ============================================================================
# FONT SIZES
# ============================================================================

# Title fonts
TITLE_FONT_SIZE_MAIN = 13
TITLE_FONT_SIZE_SUBPLOT = 12
TITLE_FONT_SIZE_INSET = 8

# Label fonts
AXIS_LABEL_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE_SMALL = 9

# Tick label fonts
TICK_LABEL_FONT_SIZE = 10
TICK_LABEL_FONT_SIZE_SMALL = 8
TICK_LABEL_FONT_SIZE_TINY = 7

# Legend fonts
LEGEND_FONT_SIZE = 9
LEGEND_FONT_SIZE_SMALL = 8

# Text annotation fonts
ANNOTATION_FONT_SIZE = 10
ANNOTATION_FONT_SIZE_SMALL = 8

# ============================================================================
# LEGEND PROPERTIES
# ============================================================================

# Legend positions
LEGEND_POSITION_UPPER_RIGHT = 'upper right'
LEGEND_POSITION_UPPER_LEFT = 'upper left'
LEGEND_POSITION_LOWER_RIGHT = 'lower right'
LEGEND_POSITION_LOWER_LEFT = 'lower left'

# Legend box properties
LEGEND_BORDER_PAD = 0.5
LEGEND_LABEL_SPACING = 0.3
LEGEND_HANDLE_LENGTH = 1.5
LEGEND_HANDLE_TEXT_PAD = 0.4
LEGEND_BORDER_AXES_PAD = 0.5

# ============================================================================
# COLORS
# ============================================================================

# Modern color palettes
COLOR_PALETTE_MODERN = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]

# Spectral colors for multiple elements
SPECTRAL_COLORS = [
    '#2ca02c',  # Green for spectra
    '#1f77b4',  # Blue for reference
    '#d62728',  # Red for emphasis
    '#ff7f0e',  # Orange for secondary
]

# ============================================================================
# LINE AND MARKER PROPERTIES
# ============================================================================

# Line widths
LINE_WIDTH_MAIN = 2.5
LINE_WIDTH_SECONDARY = 2.0
LINE_WIDTH_THIN = 1.5
LINE_WIDTH_VERY_THIN = 1.0

# Line alpha values
LINE_ALPHA_MAIN = 0.9
LINE_ALPHA_SECONDARY = 0.8
LINE_ALPHA_LIGHT = 0.6

# Marker properties
MARKER_SIZE_MAIN = 10
MARKER_SIZE_MEDIUM = 8
MARKER_SIZE_SMALL = 6
MARKER_SIZE_TINY = 4

MARKER_EDGE_WIDTH = 2

# ============================================================================
# LAYOUT PROPERTIES
# ============================================================================

# Figure margins
FIGURE_MARGIN_TOP = 0.95
FIGURE_MARGIN_BOTTOM = 0.1
FIGURE_MARGIN_LEFT = 0.08
FIGURE_MARGIN_RIGHT = 0.95

# Subplot spacing
SUBPLOT_WSPACE = None  # Use constrained_layout default
SUBPLOT_HSPACE = None  # Use constrained_layout default

# ============================================================================
# GRID AND BACKGROUND PROPERTIES
# ============================================================================

# Grid properties
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.8

# Background properties
AXIS_BACKGROUND_COLOR = 'white'
FIGURE_BACKGROUND_COLOR = 'white'

# ============================================================================
# TEXT PROPERTIES
# ============================================================================

# Text box properties for annotations
ANNOTATION_BOX_STYLE = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
ANNOTATION_BOX_STYLE_LIGHT = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)

# ============================================================================
# SPECIALIZED SETTINGS PER PLOT TYPE
# ============================================================================

# Sensor spectrum specific
SENSOR_SPECTRUM_INSET_POSITION = [0.05, 0.75, 0.15, 0.1]  # [x, y, width, height]
SENSOR_SPECTRUM_INSET_ZOOM_LINE_COLOR = 'gray'
SENSOR_SPECTRUM_INSET_TITLE_FONT_SIZE = 8
SENSOR_SPECTRUM_INSET_TICK_FONT_SIZE = 7

# AoA spectrum specific
AOA_SPECTRUM_PEAK_REGION_SHRINK_SIZE = 2
AOA_SPECTRUM_PEAK_REGION_ALPHA = 0.2
AOA_SPECTRUM_ERROR_ANNOTATION_POSITION_Y_HIGH_AOA = 0.85
AOA_SPECTRUM_ERROR_ANNOTATION_POSITION_Y_LOW_AOA = 0.15

# Time domain heterodyne specific
TIME_DOMAIN_MAX_X_POINTS = 100
TIME_DOMAIN_LEGEND_POSITION = (1.05, 1)
TIME_DOMAIN_LEGEND_ANCHOR = 'upper left'

# FFT validation specific
FFT_VALIDATION_SPECTRUM_FRACTION = 20  # Show 1/x of spectrum

# ============================================================================
# UNITS AND FORMATTING
# ============================================================================

# Energy unit labels
ENERGY_UNIT_MHZ = 'MHz'
ENERGY_UNIT_GHZ = 'GHz'

# Time unit labels
TIME_UNIT_MICROSECONDS = 'μs'
TIME_UNIT_NANOSECONDS = 'ns'

# Angle unit labels
ANGLE_UNIT_DEGREES = '°'

# Frequency unit labels
FREQUENCY_UNIT_MHZ = 'MHz'
FREQUENCY_UNIT_GHZ = 'GHz'
