import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker # Add this import
import warnings
warnings.filterwarnings('ignore')
import os # Add os import
import json # Add this import for manual JSON parsing
import math # Add math import for target visualization
import matplotlib.patches as patches

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QTabWidget, QComboBox, QDoubleSpinBox,
    QSlider, QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox, QGridLayout,
    QScrollArea, QSizePolicy, QSpacerItem, QTextEdit, QMessageBox, QHeaderView, QListWidgetItem,
    QFrame, QSpinBox # Added QSpinBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QTimer, QRect, QPoint, pyqtSignal # Added QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen, QColor # Added QImage, QPixmap, QPainter, QPen, QColor

# UNIFIED PHASE COLOR SCHEME - Use these colors consistently throughout the application
UNIFIED_PHASE_COLORS = {
    'preparing': '#E74C3C',   # Red
    'aiming': '#3498DB',      # Blue  
    'aftershot': '#2ECC71',   # Green
    'after_shot': '#2ECC71',  # Green (alternative naming)
    'up': '#F39C12',          # Orange
    'free': '#95A5A6',        # Gray
    'unknown': '#BDC3C7'      # Light Gray
}

# Phase colors as lists for charts (in the order: Preparing, Aiming, After_Shot)
PHASE_COLORS_LIST = ['#E74C3C', '#3498DB', '#2ECC71']
PHASE_COLORS_LIST_WITH_UP = ['#F39C12', '#E74C3C', '#3498DB', '#2ECC71']  # Up, Preparing, Aiming, After_Shot

# Helper function to convert hex colors to QColor
def hex_to_qcolor(hex_color):
    """Convert hex color string to QColor object"""
    color = QColor(hex_color)
    return color

# Phase colors as QColor objects for PyQt5 drawing
UNIFIED_PHASE_QCOLORS = {
    'preparing': hex_to_qcolor(UNIFIED_PHASE_COLORS['preparing']),
    'aiming': hex_to_qcolor(UNIFIED_PHASE_COLORS['aiming']),
    'aftershot': hex_to_qcolor(UNIFIED_PHASE_COLORS['aftershot']),
    'after_shot': hex_to_qcolor(UNIFIED_PHASE_COLORS['after_shot']),
    'up': hex_to_qcolor(UNIFIED_PHASE_COLORS['up']),
    'free': hex_to_qcolor(UNIFIED_PHASE_COLORS['free']),
    'unknown': hex_to_qcolor(UNIFIED_PHASE_COLORS['unknown'])
}

# Set style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(PHASE_COLORS_LIST)  # Use unified phase colors

# Define session color palette for multi-session visualization
SESSION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",  # Blue, Orange, Green, Red, Purple
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",  # Brown, Pink, Gray, Olive, Cyan
    "#a6cee3", "#fb9a99", "#fdbf6f", "#cab2d6", "#ff9896",  # Light Blue, Light Red, Light Orange, Light Purple, Light Pink
    "#f0027f", "#386cb0", "#7fc97f", "#beaed4", "#fdc086",  # Magenta, Dark Blue, Dark Green, Lavender, Peach
    "#ffff99", "#bf5b17", "#666666", "#fb8072", "#80b1d3"   # Yellow, Dark Orange, Dark Gray, Salmon, Sky Blue
]

# Define skeleton connections (based on MediaPipe pose landmarks)
POSE_CONNECTIONS = [
    (11, 12), (12, 24), (24, 23), (23, 11),
    (12, 14), (14, 16),
    (11, 13), (13, 15),
    (24, 26), (26, 28), (28, 32), (32, 30),
    (23, 25), (25, 27), (27, 31), (31, 29),
]

# Define required columns for skeleton data (assuming one landmark per row)
SKELETON_REQUIRED_COLUMNS = ['Frame_Number', 'Player_State', 'Landmark_ID', 'Landmark_X', 'Landmark_Y']
# Shot_Number is optional but will be included if available in the JSON


class DrawableDisplayWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 60)  # Width stays 160, height reduced to 60 (2x shorter)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; border: 1px solid gray;")
        
        self.shapes = []
        self.current_shape_drawing = None 
        self.drawing_enabled = False
        self.current_shape_type = "Line"
        self.current_color = QColor("green")
        self.pixmap_bg = None 

    def set_image(self, q_image):
        if q_image:
            self.pixmap_bg = QPixmap.fromImage(q_image) 
            self.redraw_canvas() 
        else:
            self.pixmap_bg = None
            self.clear() 
            self.setPixmap(QPixmap())

    def redraw_canvas(self):
        if not self.pixmap_bg:
            temp_pixmap = QPixmap(self.size())
            temp_pixmap.fill(Qt.black)
            painter = QPainter(temp_pixmap)
        else:
            temp_pixmap = self.pixmap_bg.copy()
            painter = QPainter(temp_pixmap)

        self.draw_shapes_on_painter(painter) 
        
        if self.drawing_enabled and self.current_shape_drawing:
            self._draw_single_shape(painter, self.current_shape_drawing)
            
        painter.end()
        self.setPixmap(temp_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event):
        if self.drawing_enabled and event.button() == Qt.LeftButton:
            scaled_pos = self.map_widget_to_pixmap(event.pos())
            if scaled_pos:
                self.current_shape_drawing = {
                    "type": self.current_shape_type,
                    "color": self.current_color,
                    "points": [scaled_pos]
                }
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_enabled and self.current_shape_drawing and (event.buttons() & Qt.LeftButton):
            scaled_pos = self.map_widget_to_pixmap(event.pos())
            if scaled_pos:
                self.current_shape_drawing["points"].append(scaled_pos)
                self.redraw_canvas() 
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_enabled and self.current_shape_drawing and event.button() == Qt.LeftButton:
            if len(self.current_shape_drawing["points"]) > 1 or \
               (self.current_shape_type == "Point" and len(self.current_shape_drawing["points"]) == 1): 
                self.shapes.append(self.current_shape_drawing)
            self.current_shape_drawing = None
            self.redraw_canvas() 
        super().mouseReleaseEvent(event)

    def map_widget_to_pixmap(self, widget_pos):
        if not self.pixmap() or self.pixmap().isNull() or not self.pixmap_bg: # Added check for pixmap_bg
            return widget_pos 

        pixmap_size = self.pixmap_bg.size() # Use pixmap_bg size for original dimensions
        widget_size = self.size()

        if pixmap_size.width() == 0 or pixmap_size.height() == 0: return widget_pos

        displayed_pixmap_size = pixmap_size.scaled(widget_size, Qt.KeepAspectRatio)
        
        dx = (widget_size.width() - displayed_pixmap_size.width()) / 2
        dy = (widget_size.height() - displayed_pixmap_size.height()) / 2

        if not (dx <= widget_pos.x() < dx + displayed_pixmap_size.width() and \
                dy <= widget_pos.y() < dy + displayed_pixmap_size.height()):
            return None 

        x = (widget_pos.x() - dx) * (pixmap_size.width() / displayed_pixmap_size.width())
        y = (widget_pos.y() - dy) * (pixmap_size.height() / displayed_pixmap_size.height())
        
        return QPoint(int(x), int(y))

    def clear_drawings(self):
        self.shapes = []
        self.redraw_canvas()

    def draw_shapes_on_painter(self, painter):
        for shape in self.shapes:
            self._draw_single_shape(painter, shape)

    def _draw_single_shape(self, painter, shape_data):
        painter.setPen(QPen(shape_data["color"], 2)) 
        points = shape_data["points"]
        if not points: return

        if shape_data["type"] == "Line" and len(points) >= 2:
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i+1])
        elif shape_data["type"] == "Rectangle" and len(points) >= 2:
            start_point = points[0]
            end_point = points[-1] 
            rect = QRect(start_point, end_point).normalized()
            painter.drawRect(rect)

    def set_drawing_tool(self, tool_name):
        self.current_shape_type = tool_name

    def set_drawing_color(self, color_name):
        self.current_color = QColor(color_name)

    def toggle_drawing(self, enabled):
        self.drawing_enabled = enabled
        if not enabled:
            self.current_shape_drawing = None 
            self.redraw_canvas()


class VideoControls(QFrame): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: #f0f0f0; border-radius: 4px; padding: 5px; }
            QPushButton { min-width: 35px; padding: 5px; } QLabel { color: black; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5,5,5,5)

        # Create dual_phase_bar_widget FIRST
        self.dual_phase_bar_widget = DualPhaseBarWidget()
        self.dual_phase_bar_widget.set_phase_clicked_callback(self.jump_to_phase_frame)

        # Add labels for clarity
        self.primary_slider_label = QLabel("Primary Shot")
        self.compare_slider_label = QLabel("Comparison Shot")
        self.primary_slider_label.setVisible(True)
        self.compare_slider_label.setVisible(False)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0,0)
        self.compare_progress_slider = QSlider(Qt.Horizontal)
        self.compare_progress_slider.setRange(0, 0)
        self.compare_progress_slider.setVisible(False)
        self.compare_progress_slider.sliderMoved.connect(self.on_compare_slider_moved)
        self.compare_progress_slider.valueChanged.connect(self.on_compare_slider_value_changed)

        # Phase bars and sliders layout
        self.phase_and_slider_layout = QVBoxLayout()
        self.phase_and_slider_layout.setSpacing(2)
        self.phase_and_slider_layout.setContentsMargins(0, 0, 0, 0)
        self.phase_and_slider_layout.addWidget(self.dual_phase_bar_widget)
        self.phase_and_slider_layout.addWidget(self.primary_slider_label)
        self.phase_and_slider_layout.addWidget(self.progress_slider)
        self.phase_and_slider_layout.addWidget(self.compare_slider_label)
        self.phase_and_slider_layout.addWidget(self.compare_progress_slider)

        controls_layout = QHBoxLayout()
        self.play_pause_btn = QPushButton("▶") 
        self.play_pause_btn.setCheckable(True)
        self.stop_btn = QPushButton("⏹")
        self.backward_btn = QPushButton("⏪")
        self.forward_btn = QPushButton("⏩")
        
        for btn in [self.backward_btn, self.play_pause_btn, self.stop_btn, self.forward_btn]:
            controls_layout.addWidget(btn)
            btn.setFixedWidth(40)
        
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 300)  
        self.speed_slider.setValue(100)    
        self.speed_value_label = QLabel("1.0x") 
        self.speed_value_label.setFixedWidth(40)
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        
        progress_layout = QHBoxLayout()
        self.time_label = QLabel("Frame: 0 / 0") 
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.time_label)
        
        layout.addLayout(self.phase_and_slider_layout)
        layout.addLayout(controls_layout)
        layout.addLayout(speed_layout)
        layout.addLayout(progress_layout)

        self.speed_slider.valueChanged.connect(self._update_speed_label)
        
        # Phase markers data (for compatibility, not used in new widget)
        self.phase_markers = []
        self.phase_colors = UNIFIED_PHASE_COLORS

        # --- In VideoControls ---
        # In __init__, after self.progress_slider:
        self.compare_progress_slider = QSlider(Qt.Horizontal)
        self.compare_progress_slider.setRange(0, 0)
        self.compare_progress_slider.setVisible(False)
        self.compare_progress_slider.sliderMoved.connect(self.on_compare_slider_moved)
        self.compare_progress_slider.valueChanged.connect(self.on_compare_slider_value_changed)

        # Add labels for clarity
        self.primary_slider_label = QLabel("Primary Shot")
        self.compare_slider_label = QLabel("Comparison Shot")
        self.primary_slider_label.setVisible(True)
        self.compare_slider_label.setVisible(False)

        # Add to layout (after phase bars)
        layout.addWidget(self.primary_slider_label)
        layout.addWidget(self.progress_slider)
        layout.addWidget(self.compare_slider_label)
        layout.addWidget(self.compare_progress_slider)

    def _update_speed_label(self, value):
        speed = value / 100.0
        self.speed_value_label.setText(f"{speed:.1f}x")

    def set_total_frames(self, total_frames):
        self.progress_slider.setRange(0, max(0, total_frames -1 if total_frames > 0 else 0))
        self.update_frame_display(0, total_frames)

    def update_frame_display(self, current_frame_idx, total_frames):
        display_frame = current_frame_idx + 1 if total_frames > 0 else 0
        self.time_label.setText(f"Frame: {display_frame} / {total_frames}") 
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(current_frame_idx)

    def set_phase_markers(self, phase_data, total_frames=None):
        """Set phase markers using the custom PhaseMarkerBar widget"""
        if total_frames is None:
            total_frames = self.progress_slider.maximum() + 1
        self.dual_phase_bar_widget.set_phases(phase_data, total_frames)

    def set_dual_phase_markers(self, primary_phases, primary_total_frames, compare_phases=None, compare_total_frames=None):
        self.dual_phase_bar_widget.set_phases(primary_phases, primary_total_frames, compare_phases, compare_total_frames)

    def jump_to_phase_frame(self, frame_number):
        """Jump to a specific frame when a phase marker is clicked"""
        if 0 <= frame_number <= self.progress_slider.maximum():
            self.progress_slider.setValue(frame_number)
            # Emit the valueChanged signal to notify parent
            self.progress_slider.valueChanged.emit(frame_number)

    # --- Add methods to VideoControls ---
    def set_compare_slider_visible(self, visible, total_frames=0):
        self.compare_progress_slider.setVisible(visible)
        self.compare_slider_label.setVisible(visible)
        if visible:
            self.compare_progress_slider.setRange(0, max(0, total_frames-1 if total_frames > 0 else 0))
            self.compare_progress_slider.setValue(0)
        else:
            self.compare_progress_slider.setRange(0, 0)
            self.compare_progress_slider.setValue(0)

    def on_compare_slider_moved(self, frame_index):
        # Synchronize both sliders by relative position
        if self.compare_progress_slider.maximum() > 0:
            rel = frame_index / max(1, self.compare_progress_slider.maximum())
            main_max = self.progress_slider.maximum()
            main_idx = int(round(rel * main_max))
            self.progress_slider.setValue(main_idx)

    def on_compare_slider_value_changed(self, frame_index):
        # Only update if not dragging main slider
        if not self.progress_slider.isSliderDown():
            self.on_compare_slider_moved(frame_index)


class PlotlyWebView(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.html_path = None
        self.setMinimumSize(400, 300) 

    def set_figure(self, fig):
        # Create a unique temporary HTML file for each figure
        self.html_path = os.path.join(os.getcwd(), f"temp_plotly_fig_{id(self)}.html")
        try:
            pio.write_html(fig, self.html_path, include_plotlyjs='cdn')
            self.setUrl(QUrl.fromLocalFile(self.html_path))
        except Exception as e:
            print(f"Error writing or loading Plotly HTML: {e}")
            self.setHtml(f"<html><body><h1>Error generating plot</h1><p>{e}</p></body></html>")

    def __del__(self):
        if self.html_path and os.path.exists(self.html_path):
            try:
                os.remove(self.html_path)
            except OSError as e:
                # This can sometimes fail if the browser process still has a lock
                print(f"Error removing temporary file {self.html_path}: {e}")
        super().__del__()
class ShotTimingVisualizer:
    def __init__(self, data):
        self.df = data
        self.time_phases = ['Up_Time(s)', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
        
    def _get_session_colors(self):
        """Get color mapping for sessions"""
        if 'Session' not in self.df.columns:
            return {}
        
        sessions = self.df['Session'].unique()
        session_colors = {}
        for i, session in enumerate(sessions):
            session_colors[session] = SESSION_COLORS[i % len(SESSION_COLORS)]
        return session_colors
    
    def _get_shot_colors(self, use_session_colors=True):
        """Get colors for shots - either session-based or score-based"""
        if not use_session_colors or 'Session' not in self.df.columns:
            # Fall back to score-based colors
            return None
        
        session_colors = self._get_session_colors()
        shot_colors = []
        
        for _, row in self.df.iterrows():
            session = row.get('Session', 'Unknown')
            shot_colors.append(session_colors.get(session, '#666666'))
        
        return shot_colors
    
    def _get_legend_type(self):
        """Determine legend type based on data"""
        if 'Session' in self.df.columns and len(self.df['Session'].unique()) > 1:
            return 'session'  # Multiple sessions - show session colors
        else:
            return 'score'    # Single session - show score rings
        
    def small_multiples_line_charts(self, figsize=(12, 8), scaling_settings=None):
        """1. Small-Multiples Line Charts for Player × Session combinations"""
        players = self.df['Player'].unique()
        sessions = self.df['Session'].unique()
        
        # Find valid combinations (those that have data)
        valid_combinations = []
        for player in players:
            for session in sessions:
                subset = self.df[(self.df['Player'] == player) & (self.df['Session'] == session)]
                if not subset.empty:
                    valid_combinations.append((player, session))
        
        if not valid_combinations:
            # No valid data combinations
            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for line charts", ha='center', va='center', fontsize=12)
            fig.suptitle('Shot Timing Phases Across Sessions by Player', fontsize=14, fontweight='bold')
            fig.tight_layout()
            return fig
        
        # Calculate optimal grid layout
        n_combinations = len(valid_combinations)
        
        # Determine grid dimensions
        if n_combinations <= 2:
            n_cols = n_combinations
            n_rows = 1
        elif n_combinations <= 4:
            n_cols = 2
            n_rows = 2
        elif n_combinations <= 6:
            n_cols = 3
            n_rows = 2
        elif n_combinations <= 9:
            n_cols = 3
            n_rows = 3
        else:
            n_cols = 4
            n_rows = (n_combinations + 3) // 4  # Ceiling division
        
        fig = Figure(figsize=figsize)
        
        # Get session information for title
        num_sessions = len(sessions)
        title_text = 'Shot Timing Phases Across Sessions by Player'
        if num_sessions > 1:
            title_text += f' ({num_sessions} Sessions)'
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
        
        colors = PHASE_COLORS_LIST_WITH_UP
        
        # Create subplots only for valid combinations
        for idx, (player, session) in enumerate(valid_combinations):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            subset = self.df[(self.df['Player'] == player) & (self.df['Session'] == session)]
            
            # Plot data for each phase
            for k, phase in enumerate(self.time_phases):
                if phase in subset.columns and 'Shot_Number' in subset.columns:
                    ax.plot(subset['Shot_Number'], subset[phase], 
                           marker='o', linewidth=2, markersize=3,
                           label=phase.replace('_Time(s)', ''), color=colors[k % len(colors)])
            
            ax.set_title(f'{player} - {session}', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('Shot Number')
            ax.set_ylabel('Time (seconds)')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            # Improve y-axis formatting
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}s'))
        
        # Add legend
        handles, labels = [], []
        # Get handles and labels from the first subplot that has data
        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                # Ensure unique labels for the legend
                temp_handles, temp_labels = [], []
                seen_labels = set()
                for handle, label_text in zip(h, l):
                    if label_text not in seen_labels:
                        temp_handles.append(handle)
                        temp_labels.append(label_text)
                        seen_labels.add(label_text)
                handles.extend(temp_handles)
                labels.extend(temp_labels)
                break
        
        if labels:
            fig.legend(handles, labels, title='Metric', loc='upper right', bbox_to_anchor=(0.98, 0.95))
        
        fig.tight_layout()
        return fig
    
    def stacked_bars_shot_composition(self, figsize=(12, 6), scaling_settings=None):
        """Create stacked bar chart showing shot composition with improved scalability"""
        import numpy as np
        
        # Apply scaling settings
        if scaling_settings:
            figsize = scaling_settings['figsize']
            fontsize = scaling_settings['fontsize']
            auto_scale = scaling_settings['auto_scale']
            compact_mode = scaling_settings['compact_mode']
        else:
            fontsize = 10
            auto_scale = True
            compact_mode = False
        
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        df = self.df.copy()
        if df.empty:
            ax.text(0.5, 0.5, "No data available for stacked bar chart", ha='center', va='center', fontsize=fontsize)
            fig.tight_layout()
            return fig
        
        # Get unique shots for per-shot analysis
        if 'Shot_Number' in df.columns:
            shots = sorted(df['Shot_Number'].unique())
        else:
            shots = list(range(1, len(df) + 1))
        
        # Auto-scale for multiple shots
        if auto_scale and len(shots) > 10:
            # Adjust figure width based on number of shots
            width = min(24, 8 + len(shots) * 0.8)  # Max 24 inches
            fig.set_size_inches(width, figsize[1])
        
        # Prepare data for stacked bars - exclude Up_Time for cleaner visualization
        phases = ['Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
        phase_labels = ['Preparing', 'Aiming', 'After Shot']
        
        # Calculate total duration for each shot
        shot_data = []
        shot_numbers = []
        
        for shot in shots:
            if 'Shot_Number' in df.columns:
                shot_df = df[df['Shot_Number'] == shot]
            else:
                shot_df = df.iloc[shot-1:shot] if shot <= len(df) else pd.DataFrame()
            
            if not shot_df.empty:
                # Check if all required phase columns exist
                available_phases = [phase for phase in phases if phase in shot_df.columns]
                if available_phases:
                    total_duration = shot_df[available_phases].sum().sum()
                    if total_duration > 0:  # Only include shots with data
                        shot_data.append(shot_df)
                        shot_numbers.append(shot)
        
        if not shot_data:
            ax.text(0.5, 0.5, "No valid data for stacked bar chart", ha='center', va='center', fontsize=fontsize)
            fig.tight_layout()
            return fig
        
        # Create stacked bars
        x_pos = np.arange(len(shot_numbers))
        bar_width = 0.6 if compact_mode else 0.8
        
        # Phase colors for better visibility - use the list without UP phase
        phase_colors = PHASE_COLORS_LIST  # ['#E74C3C', '#3498DB', '#2ECC71']
        
        # Create stacked bars - per shot visualization
        bottom = np.zeros(len(shot_numbers))
        
        for i, phase in enumerate(phases):
            values = []
            for shot_df in shot_data:
                if phase in shot_df.columns:
                    total = shot_df[phase].sum()
                else:
                    total = 0
                values.append(total)
            
            # Create the stacked bar segment
            bars = ax.bar(x_pos, values, bottom=bottom, label=phase_labels[i], 
                         color=phase_colors[i % len(phase_colors)], width=bar_width, alpha=0.8, edgecolor='white', linewidth=0.5)
            bottom += np.array(values)
        
        # Customize the plot
        ax.set_xlabel('Shot Number', fontsize=fontsize)
        ax.set_ylabel('Total Time (seconds)', fontsize=fontsize)
        title_text = 'Shot Timing Breakdown - Phase Analysis per Shot'
        if len(shot_numbers) > 1:
            title_text += f' ({len(shot_numbers)} Shots)'
        ax.set_title(title_text, fontsize=fontsize, fontweight='bold', pad=20)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(shot_numbers, fontsize=max(8, fontsize-2), rotation=45 if len(shot_numbers) > 10 else 0)
        
        # Add legend with better positioning
        ax.legend(loc='upper right', fontsize=max(8, fontsize-2), framealpha=0.9)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Enhanced analysis annotations for detecting inefficient routines per shot
        # Calculate phase averages and identify potential issues
        phase_averages = []
        efficiency_notes = []
        
        for i, shot_df in enumerate(shot_data):
            # Calculate total time for this shot
            total = 0
            for phase in phases:
                if phase in shot_df.columns:
                    total += shot_df[phase].sum()
            
            # Display total time on top of bar
            ax.text(i, total + total * 0.02, f'{total:.1f}s', ha='center', va='bottom', 
                   fontsize=max(8, fontsize-2), fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Calculate phase percentages for this shot
            phase_pcts = []
            phase_times = []
            for phase in phases:
                if phase in shot_df.columns:
                    phase_time = shot_df[phase].sum()
                else:
                    phase_time = 0
                phase_pct = (phase_time / total * 100) if total > 0 else 0
                phase_pcts.append(phase_pct)
                phase_times.append(phase_time)
            
            phase_averages.append(phase_pcts)
            
            # Identify potential inefficiencies (without Up time)
            prep_pct, aim_pct, after_pct = phase_pcts
            notes = []
            
            # Flag potential issues
            if prep_pct > 40:  # Preparing takes too long
                notes.append("Long Prep")
            if aim_pct < 30:  # Aiming too short (adjusted threshold without up time)
                notes.append("Short Aim")
            if after_pct > 25:  # After shot too long
                notes.append("Long After")
            
            # Check for imbalanced routines
            if prep_pct > aim_pct * 1.5:  # Prep more than 1.5x aim time
                notes.append("Imbalanced")
            
            efficiency_notes.append(notes)
            
            # Add efficiency indicators below each bar
            if notes:
                note_text = ", ".join(notes)
                ax.text(i, -total * 0.05, note_text, ha='center', va='top', 
                       fontsize=max(6, fontsize-4), style='italic', color='red',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
        
        # Add percentage labels within each bar segment for better analysis
        for i, shot_df in enumerate(shot_data):
            cumulative = 0
            total = 0
            for phase in phases:
                if phase in shot_df.columns:
                    total += shot_df[phase].sum()
            
            for j, phase in enumerate(phases):
                if phase in shot_df.columns:
                    phase_value = shot_df[phase].sum()
                else:
                    phase_value = 0
                phase_pct = (phase_value / total * 100) if total > 0 else 0
                
                # Only show percentage if segment is large enough (>8%)
                if phase_pct > 8:
                    y_pos = cumulative + phase_value / 2
                    ax.text(i, y_pos, f'{phase_pct:.0f}%', ha='center', va='center',
                           fontsize=max(6, fontsize-4), fontweight='bold', 
                           color='white' if phase_pct > 15 else 'black')
                
                cumulative += phase_value
        
        # Add target efficiency guidelines as horizontal reference lines
        if len(shot_data) > 0:
            max_total = max([sum(shot_df[phase].sum() for phase in phases if phase in shot_df.columns) for shot_df in shot_data])
            
            # Add reference lines for ideal phase distribution (without up time)
            ideal_prep = max_total * 0.35  # 35% preparing (adjusted without up time)
            ideal_aim = max_total * 0.55   # 55% aiming (adjusted without up time)
            ax.axhline(y=ideal_prep, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=ideal_prep + ideal_aim, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add text annotations for reference lines
            ax.text(len(shot_data) - 0.3, ideal_prep, 'Target: 35% Prep', 
                   fontsize=max(6, fontsize-4), color='orange', alpha=0.7)
            ax.text(len(shot_data) - 0.3, ideal_prep + ideal_aim, 'Target: 90% Prep+Aim', 
                   fontsize=max(6, fontsize-4), color='blue', alpha=0.7)
        
        # Improve y-axis formatting
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0f}s'))
        
        # Add some padding to the y-axis to accommodate efficiency notes
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        ax.set_ylim(y_min - y_max * 0.15, y_max * 1.1)  # Add space below for notes
        
        # Enhanced title with efficiency analysis (without up time)
        if phase_averages:
            avg_phases = np.mean(phase_averages, axis=0)
            title_text += f'\nAvg Distribution: Prep {avg_phases[0]:.0f}% | Aim {avg_phases[1]:.0f}% | After {avg_phases[2]:.0f}%'
        
        ax.set_title(title_text, fontsize=fontsize, fontweight='bold', pad=30)
        
        fig.tight_layout()
        return fig
    
    def box_violin_plots(self, figsize=(14, 10), scaling_settings=None):
        """Create box and violin plots with improved scalability"""
        import numpy as np
        
        # Apply scaling settings
        if scaling_settings:
            figsize = scaling_settings['figsize']
            fontsize = scaling_settings['fontsize']
            auto_scale = scaling_settings['auto_scale']
            compact_mode = scaling_settings['compact_mode']
        else:
            fontsize = 10
            auto_scale = True
            compact_mode = False
        
        fig = Figure(figsize=figsize)
        
        df = self.df.copy()
        if df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for box/violin plots", ha='center', va='center', fontsize=fontsize)
            fig.tight_layout()
            return fig
        
        # Get unique sessions
        sessions = df['Session'].unique() if 'Session' in df.columns else ['Session']
        
        # Auto-scale for multiple sessions
        if auto_scale and len(sessions) > 3:
            # Adjust figure size based on number of sessions
            width = min(24, 8 + len(sessions) * 2)  # Max 24 inches
            height = min(16, 6 + len(sessions) * 0.5)  # Max 16 inches
            fig.set_size_inches(width, height)
        
        # Create subplots for each session - Always prefer side-by-side layout
        n_sessions = len(sessions)
        if n_sessions <= 5:
            # Side by side for up to 5 sessions
            n_cols = n_sessions
            n_rows = 1
        else:
            # Grid layout for many sessions
            n_cols = min(5, n_sessions)  # Max 5 columns
            n_rows = (n_sessions + n_cols - 1) // n_cols
        
        # Compact mode for many sessions
        if compact_mode:
            fontsize = max(6, fontsize - 2)  # Smaller fonts
        
        phases = ['Up_Time(s)', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
        
        # Use session colors if multiple sessions, otherwise use phase colors
        if len(sessions) > 1 and 'Session' in df.columns:
            colors = SESSION_COLORS[:len(sessions)]
        else:
            colors = PHASE_COLORS_LIST_WITH_UP
        
        for i, session in enumerate(sessions):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Filter data for this session
            session_data = df[df['Session'] == session] if 'Session' in df.columns else df
            
            if session_data.empty:
                ax.text(0.5, 0.5, f"No data for {session}", ha='center', va='center', fontsize=fontsize)
                ax.set_title(f"{session}", fontsize=fontsize)
                continue
            
            # Prepare data for plotting
            plot_data = []
            labels = []
            
            for phase in phases:
                if phase in session_data.columns:
                    values = session_data[phase].dropna()
                    if not values.empty:
                        plot_data.append(values)
                        labels.append(phase.replace('_Time(s)', ''))
            
            if not plot_data:
                ax.text(0.5, 0.5, f"No valid data for {session}", ha='center', va='center', fontsize=fontsize)
                ax.set_title(f"{session}", fontsize=fontsize)
                continue
            
            # Create box plots
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color the boxes with better colors
            phase_colors = PHASE_COLORS_LIST
            
            if len(sessions) > 1 and 'Session' in df.columns:
                # Use session color for all boxes in this session
                session_color = colors[i % len(colors)]
                for patch in bp['boxes']:
                    patch.set_facecolor(session_color)
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('white')
                    patch.set_linewidth(1)
            else:
                # Use phase colors
                for patch, color in zip(bp['boxes'], phase_colors[:len(plot_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('white')
                    patch.set_linewidth(1)
            
            # Customize the plot
            ax.set_ylabel('Time (seconds)', fontsize=fontsize)
            title_text = f"{session}"
            if len(sessions) > 1 and 'Session' in df.columns:
                title_text += f" (Session {i+1})"
            ax.set_title(title_text, fontsize=fontsize, pad=10)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Improve y-axis formatting
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}s'))
            
            # Rotate x-axis labels if needed
            if len(labels) > 4:
                ax.tick_params(axis='x', rotation=45, labelsize=fontsize-2)
            else:
                ax.tick_params(labelsize=fontsize-2)
        
        fig.tight_layout()
        return fig
    
    def heatmap_phase_durations(self, figsize=(12, 8), scaling_settings=None):
        """Create heatmap of phase durations across sessions with improved scalability"""
        import numpy as np
        
        # Apply scaling settings
        if scaling_settings:
            figsize = scaling_settings['figsize']
            fontsize = scaling_settings['fontsize']
            auto_scale = scaling_settings['auto_scale']
            compact_mode = scaling_settings['compact_mode']
        else:
            fontsize = 10
            auto_scale = True
            compact_mode = False
        
        fig = Figure(figsize=figsize)
        
        df = self.df.copy()
        if df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for heatmap", ha='center', va='center', fontsize=fontsize)
            fig.tight_layout()
            return fig
        
        # Get unique sessions and phases
        sessions = df['Session'].unique() if 'Session' in df.columns else ['Session']
        phases = ['Up_Time(s)', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
        
        # Auto-scale for multiple sessions
        if auto_scale and len(sessions) > 3:
            # Adjust figure size based on number of sessions
            width = min(24, 8 + len(sessions) * 2)  # Max 24 inches
            height = min(16, 6 + len(sessions) * 0.5)  # Max 16 inches
            fig.set_size_inches(width, height)
        
        # Create subplots for each session
        n_sessions = len(sessions)
        if n_sessions <= 3:
            # Side by side for few sessions
            n_cols = n_sessions
            n_rows = 1
        else:
            # Grid layout for many sessions
            n_cols = min(5, n_sessions)  # Max 5 columns
            n_rows = (n_sessions + n_cols - 1) // n_cols
        
        # Compact mode for many shots
        if compact_mode:
            fontsize = max(6, fontsize - 2)  # Smaller fonts
            cell_text_size = max(4, fontsize - 4)
        else:
            cell_text_size = max(6, fontsize - 2)
        
        for i, session in enumerate(sessions):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Filter data for this session
            session_data = df[df['Session'] == session] if 'Session' in df.columns else df
            
            if session_data.empty:
                ax.text(0.5, 0.5, f"No data for {session}", ha='center', va='center', fontsize=fontsize)
                ax.set_title(f"{session}", fontsize=fontsize)
                continue
            
            # Prepare heatmap data
            heatmap_data = []
            shot_numbers = []
            
            for _, row in session_data.iterrows():
                shot_num = row.get('Shot_Number', 'Unknown')
                shot_numbers.append(str(shot_num))
                
                # Get phase durations
                phase_values = []
                for phase in phases:
                    value = row.get(phase, 0)
                    try:
                        phase_values.append(float(value))
                    except (ValueError, TypeError):
                        phase_values.append(0)
                
                heatmap_data.append(phase_values)
            
            if not heatmap_data:
                ax.text(0.5, 0.5, f"No valid data for {session}", ha='center', va='center', fontsize=fontsize)
                ax.set_title(f"{session}", fontsize=fontsize)
                continue
            
            # Create heatmap
            heatmap_data = np.array(heatmap_data).T  # Transpose for correct orientation
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(shot_numbers)))
            ax.set_yticks(range(len(phases)))
            ax.set_xticklabels(shot_numbers, fontsize=cell_text_size, rotation=45 if len(shot_numbers) > 10 else 0)
            ax.set_yticklabels([p.replace('_Time(s)', '') for p in phases], fontsize=fontsize)
            
            # Add text annotations
            for i in range(len(phases)):
                for j in range(len(shot_numbers)):
                    value = heatmap_data[i, j]
                    text = ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                                 fontsize=cell_text_size, color='white' if value > np.max(heatmap_data)/2 else 'black')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Time (seconds)', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize-2)
            
            ax.set_title(f"{session}", fontsize=fontsize, pad=10)
            ax.set_xlabel('Shot Number', fontsize=fontsize)
            ax.set_ylabel('Timing Phases', fontsize=fontsize)
        
        fig.tight_layout()
        return fig

    def side_by_side_bar_chart(self, figsize=(15, 7), scaling_settings=None):
        """Create side-by-side bar chart with improved scalability"""
        import numpy as np
        
        # Apply scaling settings
        if scaling_settings:
            figsize = scaling_settings['figsize']
            fontsize = scaling_settings['fontsize']
            auto_scale = scaling_settings['auto_scale']
            compact_mode = scaling_settings['compact_mode']
        else:
            fontsize = 10
            auto_scale = True
            compact_mode = False
        
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        df = self.df.copy()
        if df.empty:
            ax.text(0.5, 0.5, "No data available for side-by-side bar chart", ha='center', va='center', fontsize=fontsize)
            fig.tight_layout()
            return fig
        
        # Get unique sessions and players
        sessions = df['Session'].unique() if 'Session' in df.columns else ['Session']
        players = df['Player'].unique() if 'Player' in df.columns else ['Player']
        
        # Auto-scale for multiple sessions/players
        if auto_scale and (len(sessions) > 5 or len(players) > 3):
            # Adjust figure width based on number of categories
            width = min(24, 8 + max(len(sessions), len(players)) * 1.5)  # Max 24 inches
            fig.set_size_inches(width, figsize[1])
        
        # Prepare data for plotting
        phases = ['Up_Time(s)', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
        colors = PHASE_COLORS_LIST_WITH_UP
        
        # Calculate averages for each session/player combination
        plot_data = []
        labels = []
        
        for session in sessions:
            for player in players:
                subset = df[(df['Session'] == session) & (df['Player'] == player)]
                if not subset.empty:
                    session_data = []
                    for phase in phases:
                        if phase in subset.columns:
                            avg_time = subset[phase].mean()
                            session_data.append(avg_time)
                        else:
                            session_data.append(0)
                    plot_data.append(session_data)
                    labels.append(f'{player}-{session}')
        
        if not plot_data:
            ax.text(0.5, 0.5, "No valid data for side-by-side bar chart", ha='center', va='center', fontsize=fontsize)
            fig.tight_layout()
            return fig
        
        # Create side-by-side bars
        x_pos = np.arange(len(labels))
        bar_width = 0.15 if compact_mode else 0.2
        
        # Phase colors for better visibility - use the correct list with UP phase
        phase_colors = PHASE_COLORS_LIST_WITH_UP
        phase_labels = ['Up', 'Preparing', 'Aiming', 'After Shot']
        
        for i, phase in enumerate(phases):
            values = [data[i] for data in plot_data]
            bars = ax.bar(x_pos + i * bar_width, values, bar_width, 
                         label=phase_labels[i], color=phase_colors[i % len(phase_colors)], alpha=0.8, 
                         edgecolor='white', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Player-Session', fontsize=fontsize)
        ax.set_ylabel('Average Time (seconds)', fontsize=fontsize)
        title_text = 'Average Phase Durations by Player and Session'
        if len(sessions) > 1 and 'Session' in df.columns:
            title_text += f' ({len(sessions)} Sessions)'
        ax.set_title(title_text, fontsize=fontsize, fontweight='bold', pad=20)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_pos + bar_width * 1.5)
        ax.set_xticklabels(labels, fontsize=max(8, fontsize-2), rotation=45 if len(labels) > 5 else 0)
        
        # Add legend with better positioning
        ax.legend(loc='upper right', fontsize=max(8, fontsize-2), framealpha=0.9)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Improve y-axis formatting
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}s'))
        
        # Add some padding to the y-axis
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.1)
        
        fig.tight_layout()
        return fig
    
    def create_interactive_plotly(self):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Line Chart: Timing Phases', 'Box Plot Distribution',
                          'Bar Chart: Average Times', 'Score vs Durations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Line chart
        colors = PHASE_COLORS_LIST_WITH_UP
        for i, phase in enumerate(self.time_phases):
            if phase in self.df.columns:
                fig.add_trace(
                    go.Scatter(x=self.df['Shot_Number'], y=self.df[phase],
                              name=phase.replace('_Time(s)', ''),
                              line=dict(color=colors[i % len(colors)]),
                              mode='lines+markers'),
                    row=1, col=1
                )
        
        # Box plot
        for i, phase in enumerate(self.time_phases):
            if phase in self.df.columns:
                fig.add_trace(
                    go.Box(y=self.df[phase], name=phase.replace('_Time(s)', ''),
                           marker_color=colors[i % len(colors)]),
                    row=1, col=2
                )
        
        # Bar chart - average times
        avg_times = [self.df[phase].mean() for phase in self.time_phases if phase in self.df.columns]
        phase_names = [phase.replace('_Time(s)', '') for phase in self.time_phases if phase in self.df.columns]
        
        fig.add_trace(
            go.Bar(x=phase_names, y=avg_times, marker_color=colors[:len(avg_times)]),
            row=2, col=1
        )
        
        # Score vs Durations chart (replacing scatter plot)
        if 'Score' in self.df.columns:
            # Prepare data for score vs durations
            df_clean = self.df.copy()
            df_clean = df_clean[pd.to_numeric(df_clean['Score'], errors='coerce').notnull()]
            df_clean['Score'] = pd.to_numeric(df_clean['Score'])
            
            if not df_clean.empty:
                durations = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
                colors_durations = PHASE_COLORS_LIST
                
                for i, (duration, color) in enumerate(zip(durations, colors_durations)):
                    if duration in df_clean.columns:
                        fig.add_trace(
                            go.Bar(x=df_clean['Score'], y=df_clean[duration],
                                   name=duration.replace('_Time(s)', ''),
                                   marker_color=color,
                                   opacity=0.7),
                            row=2, col=2
                        )
                
                # Add total duration as a line
                if all(duration in df_clean.columns for duration in durations):
                    total_duration = df_clean[durations].sum(axis=1)
                    fig.add_trace(
                        go.Scatter(x=df_clean['Score'], y=total_duration,
                                   name='Total Duration',
                                   mode='lines+markers',
                                   line=dict(color='black', width=2),
                                   marker=dict(size=8)),
                        row=2, col=2
                    )
        
        fig.update_layout(height=800, showlegend=True, title_text="Interactive Shot Timing Dashboard")
        return fig
    def score_vs_durations(self, figsize=(10, 6), scaling_settings=None):
        """Create score vs durations chart with session information"""
        # Apply scaling settings
        if scaling_settings:
            figsize = scaling_settings['figsize']
            fontsize = scaling_settings['fontsize']
            auto_scale = scaling_settings['auto_scale']
            compact_mode = scaling_settings['compact_mode']
        else:
            fontsize = 10
            auto_scale = True
            compact_mode = False
        
        # Adjust figure size for large datasets
        if len(self.df) > 30:
            figsize = (max(16, figsize[0]), max(8, figsize[1]))
        
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)
        df = self.df.copy()
        
        # Ensure Score column exists and is numeric
        if 'Score' not in df.columns:
            ax.text(0.5, 0.5, "No 'Score' column found.", ha='center', va='center', fontsize=14)
            fig.tight_layout()
            return fig
        
        # Remove rows with missing or non-numeric scores
        df = df[pd.to_numeric(df['Score'], errors='coerce').notnull()]
        df['Score'] = pd.to_numeric(df['Score'])
        if df.empty:
            ax.text(0.5, 0.5, "No valid scores to plot.", ha='center', va='center', fontsize=14)
            fig.tight_layout()
            return fig
        
        # Prepare data with session information
        durations = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
        colors = PHASE_COLORS_LIST
        
        # Get unique sessions
        sessions = df['Session'].unique().tolist() if 'Session' in df.columns else ['Session']
        
        # For large datasets, group by score instead of individual shots
        if len(df) > 30 and compact_mode:
            # Group by score and calculate average durations
            score_groups = df.groupby('Score')[durations].mean()
            
            x_positions = list(range(len(score_groups)))
            x_labels = [f"Score {score}" for score in score_groups.index]
            
            # Plot bars for each duration
            width = 0.2
            for i, duration in enumerate(durations):
                values = score_groups[duration].values
                ax.bar([x + i*width - width for x in x_positions], values, width,
                      label=duration.replace("_Time(s)", ""),
                      color=colors[i % len(colors)], alpha=0.8)
            
            # Add total duration bars
            for i, score in enumerate(score_groups.index):
                total = score_groups.loc[score].sum()
                ax.bar(x_positions[i] + width, total, width, fill=False, 
                      edgecolor='black', linewidth=2, label="Total Duration" if i == 0 else "")
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Score")
            ax.set_ylabel("Average Duration (s)")
            ax.set_title("Average Durations by Score")
            
        else:
            # Original individual shot plotting for smaller datasets
            x_positions = []
            x_labels = []
            session_colors = []
            
            current_x = 0
            for session in sessions:
                session_data = df[df['Session'] == session] if 'Session' in df.columns else df
                for _, row in session_data.iterrows():
                    x_positions.append(current_x)
                    score = int(row['Score'])
                    shot_num = row.get('Shot_Number', '?')
                    x_labels.append(f"{score}\n({session})")
                    session_colors.append(session)
                    current_x += 1
                current_x += 0.5  # Add space between sessions
            
            if not x_positions:
                ax.text(0.5, 0.5, "No valid data to plot.", ha='center', va='center', fontsize=14)
                fig.tight_layout()
                return fig
            
            # Plot bars for each duration
            width = 0.15
            for i, duration in enumerate(durations):
                values = []
                for session in sessions:
                    session_data = df[df['Session'] == session] if 'Session' in df.columns else df
                    for _, row in session_data.iterrows():
                        values.append(row.get(duration, 0))
                
                # Plot bars
                for j, (x, value) in enumerate(zip(x_positions, values)):
                    ax.bar(x + i*width - width, value, width, 
                          label=duration.replace("_Time(s)", "") if j == 0 else "",
                          color=colors[i % len(colors)], alpha=0.8)
            
            # Add total duration bars
            for j, x in enumerate(x_positions):
                session_name = session_colors[j]
                session_data = df[df['Session'] == session_name] if 'Session' in df.columns else df
                
                # Find the corresponding row
                matching_rows = session_data[session_data['Score'] == int(x_labels[j].split('\n')[0])]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    total = sum(row.get(duration, 0) for duration in durations)
                    ax.bar(x + width, total, width, fill=False, edgecolor='black', linewidth=2, 
                          label="Total Duration" if j == 0 else "")
            
            # Improved x-axis label handling for large datasets
            if len(x_positions) > 20:
                # For very large datasets (60+ shots), show fewer labels
                if len(x_positions) > 50:
                    # Show every 5th label for 50+ shots
                    step = max(1, len(x_positions) // 12)  # Show max 12 labels
                elif len(x_positions) > 30:
                    # Show every 3rd label for 30+ shots
                    step = max(1, len(x_positions) // 20)  # Show max 20 labels
                else:
                    # Show every 2nd label for 20+ shots
                    step = max(1, len(x_positions) // 15)  # Show max 15 labels
                
                ax.set_xticks(x_positions[::step])
                ax.set_xticklabels(x_labels[::step], rotation=45, ha='right', fontsize=6)
                
                # Add a note about label spacing
                ax.text(0.02, 0.02, f"Showing every {step}th shot label", 
                       transform=ax.transAxes, fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            ax.set_xlabel("Score (Session)")
            ax.set_ylabel("Duration (s)")
            ax.set_title("Durations per Shot Score by Session")
        
        # Add legend
        ax.legend()
        
        # Add session information in the title or as text
        if len(sessions) > 1:
            session_info = f"Sessions: {', '.join(sessions)}"
            ax.text(0.02, 0.98, session_info, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=10)
        
        fig.tight_layout()
        return fig

    def virtual_target_visualization(self, figsize=(12, 10), scaling_settings=None, zoom_target=False):
        """Create a virtual 10-meter air pistol target visualization with improved scalability"""
        import numpy as np
        import matplotlib.patches as patches
        
        # Apply scaling settings
        if scaling_settings:
            figsize = scaling_settings['figsize']
            fontsize = scaling_settings['fontsize']
            auto_scale = scaling_settings['auto_scale']
            compact_mode = scaling_settings['compact_mode']
        else:
            fontsize = 10
            auto_scale = True
            compact_mode = False
        
        # Auto-enable compact mode for many shots (30+ shots)
        df = self.df.copy()
        if len(df) >= 30 and not compact_mode:
            compact_mode = True
        
        # Increase figure width to accommodate side panels
        fig = Figure(figsize=(figsize[0] + 4, figsize[1]))
        
        # Create a grid layout: target on left, cluster info in middle, shot list on right
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 2], wspace=0.1)
        
        # Main target area
        ax = fig.add_subplot(gs[0], aspect='equal')
        
        df = self.df.copy()
        
        # Check if required columns exist
        if 'Score' not in df.columns or 'Direction' not in df.columns:
            ax.text(0.5, 0.5, "Missing 'Score' or 'Direction' columns.\nPlease ensure both columns are present.", 
                   ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)
            fig.tight_layout()
            return fig
        
        # Filter out rows without scores or directions
        df = df.dropna(subset=['Score', 'Direction'])
        df = df[df['Direction'] != '']
        
        if df.empty:
            ax.text(0.5, 0.5, "No valid shots with both score and direction data.", 
                   ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)
            fig.tight_layout()
            return fig
        
        # Convert scores to numeric
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df = df.dropna(subset=['Score'])
        
        if df.empty:
            ax.text(0.5, 0.5, "No valid numeric scores found.", 
                   ha='center', va='center', fontsize=fontsize, transform=ax.transAxes)
            fig.tight_layout()
            return fig
        
        # Auto-scale for many sessions
        if auto_scale and len(df['Session'].unique()) > 3:
            # Adjust target size based on number of sessions
            target_scale = min(1.5, 1.0 + len(df['Session'].unique()) * 0.1)
            ax.set_xlim(-100 * target_scale, 100 * target_scale)
            ax.set_ylim(-100 * target_scale, 100 * target_scale)
        
        # Define target rings (10-meter air pistol target)
        # Ring diameters in mm (from center outward)
        ring_diameters = [11.5, 27.5, 43.5, 59.5, 75.5, 91.5, 107.5, 123.5, 139.5, 155.5, 171.5]
        ring_scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # Score for each ring
        
        # Apply zoom if requested
        if zoom_target:
            # Zoom to show only the high-scoring rings (8, 9, 10)
            zoom_radius = 60  # Show rings up to score 6
            ax.set_xlim(-zoom_radius, zoom_radius)
            ax.set_ylim(-zoom_radius, zoom_radius)
            # Adjust font sizes for zoomed view
            fontsize = max(8, fontsize - 1)
            label_fontsize = max(6, fontsize - 2)
        else:
            # Full target view
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            label_fontsize = max(4, fontsize-4) if compact_mode else max(6, fontsize-2)
        
        # Draw target rings
        colors = ['#FFD700', '#FFA500', '#FF6347', '#FF4500', '#FF0000', 
                 '#FF69B4', '#FF1493', '#8A2BE2', '#4B0082', '#0000FF', '#00FFFF']
        
        for i, (diameter, score, color) in enumerate(zip(ring_diameters, ring_scores, colors)):
            circle = patches.Circle((0, 0), diameter/2, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(circle)
            
            # Add score labels on all sides (top, bottom, left, right)
            if i < len(ring_scores) - 1:  # Don't label the outer ring
                radius = diameter/2 + 5
                # Top
                ax.text(0, radius, str(score), ha='center', va='bottom', 
                       fontsize=max(6, fontsize-2), fontweight='bold', color=color)
                # Bottom
                ax.text(0, -radius, str(score), ha='center', va='top', 
                       fontsize=max(6, fontsize-2), fontweight='bold', color=color)
                # Left
                ax.text(-radius, 0, str(score), ha='right', va='center', 
                       fontsize=max(6, fontsize-2), fontweight='bold', color=color)
                # Right
                ax.text(radius, 0, str(score), ha='left', va='center', 
                       fontsize=max(6, fontsize-2), fontweight='bold', color=color)
        
        # Define direction mappings (angles in degrees)
        direction_angles = {
            'N': 90, 'S': 270, 'E': 0, 'W': 180,
            'NE': 45, 'NW': 135, 'SE': 315, 'SW': 225
        }
        
        # Group shots by score and direction to handle overlapping
        shot_groups = {}
        for idx, row in df.iterrows():
            score = row['Score']
            direction = row['Direction']
            shot_num = row.get('Shot_Number', idx + 1)
            
            if direction in direction_angles and score in ring_scores:
                key = (score, direction)
                if key not in shot_groups:
                    shot_groups[key] = []
                shot_groups[key].append(shot_num)
        
        # Get session colors and determine legend type
        session_colors = self._get_session_colors()
        legend_type = self._get_legend_type()
        
        # Calculate session scores (out of 600 for 60 shots)
        session_scores = {}
        if 'Session' in df.columns:
            for session in df['Session'].unique():
                session_df = df[df['Session'] == session]
                total_score = session_df['Score'].sum()
                session_scores[session] = total_score
        
        # Plot shots with clustering for overlapping
        shot_markers = []
        shot_labels = []
        cluster_info = []  # Store cluster information for side panel
        
        # Much larger marker sizes for better visibility - significantly increased
        marker_size = 100 if compact_mode else 250  # Increased from 50/150 to 100/250
        if zoom_target:
            marker_size = min(150, marker_size + 30)  # Increased zoom bonus to 30
        
        for (score, direction), shot_numbers in shot_groups.items():
            # Calculate base position based on score and direction
            angle_rad = np.radians(direction_angles[direction])
            
            # Find the ring index for this score
            ring_idx = ring_scores.index(score)
            if ring_idx < len(ring_diameters):
                # Position within the ring
                ring_radius = ring_diameters[ring_idx] / 2
                inner_radius = ring_diameters[ring_idx - 1] / 2 if ring_idx > 0 else 0
                
                # Base radius within the ring
                base_radius = inner_radius + (ring_radius - inner_radius) * 0.7
                
                # Handle multiple shots at the same location
                num_shots = len(shot_numbers)
                
                if num_shots == 1:
                    # Single shot - plot normally
                    x = base_radius * np.cos(angle_rad)
                    y = base_radius * np.sin(angle_rad)
                    
                    # Get shot color based on session or score
                    shot_color = colors[ring_idx]  # Default to score-based color
                    if legend_type == 'session':
                        # Find the session for this shot
                        shot_data = df[df['Shot_Number'] == shot_numbers[0]]
                        if not shot_data.empty:
                            session = shot_data.iloc[0].get('Session', 'Unknown')
                            shot_color = session_colors.get(session, colors[ring_idx])
                    
                    marker = ax.scatter(x, y, s=marker_size, c=shot_color, edgecolors='black', 
                                      linewidth=1, alpha=0.8, zorder=10)
                    shot_markers.append(marker)
                    
                    # Add shot number label - smaller font for compact mode
                    if not compact_mode or len(shot_labels) < 10:  # Show fewer labels in compact mode
                        ax.annotate(str(shot_numbers[0]), (x, y), xytext=(3, 3), 
                                  textcoords='offset points', fontsize=max(6, label_fontsize//2), 
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                else:
                    # Multiple shots at same location - create cluster with RED color for overlapping
                    # Calculate cluster radius based on number of shots - much smaller for better visibility
                    cluster_radius = min(8, 3 + num_shots * 1)  # Max 8 units radius for compact display
                    
                    # Create a small circle pattern for the cluster
                    for i, shot_num in enumerate(shot_numbers):
                        # Calculate position within the cluster
                        cluster_angle = (2 * np.pi * i) / num_shots
                        cluster_x = base_radius * np.cos(angle_rad) + cluster_radius * np.cos(cluster_angle)
                        cluster_y = base_radius * np.sin(angle_rad) + cluster_radius * np.sin(cluster_angle)
                        
                        # Use RED color for overlapping shots
                        shot_color = '#FF0000'  # Red color for overlapping shots
                        
                        # Plot individual shot - much smaller for overlapping shots
                        marker = ax.scatter(cluster_x, cluster_y, s=max(50, marker_size//2), c=shot_color, 
                                          edgecolors='black', linewidth=0.5, alpha=0.8, zorder=10)
                        shot_markers.append(marker)
                        
                        # Add shot number label (only for first few in compact mode) - smaller font
                        if not compact_mode or len(shot_labels) < 10:  # Show fewer labels in compact mode
                            ax.annotate(str(shot_num), (cluster_x, cluster_y), xytext=(2, 2), 
                                      textcoords='offset points', fontsize=max(4, label_fontsize//3), 
                                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
                    
                    # Store cluster info for side panel (don't plot on target)
                    cluster_info.append({
                        'score': score,
                        'direction': direction,
                        'count': num_shots,
                        'shots': shot_numbers,
                        'color': '#FF0000',  # Red for overlapping shots
                        'sessions': df[df['Shot_Number'].isin(shot_numbers)]['Session'].unique().tolist() if legend_type == 'session' else []
                    })
                    
                    # Add cluster count label for many shots
                    if compact_mode and num_shots > 3:
                        ax.annotate(f'×{num_shots}', (base_radius * np.cos(angle_rad), base_radius * np.sin(angle_rad)), 
                                  xytext=(0, 0), textcoords='offset points', 
                                  fontsize=max(6, fontsize-4), fontweight='bold',
                                  ha='center', va='center',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.9, edgecolor='white'))
        
        # Add dynamic legend based on data type
        if legend_type == 'session':
            # Session-based legend
            legend_elements = []
            for session, color in session_colors.items():
                legend_elements.append(patches.Patch(color=color, label=f'Session: {session}'))
            ax.legend(handles=legend_elements, loc='upper right', title='Sessions', fontsize=fontsize-2)
        else:
            # Score-based legend
            legend_elements = []
            for score, color in zip(ring_scores[:6], colors[:6]):  # Show top 6 scores
                legend_elements.append(patches.Patch(color=color, label=f'Score {score}'))
            ax.legend(handles=legend_elements, loc='upper right', title='Score Rings', fontsize=fontsize-2)
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add dynamic title with session information
        title_text = 'Virtual 10-Meter Air Pistol Target\nShot Distribution by Score and Direction'
        if legend_type == 'session':
            num_sessions = len(session_colors)
            title_text += f'\n({num_sessions} Sessions)'
        if zoom_target:
            title_text += ' (Zoomed View)'
        if compact_mode:
            title_text += ' (Compact View)'
        ax.set_title(title_text, fontsize=fontsize, fontweight='bold', pad=20)
        
        # Add direction labels
        for direction, angle in direction_angles.items():
            angle_rad = np.radians(angle)
            x = 95 * np.cos(angle_rad)
            y = 95 * np.sin(angle_rad)
            ax.text(x, y, direction, ha='center', va='center', fontsize=fontsize-2, 
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Add summary statistics with session scores
        total_shots = len(df)
        avg_score = df['Score'].mean()
        
        # Create summary text
        summary_text = f'Total Shots: {total_shots}\nAverage Score: {avg_score:.1f}'
        
        # Add compact mode info
        if compact_mode:
            summary_text += f'\nCompact Mode: Enabled\nSmaller markers for {total_shots} shots'
        
        # Add session scores if available
        if session_scores:
            summary_text += '\n\nSession Scores:'
            for session, score in session_scores.items():
                summary_text += f'\n{session}: {score:.0f}/600'
        
        ax.text(0.02, 0.98, summary_text, 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
               fontsize=fontsize-2)
        
        # Create cluster info panel (middle)
        if cluster_info:
            ax_cluster = fig.add_subplot(gs[1])
            ax_cluster.set_xlim(0, 1)
            ax_cluster.set_ylim(0, len(cluster_info))
            ax_cluster.set_title('Shot Clusters', fontsize=fontsize-2, fontweight='bold')
            ax_cluster.set_xticks([])
            ax_cluster.set_yticks([])
            
            for i, cluster in enumerate(cluster_info):
                y_pos = len(cluster_info) - i - 1
                # Create colored box
                rect = patches.Rectangle((0.1, y_pos + 0.1), 0.8, 0.8, 
                                       facecolor=cluster['color'], alpha=0.8, edgecolor='black')
                ax_cluster.add_patch(rect)
                
                # Add text with session info if available
                cluster_text = f"Score {cluster['score']}\n{cluster['direction']}\n×{cluster['count']}"
                if legend_type == 'session' and cluster.get('sessions'):
                    sessions_text = ', '.join(cluster['sessions'])
                    cluster_text += f"\nSessions: {sessions_text}"
                
                ax_cluster.text(0.5, y_pos + 0.5, cluster_text, 
                              ha='center', va='center', fontsize=fontsize-3, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
        
        # Create shot list panel (right side - SIUS style)
        ax_shot_list = fig.add_subplot(gs[2])
        ax_shot_list.set_xlim(0, 1)
        ax_shot_list.set_ylim(0, len(df) + 1)
        shot_list_title = 'Shot List (SIUS Style)'
        if legend_type == 'session':
            shot_list_title += ' - Session Colors'
        ax_shot_list.set_title(shot_list_title, fontsize=fontsize-2, fontweight='bold')
        ax_shot_list.set_xticks([])
        ax_shot_list.set_yticks([])
        
        # Sort shots by shot number
        df_sorted = df.sort_values('Shot_Number')
        
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            shot_num = row.get('Shot_Number', idx + 1)
            score = row['Score']
            direction = row['Direction']
            session = row.get('Session', 'Unknown')
            
            y_pos = len(df) - i
            
            # Shot number
            ax_shot_list.text(0.1, y_pos, f"{shot_num:2d}", ha='left', va='center', 
                            fontsize=fontsize-3, fontweight='bold')
            
            # Score with session color background if multiple sessions
            score_bg_color = 'lightblue'
            if legend_type == 'session':
                score_bg_color = session_colors.get(session, 'lightblue')
            
            ax_shot_list.text(0.5, y_pos, f"{score:.1f}", ha='center', va='center', 
                            fontsize=fontsize-3, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=score_bg_color, alpha=0.7))
            
            # Direction label
            ax_shot_list.text(0.8, y_pos, direction, ha='center', va='center', 
                            fontsize=fontsize-4)
            
            # Session indicator (only show if multiple sessions)
            if legend_type == 'session':
                ax_shot_list.text(0.95, y_pos, session, ha='center', va='center', 
                                fontsize=fontsize-4, color=session_colors.get(session, '#666666'))
        
        fig.tight_layout()
        return fig
class ShotTimingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self._syncing_sliders = False  # Prevents AttributeError for slider sync logic
        
        # Multi-session data management
        self.sessions_data = {}  # Dictionary to store multiple sessions: {session_name: dataframe}
        self.sessions_info = {}  # Dictionary to store session metadata: {session_name: file_path}
        self.combined_df = None  # Combined dataframe from all sessions
        self.visualizer = None # For timing data charts

        # For Skeleton Analysis
        self.skeleton_data_store = {} 
        self.original_skeleton_names = {}
        self.skeleton_file_counter = 0
        self.current_skeleton_df_for_playback = None 
        
        self.skeleton_playback_timer = QTimer(self)
        self.current_skeleton_frame_index = 0 
        self.total_skeleton_frames = 0
        self.skeleton_fps = 30 
        self.unique_frame_numbers_for_playback = []
        
        # Initialize debounce timers for virtual targets
        self.virtual_target_debounce_timer = QTimer()
        self.virtual_target_debounce_timer.setSingleShot(True)
        self.virtual_target_debounce_timer.timeout.connect(self._debounced_virtual_target_update)

        # Add static chart zoom controls after viz_group is created
        # (This is also called in create_control_panel after viz_group is created)
        # self.add_static_chart_zoom_controls()

        self.init_ui()
        self.add_static_chart_zoom_controls()
        
    def add_static_chart_zoom_controls(self):
        """Add zoom in/out buttons for static chart (Virtual Target Visualization)"""
        from PyQt5.QtWidgets import QHBoxLayout, QPushButton
        zoom_layout = QHBoxLayout()
        self.static_zoom_in_btn = QPushButton("Zoom In")
        self.static_zoom_out_btn = QPushButton("Zoom Out")
        zoom_layout.addWidget(self.static_zoom_in_btn)
        zoom_layout.addWidget(self.static_zoom_out_btn)
        # Add to visualization options group (viz_group)
        self.viz_group.layout().addLayout(zoom_layout)

        self.static_zoom_in_btn.clicked.connect(self.static_chart_zoom_in)
        self.static_zoom_out_btn.clicked.connect(self.static_chart_zoom_out)

    def static_chart_zoom_in(self):
        """Zoom in on the static virtual target chart"""
        # Only zoom if Virtual Target Visualization is selected
        if self.chart_list.currentItem() and self.chart_list.currentItem().text() == "Virtual Target Visualization":
            for i in range(self.plot_layout.count()):
                widget = self.plot_layout.itemAt(i).widget()
                if hasattr(widget, 'figure') and widget.figure.axes:
                    ax = widget.figure.axes[0]
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    cx = (xlim[0] + xlim[1]) / 2
                    cy = (ylim[0] + ylim[1]) / 2
                    xr = (xlim[1] - xlim[0]) * 0.5
                    yr = (ylim[1] - ylim[0]) * 0.5
                    ax.set_xlim(cx - xr/2, cx + xr/2)
                    ax.set_ylim(cy - yr/2, cy + yr/2)
                    widget.draw()
                    break

    def static_chart_zoom_out(self):
        """Zoom out on the static virtual target chart, but do not exceed original target bounds (-100, 100)"""
        if self.chart_list.currentItem() and self.chart_list.currentItem().text() == "Virtual Target Visualization":
            for i in range(self.plot_layout.count()):
                widget = self.plot_layout.itemAt(i).widget()
                if hasattr(widget, 'figure') and widget.figure.axes:
                    ax = widget.figure.axes[0]
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    cx = (xlim[0] + xlim[1]) / 2
                    cy = (ylim[0] + ylim[1]) / 2
                    xr = (xlim[1] - xlim[0]) * 2.0
                    yr = (ylim[1] - ylim[0]) * 2.0
                    # Clamp to original bounds (-100, 100)
                    new_xmin = max(-100, cx - xr/2)
                    new_xmax = min(100, cx + xr/2)
                    new_ymin = max(-100, cy - yr/2)
                    new_ymax = min(100, cy + yr/2)
                    # If the new range is smaller than the bounds, expand to bounds
                    if new_xmin == -100 and new_xmax < 100:
                        new_xmax = 100
                    if new_xmax == 100 and new_xmin > -100:
                        new_xmin = -100
                    if new_ymin == -100 and new_ymax < 100:
                        new_ymax = 100
                    if new_ymax == 100 and new_ymin > -100:
                        new_ymin = -100
                    ax.set_xlim(new_xmin, new_xmax)
                    ax.set_ylim(new_ymin, new_ymax)
                    widget.draw()
                    break

    def init_ui(self):
        self.setWindowTitle("Professional Shot Timing Analysis Dashboard")
        self.setGeometry(100, 100, 1600, 900) 
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main vertical layout for the entire window
        window_layout = QVBoxLayout(central_widget)
        
        # Create the toggle button at the very top
        self.collapse_btn = QPushButton("◀ Hide Menu")
        self.collapse_btn.setMaximumWidth(80)
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        self.collapse_btn.clicked.connect(self.toggle_control_panel)
        window_layout.addWidget(self.collapse_btn)
        
        # Create horizontal layout for the main content
        main_layout = QHBoxLayout()
        window_layout.addLayout(main_layout)
        
        self.create_control_panel(main_layout) 
        
        # Create a scrollable area for the entire tab widget
        self.tab_scroll_area = QScrollArea()
        self.tab_scroll_area.setWidgetResizable(True)
        self.tab_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tab_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tab_scroll_area.setMinimumHeight(700)
        
        self.tab_widget = QTabWidget()
        
        self.matplotlib_tab = QWidget()
        matplotlib_layout = QVBoxLayout(self.matplotlib_tab)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.plot_widget = QWidget() 
        self.plot_layout = QVBoxLayout(self.plot_widget)
        scroll_area.setWidget(self.plot_widget)
        matplotlib_layout.addWidget(scroll_area)
        self.tab_widget.addTab(self.matplotlib_tab, "Static Charts")
        
        self.interactive_tab = QWidget()
        interactive_layout = QVBoxLayout(self.interactive_tab)
        self.web_view = PlotlyWebView() 
        interactive_layout.addWidget(self.web_view)
        self.tab_widget.addTab(self.interactive_tab, "Interactive Dashboard")
        
        self.table_tab = QWidget()
        table_layout = QVBoxLayout(self.table_tab)
        
        # Add save button for data table
        save_button_layout = QHBoxLayout()
        self.save_data_btn = QPushButton("Save Changes")
        self.save_data_btn.clicked.connect(self.save_data_table_changes)
        self.save_data_btn.setEnabled(False)  # Disabled until data is loaded
        save_button_layout.addWidget(self.save_data_btn)
        save_button_layout.addStretch()  # Push button to the left
        table_layout.addLayout(save_button_layout)
        
        # Add help text for direction system
        direction_help = QLabel("Direction System: Use N, S, E, W, NE, NW, SE, SW to mark shot directions (like SIUS system arrows)")
        direction_help.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        table_layout.addWidget(direction_help)
        
        self.data_table = QTableWidget()
        table_layout.addWidget(self.data_table)
        self.tab_widget.addTab(self.table_tab, "Data Table")

        # Skeleton Analysis Tab
        self.skeleton_analysis_tab = QWidget()
        self.setup_skeleton_analysis_tab(self.skeleton_analysis_tab) 
        self.tab_widget.addTab(self.skeleton_analysis_tab, "Skeleton Analysis")
        
        self.shot_playback_tab = QWidget()
        self.setup_shot_playback_tab(self.shot_playback_tab)
        self.tab_widget.addTab(self.shot_playback_tab, "Shot Playback Analysis")
        
        # Add the tab widget to the scrollable area
        self.tab_scroll_area.setWidget(self.tab_widget)
        main_layout.addWidget(self.tab_scroll_area, 1)
        
        # Connect tab changes to auto-load centralized data
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Initialize debounce timer for virtual target updates
        self.virtual_target_debounce_timer = QTimer()
        self.virtual_target_debounce_timer.setSingleShot(True)
        self.virtual_target_debounce_timer.timeout.connect(self._debounced_virtual_target_update) 
        
    def create_control_panel(self, main_layout):
        """Create the left control panel FOR TIMING ANALYSIS"""
        # Create a container widget for the entire control panel
        self.control_container = QWidget()
        self.control_container.setMaximumWidth(300)
        self.control_container.setMinimumWidth(250)
        
        # Create the main control layout
        control_layout = QVBoxLayout(self.control_container)
        
        # Simple data loading
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)
        
        # CSV file loading
        csv_layout = QHBoxLayout()
        self.csv_load_btn = QPushButton("Load CSV File")
        self.csv_load_btn.clicked.connect(self.load_file)
        csv_layout.addWidget(self.csv_load_btn)
        
        self.csv_status_label = QLabel("No CSV file loaded")
        self.csv_status_label.setStyleSheet("color: #666; font-style: italic;")
        csv_layout.addWidget(self.csv_status_label)
        data_layout.addLayout(csv_layout)
        
        # JSON file loading
        json_layout = QHBoxLayout()
        self.json_load_btn = QPushButton("Load JSON File")
        self.json_load_btn.clicked.connect(self.load_shot_playback_json)
        json_layout.addWidget(self.json_load_btn)
        
        self.json_status_label = QLabel("No JSON file loaded")
        self.json_status_label.setStyleSheet("color: #666; font-style: italic;")
        json_layout.addWidget(self.json_status_label)
        data_layout.addLayout(json_layout)
        
        # Status summary
        self.data_status_label = QLabel("Ready to load data")
        self.data_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
        data_layout.addWidget(self.data_status_label)
        
        control_layout.addWidget(data_group)
        
        # Filter controls group
        self.filter_group = QGroupBox("Filters")
        self.filter_group.setEnabled(False)
        filter_layout = QVBoxLayout(self.filter_group)
        
        # Player filter
        filter_layout.addWidget(QLabel("Player:"))
        self.player_combo = QComboBox()
        self.player_combo.currentTextChanged.connect(self._debounced_apply_filters)
        filter_layout.addWidget(self.player_combo)
        
        # Session filter
        filter_layout.addWidget(QLabel("Session:"))
        self.session_combo = QComboBox()
        self.session_combo.currentTextChanged.connect(self._debounced_apply_filters)
        filter_layout.addWidget(self.session_combo)
        
        # Shot range filter
        filter_layout.addWidget(QLabel("Shot Range:"))
        shot_range_layout = QHBoxLayout()
        self.shot_min_spin = QDoubleSpinBox()
        self.shot_max_spin = QDoubleSpinBox()
        self.shot_min_spin.valueChanged.connect(self._debounced_apply_filters)
        self.shot_max_spin.valueChanged.connect(self._debounced_apply_filters)
        shot_range_layout.addWidget(self.shot_min_spin)
        shot_range_layout.addWidget(QLabel("to"))
        shot_range_layout.addWidget(self.shot_max_spin)
        filter_layout.addLayout(shot_range_layout)

        # Score range filter
        filter_layout.addWidget(QLabel("Score Range:"))
        score_range_layout = QHBoxLayout()
        self.score_min_spin = QDoubleSpinBox()
        self.score_max_spin = QDoubleSpinBox()
        self.score_min_spin.setRange(0, 10)
        self.score_max_spin.setRange(0, 10)
        self.score_min_spin.setDecimals(1)
        self.score_max_spin.setDecimals(1)
        self.score_min_spin.setValue(0)
        self.score_max_spin.setValue(10)
        self.score_min_spin.valueChanged.connect(self._debounced_apply_filters)
        self.score_max_spin.valueChanged.connect(self._debounced_apply_filters)
        score_range_layout.addWidget(self.score_min_spin)
        score_range_layout.addWidget(QLabel("to"))
        score_range_layout.addWidget(self.score_max_spin)
        filter_layout.addLayout(score_range_layout)

        # Filter for completed shots
        self.complete_shot_checkbox = QCheckBox("Only show completed shots")
        self.complete_shot_checkbox.setChecked(True) # Default to showing only completed shots
        self.complete_shot_checkbox.stateChanged.connect(self._debounced_apply_filters)
        filter_layout.addWidget(self.complete_shot_checkbox)
        
        control_layout.addWidget(self.filter_group)
        
        # Visualization controls
        self.viz_group = QGroupBox("Visualization Options")
        self.viz_group.setEnabled(False)
        viz_layout = QVBoxLayout(self.viz_group)
        
        # Chart scaling controls
        scaling_group = QGroupBox("Chart Scaling & Readability")
        scaling_layout = QVBoxLayout(scaling_group)
        
        # Figure size controls
        fig_size_layout = QHBoxLayout()
        fig_size_layout.addWidget(QLabel("Figure Size:"))
        self.fig_width_spin = QSpinBox()
        self.fig_width_spin.setRange(8, 24)
        self.fig_width_spin.setValue(12)
        self.fig_width_spin.setSuffix(" in")
        fig_size_layout.addWidget(self.fig_width_spin)
        
        self.fig_height_spin = QSpinBox()
        self.fig_height_spin.setRange(6, 20)
        self.fig_height_spin.setValue(8)
        self.fig_height_spin.setSuffix(" in")
        fig_size_layout.addWidget(self.fig_height_spin)
        scaling_layout.addLayout(fig_size_layout)
        
        # Font size controls
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(10)
        font_size_layout.addWidget(self.font_size_spin)
        scaling_layout.addLayout(font_size_layout)
        
        # DPI control for high resolution
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 300)
        self.dpi_spin.setValue(100)
        dpi_layout.addWidget(self.dpi_spin)
        scaling_layout.addLayout(dpi_layout)
        
        # Auto-scale checkbox
        self.auto_scale_checkbox = QCheckBox("Auto-scale for multiple sessions")
        self.auto_scale_checkbox.setChecked(True)
        scaling_layout.addWidget(self.auto_scale_checkbox)
        
        # Compact mode for many shots
        self.compact_mode_checkbox = QCheckBox("Compact mode (for 60+ shots)")
        self.compact_mode_checkbox.setChecked(False)
        scaling_layout.addWidget(self.compact_mode_checkbox)
        
        # Zoom option for target visualization
        self.zoom_target_checkbox = QCheckBox("Zoom target (focus on high scores)")
        self.zoom_target_checkbox.setChecked(False)
        scaling_layout.addWidget(self.zoom_target_checkbox)
        
        viz_layout.addWidget(scaling_group)
        
        # Chart type selection
        viz_layout.addWidget(QLabel("Chart Type:"))
        self.chart_list = QListWidget()
        chart_types = [
            "Small-Multiples Line Charts",
            "Stacked Bar Composition",
            "Box-Violin Distribution",
            "Phase Duration Heatmap",
            "Side-by-Side Bar Chart",
            "Interactive Dashboard",
            "Score vs Durations",
            "Virtual Target Visualization"
        ]
        
        for chart_type in chart_types:
            item = QListWidgetItem(chart_type)
            self.chart_list.addItem(item)
        
        self.chart_list.currentItemChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.chart_list)
        
        # Generate button
        generate_btn = QPushButton("Generate Visualization")
        generate_btn.clicked.connect(self.generate_selected_chart)
        viz_layout.addWidget(generate_btn)
        
        control_layout.addWidget(self.viz_group)
        
        # Export options
        self.export_group = QGroupBox("Export Options")
        self.export_group.setEnabled(False)
        export_layout = QVBoxLayout(self.export_group)
        
        export_png_btn = QPushButton("Export as PNG")
        export_png_btn.clicked.connect(self.export_png)
        export_layout.addWidget(export_png_btn)
        
        export_html_btn = QPushButton("Export Interactive HTML")
        export_html_btn.clicked.connect(self.export_html)
        export_layout.addWidget(export_html_btn)
        
        control_layout.addWidget(self.export_group)
        
        # Add stretch to push everything to top
        control_layout.addStretch()
        
        # Add the control container to the main layout
        main_layout.addWidget(self.control_container)

    def toggle_control_panel(self):
        """Toggle the visibility of the control panel"""
        if self.control_container.isVisible():
            # Hide the control panel
            self.control_container.hide()
            self.collapse_btn.setText("▶ Show Menu")
            self.collapse_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 8px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
        else:
            # Show the control panel
            self.control_container.show()
            self.collapse_btn.setText("◀ Hide Menu")
            self.collapse_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    border: none;
                    padding: 8px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
            """)

    def setup_skeleton_analysis_tab(self, tab_widget):
        main_skeleton_layout = QHBoxLayout(tab_widget)

        skeleton_controls_widget = QWidget()
        skeleton_controls_widget.setMaximumWidth(350) 
        skeleton_controls_layout = QVBoxLayout(skeleton_controls_widget)

        file_management_group = QGroupBox("Skeleton Data Management")
        file_management_layout = QVBoxLayout(file_management_group)
        
        # Status label for centralized data
        self.skeleton_tab_status_label = QLabel("Using centralized data from main menu")
        self.skeleton_tab_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px; background-color: #E8F5E8;")
        file_management_layout.addWidget(self.skeleton_tab_status_label)
        
        # Legacy upload button (kept for compatibility)
        self.skeleton_upload_button = QPushButton("Load Additional Skeleton Data")
        self.skeleton_upload_button.clicked.connect(self.load_skeleton_file_dialog)
        file_management_layout.addWidget(self.skeleton_upload_button)
        
        self.loaded_skeleton_files_list = QListWidget()
        self.loaded_skeleton_files_list.itemSelectionChanged.connect(self.on_loaded_skeleton_selection_change)
        file_management_layout.addWidget(QLabel("Additional Skeleton Files:"))
        file_management_layout.addWidget(self.loaded_skeleton_files_list)
        
        # Add shot number filter
        file_management_layout.addWidget(QLabel("Shot Number:"))
        self.skeleton_shot_number_combo = QComboBox()
        self.skeleton_shot_number_combo.currentTextChanged.connect(self.on_skeleton_shot_number_change)
        file_management_layout.addWidget(self.skeleton_shot_number_combo)
        
        file_management_layout.addWidget(QLabel("Player State:"))
        self.skeleton_player_state_combo = QComboBox()
        self.skeleton_player_state_combo.currentTextChanged.connect(self.on_skeleton_player_state_change)
        file_management_layout.addWidget(self.skeleton_player_state_combo)
        skeleton_controls_layout.addWidget(file_management_group)

        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout(playback_group)
        self.skeleton_video_controls = VideoControls() 
        playback_layout.addWidget(self.skeleton_video_controls)
        skeleton_controls_layout.addWidget(playback_group)

        drawing_tools_group = QGroupBox("Drawing Tools")
        drawing_tools_layout = QGridLayout(drawing_tools_group)
        self.enable_drawing_checkbox = QCheckBox("Enable Drawing")
        self.enable_drawing_checkbox.toggled.connect(self.toggle_skeleton_drawing_mode)
        drawing_tools_layout.addWidget(self.enable_drawing_checkbox, 0, 0, 1, 2)
        drawing_tools_layout.addWidget(QLabel("Shape:"), 1, 0)
        self.shape_type_combo = QComboBox()
        self.shape_type_combo.addItems(["Line", "Rectangle"])
        self.shape_type_combo.currentTextChanged.connect(self.set_skeleton_drawing_shape)
        drawing_tools_layout.addWidget(self.shape_type_combo, 1, 1)
        drawing_tools_layout.addWidget(QLabel("Color:"), 2, 0)
        self.shape_color_combo = QComboBox()
        self.shape_color_combo.addItems(["Green", "Red", "Blue", "Yellow", "White", "Cyan", "Magenta"])
        self.shape_color_combo.currentTextChanged.connect(self.set_skeleton_drawing_color)
        self.shape_color_combo.setCurrentText("Green")
        drawing_tools_layout.addWidget(self.shape_color_combo, 2, 1)
        self.clear_drawings_button = QPushButton("Clear Drawings")
        self.clear_drawings_button.clicked.connect(self.clear_skeleton_drawings)
        drawing_tools_layout.addWidget(self.clear_drawings_button, 3, 0, 1, 2)
        skeleton_controls_layout.addWidget(drawing_tools_group)
        
        skeleton_controls_layout.addStretch()
        main_skeleton_layout.addWidget(skeleton_controls_widget)

        # Create a vertical layout for skeleton display and virtual target
        skeleton_display_layout = QVBoxLayout()
        
        # Skeleton display widget
        self.skeleton_display_widget = DrawableDisplayWidget()
        skeleton_display_layout.addWidget(self.skeleton_display_widget, 3)
        
        # Virtual target visualization for skeleton analysis
        skeleton_virtual_target_label = QLabel("Skeleton Virtual Target")
        skeleton_virtual_target_label.setAlignment(Qt.AlignCenter)
        skeleton_virtual_target_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        skeleton_display_layout.addWidget(skeleton_virtual_target_label)
        
        # Add filter controls for skeleton virtual target (separate from shot playback)
        skeleton_filter_layout = QHBoxLayout()
        
        # Shot range filter for skeleton
        skeleton_shot_range_layout = QHBoxLayout()
        skeleton_shot_range_layout.addWidget(QLabel("Shot Range:"))
        self.skeleton_min_spin = QSpinBox()
        self.skeleton_min_spin.setRange(0, 1000)
        self.skeleton_min_spin.setValue(0)
        self.skeleton_min_spin.valueChanged.connect(self._debounced_skeleton_virtual_target_update)
        skeleton_shot_range_layout.addWidget(self.skeleton_min_spin)
        skeleton_shot_range_layout.addWidget(QLabel("to"))
        self.skeleton_max_spin = QSpinBox()
        self.skeleton_max_spin.setRange(0, 1000)
        self.skeleton_max_spin.setValue(1000)
        self.skeleton_max_spin.valueChanged.connect(self._debounced_skeleton_virtual_target_update)
        skeleton_shot_range_layout.addWidget(self.skeleton_max_spin)
        skeleton_filter_layout.addLayout(skeleton_shot_range_layout)
        
        skeleton_filter_layout.addSpacing(20)
        
        # Score range filter for skeleton
        skeleton_score_range_layout = QHBoxLayout()
        skeleton_score_range_layout.addWidget(QLabel("Score Range:"))
        self.skeleton_score_min_spin = QDoubleSpinBox()
        self.skeleton_score_min_spin.setRange(0.0, 10.0)
        self.skeleton_score_min_spin.setValue(0.0)
        self.skeleton_score_min_spin.setDecimals(1)
        self.skeleton_score_min_spin.valueChanged.connect(self._debounced_skeleton_virtual_target_update)
        skeleton_score_range_layout.addWidget(self.skeleton_score_min_spin)
        skeleton_score_range_layout.addWidget(QLabel("to"))
        self.skeleton_score_max_spin = QDoubleSpinBox()
        self.skeleton_score_max_spin.setRange(0.0, 10.0)
        self.skeleton_score_max_spin.setValue(10.0)
        self.skeleton_score_max_spin.setDecimals(1)
        self.skeleton_score_max_spin.valueChanged.connect(self._debounced_skeleton_virtual_target_update)
        skeleton_score_range_layout.addWidget(self.skeleton_score_max_spin)
        skeleton_filter_layout.addLayout(skeleton_score_range_layout)
        
        skeleton_filter_layout.addStretch()
        
        # Clear filters button for skeleton
        self.clear_skeleton_filters_btn = QPushButton("Clear Filters")
        self.clear_skeleton_filters_btn.clicked.connect(self.clear_skeleton_filters)
        skeleton_filter_layout.addWidget(self.clear_skeleton_filters_btn)
        
        skeleton_display_layout.addLayout(skeleton_filter_layout)
        
        self.skeleton_virtual_target_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        skeleton_display_layout.addWidget(self.skeleton_virtual_target_canvas, 2)
        
        main_skeleton_layout.addLayout(skeleton_display_layout, 1)

        self.skeleton_video_controls.play_pause_btn.clicked.connect(self.toggle_skeleton_playback)
        self.skeleton_video_controls.stop_btn.clicked.connect(self.reset_skeleton_playback) 
        self.skeleton_video_controls.speed_slider.valueChanged.connect(self.set_skeleton_playback_speed)
        self.skeleton_video_controls.progress_slider.sliderMoved.connect(self.skeleton_playback_slider_moved)
        self.skeleton_video_controls.progress_slider.valueChanged.connect(self.skeleton_playback_slider_value_changed) 
        
        self.skeleton_video_controls.forward_btn.clicked.connect(lambda: self.seek_skeleton_relative(15)) 
        self.skeleton_video_controls.backward_btn.clicked.connect(lambda: self.seek_skeleton_relative(-15))

        self.skeleton_playback_timer.timeout.connect(self.advance_skeleton_frame)
        
        # Initialize skeleton virtual target
        self.update_skeleton_virtual_target()
        
        # Test virtual target in shot playback tab
        QTimer.singleShot(1000, lambda: self.update_shot_playback_virtual_target(1))

    def update_skeleton_virtual_target(self, shot_num=None, compare_shot_num=None):
        """
        Update the virtual target visualization for the skeleton analysis tab.
        """
        # Prevent infinite recursion with a counter
        if not hasattr(self, '_skeleton_virtual_target_update_count'):
            self._skeleton_virtual_target_update_count = 0
        
        if self._skeleton_virtual_target_update_count > 5:  # Max 5 updates in a row
            print("Preventing infinite loop in skeleton virtual target - too many consecutive updates")
            self._skeleton_virtual_target_update_count = 0
            return
        
        self._skeleton_virtual_target_update_count += 1
        
        print(f"Skeleton virtual target method called with shot_num={shot_num}, compare_shot_num={compare_shot_num}")
        
        # Create virtual target visualization
        print("Creating skeleton virtual target visualization...")
        self.skeleton_virtual_target_canvas.figure.clf()
        ax = self.skeleton_virtual_target_canvas.figure.add_subplot(111)
        
        # Create a simple virtual target visualization
        # Draw target rings
        target_center = (0, 0)
        ring_radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Ring radii
        ring_colors = ['gold', 'red', 'blue', 'black', 'white', 'white']
        ring_scores = [10, 9, 8, 7, 6, 5]  # Score for each ring
        
        # Draw target rings
        for i, (radius, color, score) in enumerate(zip(ring_radii, ring_colors, ring_scores)):
            circle = plt.Circle(target_center, radius, fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
            # Add score labels
            ax.text(0, radius + 0.1, str(score), ha='center', va='bottom', fontsize=8, color=color)
        
        # Get shot coordinates (simulate from data)
        # For now, we'll use random positions based on shot number for demonstration
        import random
        if shot_num is not None:
            random.seed(int(shot_num))  # Use shot number as seed for consistent positioning
        else:
            random.seed(42)  # Default seed
        
        # Simulate shot positions (in real implementation, these would come from actual data)
        shot_x = random.uniform(-2, 2)
        shot_y = random.uniform(-2, 2)
        
        # Plot the shot
        ax.scatter(shot_x, shot_y, color='red', s=100, zorder=5, label=f'Shot {shot_num}' if shot_num else 'Current Shot')
        
        # If comparison shot is provided, add it
        if compare_shot_num is not None:
            random.seed(int(compare_shot_num))
            compare_x = random.uniform(-2, 2)
            compare_y = random.uniform(-2, 2)
            ax.scatter(compare_x, compare_y, color='blue', s=100, zorder=5, label=f'Shot {compare_shot_num}')
        
        # Set up the plot
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        title = "Skeleton Virtual Target"
        if shot_num:
            title += f" - Shot {shot_num}"
        if compare_shot_num:
            title += f" vs Shot {compare_shot_num}"
        ax.set_title(title)
        ax.legend()
        
        self.skeleton_virtual_target_canvas.figure.tight_layout()
        self.skeleton_virtual_target_canvas.draw()
        print("Skeleton virtual target visualization completed and drawn")
        
        # Reset the counter when done
        self._skeleton_virtual_target_update_count = 0

    def _debounced_skeleton_virtual_target_update(self, shot_num=None):
        """Debounced skeleton virtual target update to prevent infinite loops"""
        # Stop any existing timer
        if hasattr(self, 'skeleton_virtual_target_debounce_timer'):
            self.skeleton_virtual_target_debounce_timer.stop()
        else:
            # Create timer if it doesn't exist
            self.skeleton_virtual_target_debounce_timer = QTimer()
            self.skeleton_virtual_target_debounce_timer.setSingleShot(True)
            self.skeleton_virtual_target_debounce_timer.timeout.connect(
                lambda: self.update_skeleton_virtual_target(shot_num)
            )
        
        # Start the timer with 500ms delay
        self.skeleton_virtual_target_debounce_timer.start(500)
    def load_skeleton_file_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Skeleton JSON", "", "JSON Files (*.json)")
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                all_landmarks_data = []

                # Primary parsing logic for JSONs with a top-level "frames" list
                if isinstance(data, dict) and ('frames' in data or 'Frames' in data):
                    frames_list_key = 'frames' if 'frames' in data else 'Frames'
                    actual_frames_list = data.get(frames_list_key)

                    if isinstance(actual_frames_list, list):
                        for frame_idx, frame_content in enumerate(actual_frames_list):
                            if not isinstance(frame_content, dict):
                                print(f"Skipping non-dictionary frame content at index {frame_idx}")
                                continue

                            frame_number = frame_idx  # Use 0-based index as Frame_Number

                            player_state = frame_content.get('phase', frame_content.get('Phase'))
                            
                            # Extract shot number from frame content
                            shot_number = frame_content.get('shot_number', frame_content.get('Shot_Number', 
                                           frame_content.get('shot', frame_content.get('Shot'))))
                            
                            landmarks_key_options = ['landmarks', 'Landmarks']
                            landmarks_in_frame = None
                            for key_opt in landmarks_key_options:
                                if key_opt in frame_content:
                                    landmarks_in_frame = frame_content[key_opt]
                                    break
                            
                            if isinstance(landmarks_in_frame, list):
                                for landmark_idx, landmark_data_item in enumerate(landmarks_in_frame):
                                    if not isinstance(landmark_data_item, dict):
                                        print(f"Skipping non-dictionary landmark data at frame {frame_idx}, landmark index {landmark_idx}")
                                        continue

                                    landmark_id = landmark_idx  # Use 0-based index as Landmark_ID
                                    
                                    landmark_x = landmark_data_item.get('x', landmark_data_item.get('X'))
                                    landmark_y = landmark_data_item.get('y', landmark_data_item.get('Y'))

                                    if landmark_x is not None and landmark_y is not None:
                                        record = {
                                            'Frame_Number': frame_number,
                                            'Player_State': player_state,
                                            'Shot_Number': shot_number,
                                            'Landmark_ID': landmark_id,
                                            'Landmark_X': landmark_x,
                                            'Landmark_Y': landmark_y
                                        }
                                        all_landmarks_data.append(record)
                                    else:
                                        # This can happen if a landmark dict is present but missing x/y
                                        print(f"Skipping landmark at frame {frame_idx}, id {landmark_idx} due to missing X or Y.")
                            # else: # This case means 'landmarks' key is missing or not a list for this frame_content
                                # If landmarks_in_frame is an empty list, the inner loop won't run, which is fine.
                                # Only print if it's not a list or missing entirely and expected.
                                # if landmarks_in_frame is None:
                                #    print(f"Frame {frame_idx} is missing 'landmarks' key or it's not a list.")

                    else:
                        QMessageBox.warning(self, "Invalid JSON Structure", f"The '{frames_list_key}' key was found, but it does not contain a list of frames.")
                        return
                
                # Fallback for JSON being a direct list of frames (older format or different structure)
                elif isinstance(data, list):
                    print("Attempting to parse JSON as a direct list of frames (fallback).")
                    for frame_idx, frame_obj in enumerate(data):
                        if not isinstance(frame_obj, dict):
                            print(f"Skipping non-dictionary item in main list at index {frame_idx}")
                            continue

                        # Attempt to get Frame_Number from key, else use index
                        frame_number_from_key = frame_obj.get('Frame_Number', frame_obj.get('frame_number'))
                        frame_number = frame_idx if frame_number_from_key is None else frame_number_from_key
                        
                        player_state = frame_obj.get('Player_State', frame_obj.get('player_state', frame_obj.get('phase')))
                        
                        # Extract shot number from frame object
                        shot_number = frame_obj.get('Shot_Number', frame_obj.get('shot_number', 
                                       frame_obj.get('shot', frame_obj.get('Shot'))))
                        
                        landmarks_list = frame_obj.get('Landmarks', frame_obj.get('landmarks'))

                        if isinstance(landmarks_list, list):
                            for lm_idx, landmark_data in enumerate(landmarks_list):
                                if not isinstance(landmark_data, dict):
                                    print(f"Skipping non-dictionary landmark in list-based frame {frame_number}, landmark index {lm_idx}")
                                    continue

                                # Attempt to get Landmark_ID from key, else use index
                                landmark_id_from_key = landmark_data.get('Landmark_ID', landmark_data.get('id'))
                                landmark_id = lm_idx if landmark_id_from_key is None else landmark_id_from_key
                                
                                landmark_x = landmark_data.get('Landmark_X', landmark_data.get('x', landmark_data.get('X')))
                                landmark_y = landmark_data.get('Landmark_Y', landmark_data.get('y', landmark_data.get('Y')))

                                if landmark_x is not None and landmark_y is not None:
                                    record = {
                                        'Frame_Number': frame_number,
                                        'Player_State': player_state,
                                        'Shot_Number': shot_number,
                                        'Landmark_ID': landmark_id,
                                        'Landmark_X': landmark_x,
                                        'Landmark_Y': landmark_y
                                    }
                                    all_landmarks_data.append(record)
                                else:
                                    print(f"Skipping landmark in list-based frame {frame_number}, id {landmark_id} due to missing X or Y.")
                        # else:
                            # print(f"Skipping frame object (index {frame_idx}) in list due to missing or malformed 'Landmarks' list.")
                else:
                    QMessageBox.warning(self, "Invalid JSON Structure", "The JSON file is not a dictionary with a 'frames' list, nor a direct list of frames. Please check the format.")
                    return

                if not all_landmarks_data:
                    QMessageBox.warning(self, "No Data Parsed", "Could not parse any landmark data from the JSON file. Ensure the file structure contains 'frames' with 'phase' and 'landmarks' (each with 'x', 'y').")
                    return

                df_skeleton = pd.DataFrame(all_landmarks_data)
                
                # If Shot_Number is missing or mostly null, try to infer it from frame ranges
                if 'Shot_Number' not in df_skeleton.columns or df_skeleton['Shot_Number'].isna().sum() > len(df_skeleton) * 0.5:
                    print("Shot_Number not found or mostly null, attempting to infer from frame ranges...")
                    # Group frames by player state changes to infer shots
                    if 'Player_State' in df_skeleton.columns:
                        df_skeleton = df_skeleton.sort_values('Frame_Number')
                        df_skeleton['Shot_Number'] = 1  # Default to shot 1
                        
                        # Try to infer shots based on state transitions
                        state_changes = df_skeleton['Player_State'].ne(df_skeleton['Player_State'].shift()).cumsum()
                        df_skeleton['Shot_Number'] = state_changes
                        
                        # Alternative: group every N frames as a shot (if state-based grouping doesn't work well)
                        if df_skeleton['Shot_Number'].nunique() < 2:  # If only one shot detected
                            frames_per_shot = 100  # Assume ~100 frames per shot
                            df_skeleton['Shot_Number'] = (df_skeleton['Frame_Number'] // frames_per_shot) + 1
                
                missing_cols = [col for col in SKELETON_REQUIRED_COLUMNS if col not in df_skeleton.columns]
                if missing_cols:
                    # Check if it's just Player_State that's missing and all_landmarks_data was populated
                    if missing_cols == ['Player_State'] and not df_skeleton.empty:
                         QMessageBox.warning(self, "Missing Player State", "Parsed landmark data, but 'Player_State' (derived from 'phase') is missing for some/all frames. It will be NaN.")
                         df_skeleton['Player_State'] = None # Ensure column exists if completely missing
                    else:
                        QMessageBox.warning(self, "Invalid Parsed Data", f"After parsing, data is missing for: {', '.join(missing_cols)}. Check JSON keys and structure (e.g., 'frames', 'phase', 'landmarks', 'x', 'y').")
                        return
                
                # Ensure correct data types, especially for numeric columns
                try:
                    for col in ['Frame_Number', 'Landmark_ID', 'Landmark_X', 'Landmark_Y', 'Shot_Number']:
                        if col in df_skeleton.columns:
                            df_skeleton[col] = pd.to_numeric(df_skeleton[col], errors='coerce') # Coerce errors to NaN
                    # Drop rows where essential numeric data became NaN after coercion, if necessary
                    # df_skeleton.dropna(subset=['Frame_Number', 'Landmark_ID', 'Landmark_X', 'Landmark_Y'], inplace=True)

                except Exception as e:
                    QMessageBox.warning(self, "Data Type Error", f"Could not convert landmark data to numeric types: {e}")
                    return
                
                if df_skeleton.empty and all_landmarks_data: # If all rows were dropped by coerce/dropna
                    QMessageBox.warning(self, "No Valid Numeric Data", "Parsed records, but essential numeric landmark data (Frame_Number, Landmark_ID, X, Y) was invalid or missing, resulting in no usable data.")
                    return
                elif df_skeleton.empty: # If all_landmarks_data was empty or df became empty for other reasons
                    QMessageBox.warning(self, "No Usable Data", "No usable skeleton data could be extracted from the file.")
                    return


                self.skeleton_file_counter += 1
                base_name = os.path.basename(filepath)
                unique_name = f"Skel_{self.skeleton_file_counter:02d}_{base_name}"
                
                self.skeleton_data_store[unique_name] = df_skeleton
                self.original_skeleton_names[unique_name] = base_name
                self.update_loaded_skeleton_files_display()
                QMessageBox.information(self, "Success", f"Skeleton file '{base_name}' loaded and parsed.")
            except json.JSONDecodeError as jde:
                QMessageBox.critical(self, "JSON Decode Error", f"Failed to decode JSON file: {str(jde)}\nPlease ensure it's a valid JSON format.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load or process skeleton file: {str(e)}")

    def update_loaded_skeleton_files_display(self):
        current_selection = self.loaded_skeleton_files_list.currentItem().text() if self.loaded_skeleton_files_list.currentItem() else None
        self.loaded_skeleton_files_list.clear()
        for unique_name in self.skeleton_data_store.keys():
            self.loaded_skeleton_files_list.addItem(unique_name)
        
        if current_selection and self.loaded_skeleton_files_list.findItems(current_selection, Qt.MatchExactly):
            self.loaded_skeleton_files_list.setCurrentItem(self.loaded_skeleton_files_list.findItems(current_selection, Qt.MatchExactly)[0])
        elif self.loaded_skeleton_files_list.count() > 0:
            self.loaded_skeleton_files_list.setCurrentRow(0)

    def on_loaded_skeleton_selection_change(self):
        self.reset_skeleton_playback() 
        self.populate_skeleton_player_states()
        
        # Update virtual target when skeleton file changes with debouncing
        self._debounced_skeleton_virtual_target_update()

    def populate_skeleton_player_states(self):
        self.skeleton_player_state_combo.blockSignals(True)
        self.skeleton_shot_number_combo.blockSignals(True)
        
        self.skeleton_player_state_combo.clear()
        self.skeleton_shot_number_combo.clear()
        
        selected_items = self.loaded_skeleton_files_list.selectedItems()
        
        if not selected_items:
            self.skeleton_player_state_combo.addItem("All") # Default if no file selected
            self.skeleton_shot_number_combo.addItem("All") # Default if no file selected
            self.skeleton_player_state_combo.blockSignals(False)
            self.skeleton_shot_number_combo.blockSignals(False)
            self.prepare_skeleton_data_for_playback() 
            return
        
        selected_file_key = selected_items[0].text()
        if selected_file_key in self.skeleton_data_store:
            df_skeleton = self.skeleton_data_store[selected_file_key]
            
            # Populate shot numbers if available
            if 'Shot_Number' in df_skeleton.columns:
                shot_numbers = sorted(df_skeleton['Shot_Number'].unique().tolist())
                self.skeleton_shot_number_combo.addItems(["All"] + [str(sn) for sn in shot_numbers])
            else:
                self.skeleton_shot_number_combo.addItem("All")
            
            # Populate player states if available
            if 'Player_State' in df_skeleton.columns:
                states = df_skeleton['Player_State'].unique().tolist()
                self.skeleton_player_state_combo.addItems(["All"] + sorted(states))
            else:
                self.skeleton_player_state_combo.addItem("All") 
        else: # Should not happen if list is populated correctly
            self.skeleton_player_state_combo.addItem("All")
            self.skeleton_shot_number_combo.addItem("All")
        
        self.skeleton_player_state_combo.setCurrentIndex(0) 
        self.skeleton_shot_number_combo.setCurrentIndex(0)
        self.skeleton_player_state_combo.blockSignals(False)
        self.skeleton_shot_number_combo.blockSignals(False)
        self.prepare_skeleton_data_for_playback()

    def on_skeleton_shot_number_change(self):
        self.reset_skeleton_playback()
        self.prepare_skeleton_data_for_playback()
        
        # Update virtual target with selected shot using debouncing
        selected_shot = self.skeleton_shot_number_combo.currentText()
        if selected_shot != "All":
            try:
                shot_num = int(selected_shot)
                self._debounced_skeleton_virtual_target_update(shot_num)
            except ValueError:
                self._debounced_skeleton_virtual_target_update()
        else:
            self._debounced_skeleton_virtual_target_update()

    def on_skeleton_player_state_change(self):
        self.reset_skeleton_playback()
        self.prepare_skeleton_data_for_playback()
        
        # Update virtual target with current shot selection using debouncing
        selected_shot = self.skeleton_shot_number_combo.currentText()
        if selected_shot != "All":
            try:
                shot_num = int(selected_shot)
                self._debounced_skeleton_virtual_target_update(shot_num)
            except ValueError:
                self._debounced_skeleton_virtual_target_update()
        else:
            self._debounced_skeleton_virtual_target_update()

    def prepare_skeleton_data_for_playback(self):
        self.current_skeleton_df_for_playback = None
        self.unique_frame_numbers_for_playback = []
        self.total_skeleton_frames = 0
        self.current_skeleton_frame_index = 0

        selected_files = self.loaded_skeleton_files_list.selectedItems()
        if not selected_files:
            self.skeleton_video_controls.set_total_frames(0)
            self.skeleton_display_widget.set_image(None)
            return

        selected_file_key = selected_files[0].text()
        selected_player_state = self.skeleton_player_state_combo.currentText()
        selected_shot_number = self.skeleton_shot_number_combo.currentText()

        if selected_file_key not in self.skeleton_data_store:
            self.skeleton_video_controls.set_total_frames(0)
            self.skeleton_display_widget.set_image(None)
            return
            
        df_full_skeleton = self.skeleton_data_store[selected_file_key]
        
        temp_df = df_full_skeleton.copy()
        
        # Apply shot number filter
        if selected_shot_number != "All" and 'Shot_Number' in temp_df.columns:
            try:
                shot_num = int(selected_shot_number)
                temp_df = temp_df[temp_df['Shot_Number'] == shot_num]
            except ValueError:
                pass  # If conversion fails, don't filter by shot number
        
        # Apply player state filter
        if selected_player_state != "All" and 'Player_State' in temp_df.columns:
            temp_df = temp_df[temp_df['Player_State'] == selected_player_state]

        if temp_df.empty or 'Frame_Number' not in temp_df.columns:
            self.skeleton_video_controls.set_total_frames(0)
            self.skeleton_display_widget.set_image(None)
            return

        self.unique_frame_numbers_for_playback = sorted(temp_df['Frame_Number'].unique())
        self.total_skeleton_frames = len(self.unique_frame_numbers_for_playback)
        # Store the filtered DataFrame that contains all landmarks for the relevant frames
        self.current_skeleton_df_for_playback = temp_df[temp_df['Frame_Number'].isin(self.unique_frame_numbers_for_playback)]
        
        self.skeleton_video_controls.set_total_frames(self.total_skeleton_frames)
        if self.total_skeleton_frames > 0:
            # Calculate and set phase markers for skeleton analysis
            self.calculate_skeleton_phase_markers()
            self.draw_current_skeleton_frame()
        else:
            self.skeleton_display_widget.set_image(None)

    def calculate_skeleton_phase_markers(self):
        """Calculate phase boundaries for skeleton analysis and set markers"""
        if self.current_skeleton_df_for_playback is None or self.total_skeleton_frames == 0:
            self.skeleton_video_controls.set_phase_markers([])
            return
        
        phase_data = []
        current_phase = None
        phase_start_frame = 0
        
        for frame_idx in range(self.total_skeleton_frames):
            frame_number = self.unique_frame_numbers_for_playback[frame_idx]
            frame_data = self.current_skeleton_df_for_playback[
                self.current_skeleton_df_for_playback['Frame_Number'] == frame_number
            ]
            
            if not frame_data.empty:
                frame_phase = frame_data.iloc[0].get('Player_State', 'unknown')
                
                # If phase changes, record the previous phase
                if current_phase is not None and frame_phase != current_phase:
                    phase_data.append({
                        'start_frame': phase_start_frame,
                        'end_frame': frame_idx,
                        'phase_name': current_phase
                    })
                    phase_start_frame = frame_idx
                
                # Update current phase
                current_phase = frame_phase
        
        # Add the last phase
        if current_phase is not None:
            phase_data.append({
                'start_frame': phase_start_frame,
                'end_frame': self.total_skeleton_frames,
                'phase_name': current_phase
            })
        
        # Set the phase markers on the slider
        self.skeleton_video_controls.set_phase_markers(phase_data, self.total_skeleton_frames)

    def draw_current_skeleton_frame(self):
        if self.current_skeleton_df_for_playback is None or \
           self.total_skeleton_frames == 0 or \
           not (0 <= self.current_skeleton_frame_index < self.total_skeleton_frames):
            self.skeleton_display_widget.set_image(None) 
            if hasattr(self, 'skeleton_video_controls'): # Check if controls exist
                 self.skeleton_video_controls.update_frame_display(self.current_skeleton_frame_index, self.total_skeleton_frames)
            return

        target_frame_number = self.unique_frame_numbers_for_playback[self.current_skeleton_frame_index]
        # Filter the already filtered df for the specific frame number
        frame_data_df = self.current_skeleton_df_for_playback[
            self.current_skeleton_df_for_playback['Frame_Number'] == target_frame_number
        ]

        img_width, img_height = self.skeleton_display_widget.width(), self.skeleton_display_widget.height()
        if img_width <=0 or img_height <=0 : # Fallback if widget not sized yet
            img_width, img_height = 640, 480

        image = QImage(img_width, img_height, QImage.Format_RGB888)
        image.fill(Qt.black)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate aspect ratio preservation for skeleton
        # Assume skeleton coordinates are in 0-1 range and need proper scaling
        skeleton_aspect_ratio = 4/3  # Standard skeleton aspect ratio (width/height)
        widget_aspect_ratio = img_width / img_height if img_height > 0 else 1
        
        if widget_aspect_ratio > skeleton_aspect_ratio:
            # Widget is wider - fit to height
            scale_factor = img_height
            x_offset = (img_width - img_height * skeleton_aspect_ratio) / 2
            y_offset = 0
            skeleton_width = img_height * skeleton_aspect_ratio
            skeleton_height = img_height
        else:
            # Widget is taller - fit to width  
            scale_factor = img_width / skeleton_aspect_ratio
            x_offset = 0
            y_offset = (img_height - img_width / skeleton_aspect_ratio) / 2
            skeleton_width = img_width
            skeleton_height = img_width / skeleton_aspect_ratio

        landmarks_for_frame = {}
        if not frame_data_df.empty and \
           all(col in frame_data_df.columns for col in ['Landmark_ID', 'Landmark_X', 'Landmark_Y']):
            for _, row in frame_data_df.iterrows():
                try:
                    lm_id = int(row['Landmark_ID'])
                    # Apply aspect ratio preserving transformation
                    x = row['Landmark_X'] * skeleton_width + x_offset
                    y = row['Landmark_Y'] * skeleton_height + y_offset
                    if pd.isna(x) or pd.isna(y) or not np.isfinite(x) or not np.isfinite(y):
                        continue 
                    landmarks_for_frame[lm_id] = (int(x), int(y))
                except ValueError:
                    continue 

        painter.setPen(QPen(Qt.white, 2))
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            if p1_idx in landmarks_for_frame and p2_idx in landmarks_for_frame:
                pt1 = landmarks_for_frame[p1_idx]
                pt2 = landmarks_for_frame[p2_idx]
                painter.drawLine(pt1[0], pt1[1], pt2[0], pt2[1])
        
        painter.setPen(QPen(Qt.cyan, 1))
        painter.setBrush(Qt.cyan)
        for lm_id, (x,y) in landmarks_for_frame.items():
            painter.drawEllipse(x-3, y-3, 6, 6) 

        painter.end()
        self.skeleton_display_widget.set_image(image) 
        if hasattr(self, 'skeleton_video_controls'):
            self.skeleton_video_controls.update_frame_display(self.current_skeleton_frame_index, self.total_skeleton_frames)


    def toggle_skeleton_playback(self, checked): 
        if self.total_skeleton_frames == 0:
            self.skeleton_video_controls.play_pause_btn.setChecked(False)
            self.skeleton_video_controls.play_pause_btn.setText("▶")
            return

        if checked: 
            self.skeleton_video_controls.play_pause_btn.setText("⏸")
            current_speed_factor = self.skeleton_video_controls.speed_slider.value() / 100.0
            interval = int(1000 / (self.skeleton_fps * current_speed_factor))
            self.skeleton_playback_timer.start(max(20, interval)) 
        else: 
            self.skeleton_video_controls.play_pause_btn.setText("▶")
            self.skeleton_playback_timer.stop()

    def reset_skeleton_playback(self):
        self.skeleton_playback_timer.stop()
        self.skeleton_video_controls.play_pause_btn.setChecked(False) 
        self.skeleton_video_controls.play_pause_btn.setText("▶")      
        self.current_skeleton_frame_index = 0
        if self.total_skeleton_frames > 0:
            self.draw_current_skeleton_frame()
        else: 
            self.skeleton_display_widget.set_image(None)
            if hasattr(self, 'skeleton_video_controls'):
                self.skeleton_video_controls.update_frame_display(0,0)


    def set_skeleton_playback_speed(self, value): 
        if self.skeleton_playback_timer.isActive():
            current_speed_factor = value / 100.0
            interval = int(1000 / (self.skeleton_fps * current_speed_factor))
            self.skeleton_playback_timer.setInterval(max(20, interval))

    def skeleton_playback_slider_moved(self, frame_index):
        if 0 <= frame_index < self.total_skeleton_frames:
            self.current_skeleton_frame_index = frame_index
            self.draw_current_skeleton_frame()
            if self.skeleton_video_controls.play_pause_btn.isChecked(): 
                self.skeleton_playback_timer.stop()
                QTimer.singleShot(50, lambda: self.skeleton_playback_timer.start() if self.skeleton_video_controls.play_pause_btn.isChecked() else None)


    def skeleton_playback_slider_value_changed(self, frame_index):
        if not self.skeleton_video_controls.progress_slider.isSliderDown(): 
            if 0 <= frame_index < self.total_skeleton_frames:
                self.current_skeleton_frame_index = frame_index
                self.draw_current_skeleton_frame()

    def seek_skeleton_relative(self, frame_offset):
        if self.total_skeleton_frames == 0:
            return
        
        new_index = self.current_skeleton_frame_index + frame_offset
        self.current_skeleton_frame_index = max(0, min(new_index, self.total_skeleton_frames - 1))
        
        self.draw_current_skeleton_frame()
        if hasattr(self, 'skeleton_video_controls'):
            self.skeleton_video_controls.progress_slider.setValue(self.current_skeleton_frame_index)

        if self.skeleton_video_controls.play_pause_btn.isChecked(): 
            self.skeleton_playback_timer.stop()
            QTimer.singleShot(50, lambda: self.skeleton_playback_timer.start() if self.skeleton_video_controls.play_pause_btn.isChecked() else None)


    def advance_skeleton_frame(self):
        if self.total_skeleton_frames == 0: return

        self.current_skeleton_frame_index += 1
        if self.current_skeleton_frame_index >= self.total_skeleton_frames:
            self.current_skeleton_frame_index = 0 
            if not self.skeleton_video_controls.play_pause_btn.isChecked(): # If was single stepping, stop
                 pass # Or if looping is not desired, stop timer: self.toggle_skeleton_playback(False)
            elif self.skeleton_video_controls.play_pause_btn.isChecked(): # If playing, continue loop
                 pass
            else: # Default to stopping if not explicitly looping while playing
                 self.toggle_skeleton_playback(False)
                 # return # uncomment to stop at end instead of looping

        self.draw_current_skeleton_frame()

    def toggle_skeleton_drawing_mode(self, checked):
        self.skeleton_display_widget.toggle_drawing(checked)

    def set_skeleton_drawing_shape(self, shape_name):
        self.skeleton_display_widget.set_drawing_tool(shape_name)

    def set_skeleton_drawing_color(self, color_name):
        self.skeleton_display_widget.set_drawing_color(color_name.lower())

    def clear_skeleton_drawings(self):
        self.skeleton_display_widget.clear_drawings()

    # --- Methods for Timing Analysis (mostly unchanged) ---
    def load_centralized_skeleton_data(self):
        """Load skeleton data centrally for all tabs"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Skeleton JSON", "", "JSON Files (*.json)")
        if not filepath:
            return
        
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Accepts both {"frames": [...]} and just a list
            frames = data.get("frames", data) if isinstance(data, dict) else data
            
            # Group frames by shot number (if available), else by phase or just index
            frames_by_shot = {}
            for idx, frame in enumerate(frames):
                # Try to get shot number from frame metadata, fallback to index
                shot_num = None
                if "shot_number" in frame:
                    shot_num = int(frame["shot_number"])
                elif "Shot_Number" in frame:
                    shot_num = int(frame["Shot_Number"])
                elif "phase" in frame and frame["phase"] is not None:
                    shot_num = str(frame["phase"])
                else:
                    shot_num = int(idx // 100 + 1)  # crude fallback: every 100 frames = new shot
                frames_by_shot.setdefault(shot_num, []).append(frame)
            
            # Store centrally
            self.centralized_shot_playback_json_data = frames
            self.centralized_shot_playback_frames_by_shot = frames_by_shot
            
            # Update status
            self.skeleton_status_label.setText(f"✅ Skeleton data loaded ({len(frames)} frames)")
            self.skeleton_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
            
            # Update data status
            self.update_data_status()
            
            QMessageBox.information(self, "Success", f"Skeleton data loaded successfully!\n{len(frames)} frames across {len(frames_by_shot)} shots")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load skeleton data: {str(e)}")
            self.skeleton_status_label.setText("❌ Failed to load skeleton data")
            self.skeleton_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 10px;")

    def load_centralized_shot_playback_data(self):
        """Load shot playback data centrally for all tabs"""
        # First check if we have data from the main system that we can use
        if hasattr(self, 'combined_df') and self.combined_df is not None:
            # Check if the combined data has the required columns for shot playback
            required_columns = ['Shot_Number', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
            if all(col in self.combined_df.columns for col in required_columns):
                # Use the existing data
                self.centralized_shot_playback_csv_df = self.combined_df.copy()
                self.shot_playback_status_label.setText(f"✅ Using existing data ({len(self.centralized_shot_playback_csv_df)} shots)")
                self.shot_playback_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
                self.update_data_status()
                QMessageBox.information(self, "Data Loaded", 
                                      f"Using existing data with {len(self.centralized_shot_playback_csv_df)} shots for shot playback analysis.")
                return
        
        # If no suitable data is available, prompt for file selection
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Shot CSV", "", "CSV Files (*.csv)")
        if not filepath:
            return
        
        try:
            import pandas as pd
            self.centralized_shot_playback_csv_df = pd.read_csv(filepath)
            
            # Update status
            self.shot_playback_status_label.setText(f"✅ Shot playback data loaded ({len(self.centralized_shot_playback_csv_df)} shots)")
            self.shot_playback_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
            
            # Update data status
            self.update_data_status()
            
            QMessageBox.information(self, "Success", f"Shot playback data loaded successfully!\n{len(self.centralized_shot_playback_csv_df)} shots loaded")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load shot playback data: {str(e)}")
            self.shot_playback_status_label.setText("❌ Failed to load shot playback data")
            self.shot_playback_status_label.setStyleSheet("color: #f44336; font-weight: bold; font-size: 10px;")

    def update_data_status(self):
        """Update the data status based on what's loaded"""
        csv_loaded = hasattr(self, 'combined_df') and self.combined_df is not None
        json_loaded = hasattr(self, 'shot_playback_json_data') and self.shot_playback_json_data is not None
        
        # Update CSV status
        if csv_loaded:
            self.csv_status_label.setText(f"✅ CSV loaded ({len(self.combined_df)} shots)")
            self.csv_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.csv_status_label.setText("No CSV file loaded")
            self.csv_status_label.setStyleSheet("color: #666; font-style: italic;")
        
        # Update JSON status
        if json_loaded:
            self.json_status_label.setText(f"✅ JSON loaded ({len(self.shot_playback_json_data)} frames)")
            self.json_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.json_status_label.setText("No JSON file loaded")
            self.json_status_label.setStyleSheet("color: #666; font-style: italic;")
        
        # Update overall status
        if csv_loaded and json_loaded:
            self.data_status_label.setText("✅ All data loaded - Ready to analyze!")
            self.data_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px; background-color: #E8F5E8;")
        elif csv_loaded or json_loaded:
            loaded_types = []
            if csv_loaded:
                loaded_types.append("CSV")
            if json_loaded:
                loaded_types.append("JSON")
            self.data_status_label.setText(f"⚠️ Partial data loaded ({', '.join(loaded_types)})")
            self.data_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px; background-color: #FFF3E0;")
        else:
            self.data_status_label.setText("📋 Ready to load data")
            self.data_status_label.setStyleSheet("color: #2196F3; font-weight: bold; padding: 5px; background-color: #E3F2FD;")

    def auto_load_centralized_data_to_tabs(self):
        """Automatically load centralized data to all tabs when they're accessed"""
        # Load skeleton data to skeleton analysis tab
        if hasattr(self, 'centralized_shot_playback_json_data') and self.centralized_shot_playback_json_data is not None:
            if hasattr(self, 'skeleton_data_by_shot'):
                self.skeleton_data_by_shot = self.centralized_shot_playback_frames_by_shot.copy()
                print(f"Auto-loaded centralized skeleton data to skeleton analysis tab: {len(self.centralized_shot_playback_json_data)} frames")
        
        # Load data to shot playback tabs
        if hasattr(self, 'centralized_shot_playback_csv_df') and self.centralized_shot_playback_csv_df is not None:
            # Update both shot playback tabs
            if hasattr(self, 'shot_playback_csv_df'):
                self.shot_playback_csv_df = self.centralized_shot_playback_csv_df.copy()
                print(f"Auto-loaded centralized CSV data to shot playback tab: {len(self.centralized_shot_playback_csv_df)} shots")
            
            if hasattr(self, 'update_shot_playback_shot_list'):
                self.update_shot_playback_shot_list()
        
        if hasattr(self, 'centralized_shot_playback_json_data') and self.centralized_shot_playback_json_data is not None:
            if hasattr(self, 'shot_playback_json_data'):
                self.shot_playback_json_data = self.centralized_shot_playback_json_data
                self.shot_playback_frames_by_shot = self.centralized_shot_playback_frames_by_shot.copy()
                print(f"Auto-loaded centralized JSON data to shot playback tab: {len(self.centralized_shot_playback_json_data)} frames")

    def on_tab_changed(self, index):
        """Handle tab changes and auto-load centralized data"""
        tab_name = self.tab_widget.tabText(index)
        print(f"Switched to tab: {tab_name}")
        
        # Auto-load centralized data when switching to relevant tabs
        if tab_name in ["Skeleton Analysis", "Shot Playback Analysis"]:
            self.auto_load_centralized_data_to_tabs()

    def load_file(self):
        """Load CSV file for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        
        if file_path:
            try:
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Store the data directly (no multi-session complexity)
                self.combined_df = df
                
                # Create visualizer for charts
                self.visualizer = ShotTimingVisualizer(df)
                
                # Clear highlights when new data is loaded
                if hasattr(self, 'clear_all_highlights'):
                    self.clear_all_highlights()
                
                # Update UI
                self.update_data_info()
                self.populate_filters()
                self.populate_data_table()
                self.enable_controls()
                self.update_data_status()
                
                QMessageBox.information(self, "Success", f"CSV file loaded successfully!\n{len(df)} shots loaded")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def remove_session(self):
        """Remove selected session"""
        current_item = self.sessions_list.currentItem()
        if current_item:
            session_name = current_item.text()
            if session_name in self.sessions_data:
                del self.sessions_data[session_name]
                del self.sessions_info[session_name]
                
                # Update the sessions list
                self.update_sessions_list()
                
                # Recombine sessions
                self.combine_sessions()
                
                # Update UI
                self.update_data_info()
                self.populate_filters()
                self.populate_data_table()
                
                QMessageBox.information(self, "Success", f"Session '{session_name}' removed!")
    
    def clear_all_sessions(self):
        """Clear all loaded sessions"""
        if self.sessions_data:
            reply = QMessageBox.question(self, "Clear All Sessions", 
                                       "Are you sure you want to clear all loaded sessions?",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.sessions_data.clear()
                self.sessions_info.clear()
                self.combined_df = None
                self.visualizer = None
                
                # Clear highlights when sessions are cleared
                if hasattr(self, 'clear_all_highlights'):
                    self.clear_all_highlights()
                
                # Update UI
                self.update_sessions_list()
                self.update_data_info()
                self.populate_filters()
                self.populate_data_table()
                self.disable_controls()
                
                QMessageBox.information(self, "Success", "All sessions cleared!")
    
    def update_sessions_list(self):
        """Update the sessions list display"""
        self.sessions_list.clear()
        for session_name in self.sessions_data.keys():
            item = QListWidgetItem(session_name)
            self.sessions_list.addItem(item)
        
        # Enable/disable session management buttons
        has_sessions = len(self.sessions_data) > 0
        self.remove_session_btn.setEnabled(has_sessions)
        self.clear_all_btn.setEnabled(has_sessions)
    
    def combine_sessions(self):
        """Combine all loaded sessions into one dataframe"""
        if not self.sessions_data:
            self.combined_df = None
            self.visualizer = None
            return
        
        # Combine all dataframes
        dfs = []
        for session_name, df in self.sessions_data.items():
            df_copy = df.copy()
            # Add session identifier if not present
            if 'Session' not in df_copy.columns:
                df_copy['Session'] = session_name
            dfs.append(df_copy)
        
        # Combine all dataframes
        self.combined_df = pd.concat(dfs, ignore_index=True)
        self.visualizer = ShotTimingVisualizer(self.combined_df)
    
    def disable_controls(self):
        """Disable control groups when no data is loaded"""
        self.filter_group.setEnabled(False)
        self.viz_group.setEnabled(False)
        self.export_group.setEnabled(False)
        self.save_data_btn.setEnabled(False)
    def update_data_info(self):
        """Update data information display"""
        if self.combined_df is not None:
            total_shots = len(self.combined_df)
            
            info_text = f"""
            Total Shots: {total_shots}
            Columns: {len(self.combined_df.columns)}
            Players: {', '.join(self.combined_df['Player'].unique()) if 'Player' in self.combined_df.columns else 'N/A'}
            Sessions: {', '.join(self.combined_df['Session'].unique()) if 'Session' in self.combined_df.columns else 'N/A'}
            Shot Range: {self.combined_df['Shot_Number'].min()} - {self.combined_df['Shot_Number'].max() if 'Shot_Number' in self.combined_df.columns else 'N/A'}
            """
            print(info_text)  # Print to console for debugging
        else:
            print("No data loaded")
    
    def populate_filters(self):
        """Populate filter dropdowns"""
        if self.combined_df is not None:
            # Player filter
            self.player_combo.clear()
            self.player_combo.addItem("All Players")
            if 'Player' in self.combined_df.columns:
                self.player_combo.addItems(self.combined_df['Player'].unique().tolist())
            
            # Session filter
            self.session_combo.clear()
            self.session_combo.addItem("All Sessions")
            if 'Session' in self.combined_df.columns:
                self.session_combo.addItems(self.combined_df['Session'].unique().tolist())
            
            # Shot range
            if 'Shot_Number' in self.combined_df.columns:
                shot_min = self.combined_df['Shot_Number'].min()
                shot_max = self.combined_df['Shot_Number'].max()
                self.shot_min_spin.setRange(shot_min, shot_max)
                self.shot_max_spin.setRange(shot_min, shot_max)
                self.shot_min_spin.setValue(shot_min)
                self.shot_max_spin.setValue(shot_max)
            
            # Score range
            if 'Score' in self.combined_df.columns:
                # Convert Score column to numeric for proper min/max calculation
                score_series = pd.to_numeric(self.combined_df['Score'], errors='coerce')
                score_min = score_series.min()
                score_max = score_series.max()
                # Handle case where all scores are NaN
                if pd.isna(score_min) or pd.isna(score_max):
                    score_min = 0
                    score_max = 10
                self.score_min_spin.setRange(score_min, score_max)
                self.score_max_spin.setRange(score_min, score_max)
                self.score_min_spin.setValue(score_min)
                self.score_max_spin.setValue(score_max)
    
    def create_direction_dropdown(self, current_value=""):
        """Create a dropdown widget for direction selection"""
        from PyQt5.QtWidgets import QComboBox
        
        combo = QComboBox()
        combo.addItem("")  # Empty option
        combo.addItems(["N", "S", "E", "W", "NE", "NW", "SE", "SW"])
        
        # Set current value if provided
        if current_value:
            index = combo.findText(current_value)
            if index >= 0:
                combo.setCurrentIndex(index)
        
        return combo

    def populate_data_table_with_filtered_data(self, df_to_display):
        # Add Score column if not present
        if 'Score' not in df_to_display.columns:
            df_to_display = df_to_display.copy()
            df_to_display['Score'] = ""  # or np.nan

        # Add Direction column if not present
        if 'Direction' not in df_to_display.columns:
            df_to_display = df_to_display.copy()
            df_to_display['Direction'] = ""  # Empty direction

        self.data_table.setRowCount(0)
        if df_to_display is not None and not df_to_display.empty:
            self.data_table.setRowCount(len(df_to_display))
            self.data_table.setColumnCount(len(df_to_display.columns))
            self.data_table.setHorizontalHeaderLabels(df_to_display.columns.tolist())

            # Get session colors if multiple sessions
            session_colors = {}
            if 'Session' in df_to_display.columns and len(df_to_display['Session'].unique()) > 1:
                sessions = df_to_display['Session'].unique()
                for i, session in enumerate(sessions):
                    session_colors[session] = SESSION_COLORS[i % len(SESSION_COLORS)]
            
            for i in range(len(df_to_display)):
                for j, col in enumerate(df_to_display.columns):
                    if col == "Direction":
                        # Create dropdown for direction
                        direction_combo = self.create_direction_dropdown(str(df_to_display.iloc[i, j]))
                        direction_combo.currentTextChanged.connect(
                            lambda text, row=i, col=j: self.on_direction_changed(row, col, text)
                        )
                        self.data_table.setCellWidget(i, j, direction_combo)
                    else:
                        item = QTableWidgetItem(str(df_to_display.iloc[i, j]))
                        if col == "Score":
                            item.setFlags(item.flags() | Qt.ItemIsEditable)
                        
                        # Add session color background if multiple sessions
                        if col == "Session" and session_colors:
                            session = str(df_to_display.iloc[i, j])
                            if session in session_colors:
                                item.setBackground(QColor(session_colors[session]))
                                item.setForeground(QColor('white'))  # White text for contrast
                        
                        self.data_table.setItem(i, j, item)

            self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            self.data_table.setColumnCount(len(self.combined_df.columns) if self.combined_df is not None else 0)
            self.data_table.setHorizontalHeaderLabels(self.combined_df.columns.tolist() if self.combined_df is not None else [])

        # Connect cellChanged to update DataFrame
        self.data_table.cellChanged.connect(self.on_score_edited)
    
    def populate_data_table(self):
        """Populate data table with initial full data"""
        if self.combined_df is not None:
            self.populate_data_table_with_filtered_data(self.combined_df)
    
    def enable_controls(self):
        """Enable control groups after data is loaded"""
        self.filter_group.setEnabled(True)
        self.viz_group.setEnabled(True)
        self.export_group.setEnabled(True)
        self.save_data_btn.setEnabled(True)  # Enable save button when data is loaded
    
    def _debounced_apply_filters(self):
        """Debounced version of apply_filters to prevent infinite loops"""
        # Stop any existing timer
        if hasattr(self, 'main_filter_debounce_timer'):
            self.main_filter_debounce_timer.stop()
        else:
            # Create timer if it doesn't exist
            self.main_filter_debounce_timer = QTimer()
            self.main_filter_debounce_timer.setSingleShot(True)
            self.main_filter_debounce_timer.timeout.connect(self.apply_filters)
        
        # Start the timer with 500ms delay
        self.main_filter_debounce_timer.start(500)

    def apply_filters(self):
        """Apply selected filters to data"""
        if self.combined_df is None:
            return
        
        filtered_df = self.combined_df.copy()
        
        # Apply player filter
        if self.player_combo.currentText() != "All Players":
            filtered_df = filtered_df[filtered_df['Player'] == self.player_combo.currentText()]
        
        # Apply session filter
        if self.session_combo.currentText() != "All Sessions":
            filtered_df = filtered_df[filtered_df['Session'] == self.session_combo.currentText()]
        
        # Apply shot range filter
        if 'Shot_Number' in filtered_df.columns:
            shot_min = self.shot_min_spin.value()
            shot_max = self.shot_max_spin.value()
            filtered_df = filtered_df[
                (filtered_df['Shot_Number'] >= shot_min) & 
                (filtered_df['Shot_Number'] <= shot_max)
            ]

        # Apply score range filter
        if 'Score' in filtered_df.columns:
            score_min = self.score_min_spin.value()
            score_max = self.score_max_spin.value()
            # Convert Score column to numeric, handling errors gracefully
            filtered_df['Score'] = pd.to_numeric(filtered_df['Score'], errors='coerce')
            # Filter only rows where Score is not NaN (valid numeric values)
            filtered_df = filtered_df.dropna(subset=['Score'])
            filtered_df = filtered_df[
                (filtered_df['Score'] >= score_min) & 
                (filtered_df['Score'] <= score_max)
            ]

        # Apply complete shot filter
        if 'Complete_Shot' in filtered_df.columns and self.complete_shot_checkbox.isChecked():
            filtered_df = filtered_df[filtered_df['Complete_Shot'].str.lower() == 'yes']
        
        self.visualizer.df = filtered_df
        # Update the data table view after filtering
        self.populate_data_table_with_filtered_data(filtered_df)
        # Regenerate chart if one is selected
        if self.chart_list.currentItem():
            self.generate_selected_chart()

    def update_visualization(self):
        """Update visualization based on selection"""
        if self.chart_list.currentItem():
            self.generate_selected_chart()
    
    def generate_selected_chart(self):
        """Generate the selected chart type"""
        if not self.chart_list.currentItem() or self.visualizer is None:
            return
        
        chart_type = self.chart_list.currentItem().text()
        
        # Get scaling settings
        scaling_settings = self.get_scaling_settings()
        
        # Clear previous plots
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)
        
        try:
            if chart_type == "Small-Multiples Line Charts":
                fig = self.visualizer.small_multiples_line_charts(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Stacked Bar Composition":
                fig = self.visualizer.stacked_bars_shot_composition(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Box-Violin Distribution":
                fig = self.visualizer.box_violin_plots(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Phase Duration Heatmap":
                fig = self.visualizer.heatmap_phase_durations(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Side-by-Side Bar Chart":
                fig = self.visualizer.side_by_side_bar_chart(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Interactive Dashboard":
                fig = self.visualizer.create_interactive_plotly()
                html_string = pio.to_html(fig, include_plotlyjs='cdn')
                self.web_view.setHtml(html_string)
                self.tab_widget.setCurrentIndex(1)  # Switch to interactive tab
                
            elif chart_type == "Score vs Durations":
                fig = self.visualizer.score_vs_durations(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Virtual Target Visualization":
                # Get zoom setting
                zoom_target = self.zoom_target_checkbox.isChecked()
                fig = self.visualizer.virtual_target_visualization(scaling_settings=scaling_settings, zoom_target=zoom_target)
                self.add_matplotlib_figure(fig)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate chart: {str(e)}")
    
    def add_matplotlib_figure(self, fig):
        """Add matplotlib figure to the layout"""
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(600)
        self.plot_layout.addWidget(canvas)
        self.tab_widget.setCurrentIndex(0)  # Switch to static charts tab
    
    def export_png(self):
        """Export current matplotlib figure as PNG"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG Files (*.png)")
        if file_path and self.plot_layout.count() > 0:
            try:
                canvas = self.plot_layout.itemAt(0).widget()
                if isinstance(canvas, FigureCanvas):
                    canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, "Success", "Chart exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def export_html(self):
        """Export interactive dashboard as HTML"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HTML", "", "HTML Files (*.html)")
        if file_path and self.visualizer:
            try:
                fig = self.visualizer.create_interactive_plotly()
                pio.write_html(fig, file_path)
                QMessageBox.information(self, "Success", "Interactive dashboard exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def setup_shot_playback_tab(self, tab_widget):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        main_layout = QHBoxLayout(tab_widget)

        # --- Left Panel: File Selection and Shot Overview ---
        left_panel = QVBoxLayout()
        self.shot_json_btn = QPushButton("Load Shot Skeleton JSON")
        self.shot_json_btn.clicked.connect(self.load_shot_playback_json)
        left_panel.addWidget(self.shot_json_btn)

        self.shot_csv_btn = QPushButton("Load Shot CSV")
        self.shot_csv_btn.clicked.connect(self.load_shot_playback_csv)
        left_panel.addWidget(self.shot_csv_btn)

        # Add shot overview chart
        left_panel.addWidget(QLabel("Shot Overview (Click to navigate):"))
        self.shot_overview_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        left_panel.addWidget(self.shot_overview_canvas, 1)

        # Add current shot/phase info
        self.current_shot_info_label = QLabel("No data loaded")
        self.current_shot_info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        left_panel.addWidget(self.current_shot_info_label)

        main_layout.addLayout(left_panel, 1)

        # --- Center Panel: Skeleton Playback and Virtual Target ---
        center_panel = QVBoxLayout()

        # --- Comparison Controls ---
        compare_controls_layout = QHBoxLayout()
        self.compare_shots_toggle = QCheckBox("Compare Shots")
        self.compare_shots_toggle.setChecked(False)
        self.compare_shots_toggle.toggled.connect(self.on_compare_shots_toggled)
        compare_controls_layout.addWidget(self.compare_shots_toggle)
        compare_controls_layout.addSpacing(10)
        self.compare_shot_dropdown = QComboBox()
        self.compare_shot_dropdown.setEnabled(False)
        self.compare_shot_dropdown.setVisible(False)
        self.compare_shot_dropdown.currentIndexChanged.connect(self.on_compare_shot_selected)
        compare_controls_layout.addWidget(QLabel("Compare with Shot:"))
        compare_controls_layout.addWidget(self.compare_shot_dropdown)
        compare_controls_layout.addSpacing(10)
        self.sync_sliders_checkbox = QCheckBox("Sync Sliders")
        self.sync_sliders_checkbox.setChecked(False)
        self.sync_sliders_checkbox.setEnabled(False)
        self.sync_sliders_checkbox.setVisible(False)
        self.sync_sliders_checkbox.stateChanged.connect(self.on_sync_sliders_toggled)
        compare_controls_layout.addWidget(self.sync_sliders_checkbox)
        compare_controls_layout.addStretch()
        center_panel.addLayout(compare_controls_layout)

        # Skeleton playback display
        self.shot_playback_display = DrawableDisplayWidget()
        self.shot_playback_display.setMinimumSize(640, 480)
        center_panel.addWidget(self.shot_playback_display, 4)

        # Playback controls
        self.shot_playback_controls = VideoControls()
        center_panel.addWidget(self.shot_playback_controls)

        # Virtual target visualization (replacing phase duration chart)
        virtual_target_label = QLabel("Virtual Target Visualization")
        virtual_target_label.setAlignment(Qt.AlignCenter)
        virtual_target_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        center_panel.addWidget(virtual_target_label)
        
        # Add filter controls for virtual target
        filter_layout = QHBoxLayout()
        
        # Shot range filter
        shot_range_layout = QHBoxLayout()
        shot_range_layout.addWidget(QLabel("Shot Range:"))
        self.shot_playback_min_spin = QSpinBox()
        self.shot_playback_min_spin.setRange(0, 1000)
        self.shot_playback_min_spin.setValue(0)
        self.shot_playback_min_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        shot_range_layout.addWidget(self.shot_playback_min_spin)
        shot_range_layout.addWidget(QLabel("to"))
        self.shot_playback_max_spin = QSpinBox()
        self.shot_playback_max_spin.setRange(0, 1000)
        self.shot_playback_max_spin.setValue(1000)
        self.shot_playback_max_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        shot_range_layout.addWidget(self.shot_playback_max_spin)
        filter_layout.addLayout(shot_range_layout)
        
        filter_layout.addSpacing(20)
        
        # Score range filter
        score_range_layout = QHBoxLayout()
        score_range_layout.addWidget(QLabel("Score Range:"))
        self.shot_playback_score_min_spin = QDoubleSpinBox()
        self.shot_playback_score_min_spin.setRange(0.0, 10.0)
        self.shot_playback_score_min_spin.setValue(0.0)
        self.shot_playback_score_min_spin.setDecimals(1)
        self.shot_playback_score_min_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        score_range_layout.addWidget(self.shot_playback_score_min_spin)
        score_range_layout.addWidget(QLabel("to"))
        self.shot_playback_score_max_spin = QDoubleSpinBox()
        self.shot_playback_score_max_spin.setRange(0.0, 10.0)
        self.shot_playback_score_max_spin.setValue(10.0)
        self.shot_playback_score_max_spin.setDecimals(1)
        self.shot_playback_score_max_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        score_range_layout.addWidget(self.shot_playback_score_max_spin)
        filter_layout.addLayout(score_range_layout)
        
        filter_layout.addStretch()
        
        # Clear filters button
        self.clear_virtual_target_filters_btn = QPushButton("Clear Filters")
        self.clear_virtual_target_filters_btn.clicked.connect(self.clear_shot_playback_filters)
        filter_layout.addWidget(self.clear_virtual_target_filters_btn)
        
        center_panel.addLayout(filter_layout)
        
        # Add zoom controls for the virtual target
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Target Zoom:"))
        
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setToolTip("Zoom in on target")
        zoom_in_btn.clicked.connect(self.zoom_in_virtual_target)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setToolTip("Zoom out from target")
        zoom_out_btn.clicked.connect(self.zoom_out_virtual_target)
        zoom_layout.addWidget(zoom_out_btn)
        
        reset_zoom_btn = QPushButton("🔄")
        reset_zoom_btn.setToolTip("Reset zoom to default")
        reset_zoom_btn.clicked.connect(self.reset_virtual_target_zoom)
        zoom_layout.addWidget(reset_zoom_btn)
        
        zoom_layout.addStretch()
        center_panel.addLayout(zoom_layout)
        
        # Create a scroll area for the virtual target canvas to handle overflow
        virtual_target_scroll = QScrollArea()
        virtual_target_scroll.setWidgetResizable(True)
        virtual_target_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        virtual_target_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        virtual_target_scroll.setMinimumSize(640, 480)  # Same size as skeleton display
        
        # Create the virtual target canvas with larger size
        self.virtual_target_canvas = FigureCanvas(Figure(figsize=(12, 10)))
        virtual_target_scroll.setWidget(self.virtual_target_canvas)
        center_panel.addWidget(virtual_target_scroll, 4)  # Same stretch as skeleton display

        main_layout.addLayout(center_panel, 3)

        # --- State ---
        self.shot_playback_json_data = None
        self.shot_playback_csv_df = None
        self.shot_playback_frames_by_shot = {}
        self.shot_playback_timer = QTimer(self)
        self.shot_playback_timer.timeout.connect(self.advance_shot_playback_frame)
        self.shot_playback_current_frames = []
        self.shot_playback_current_frame_idx = 0
        self.shot_playback_fps = 15
        self.shot_playback_current_shot = None
        self.shot_playback_current_phase = None

        # --- Comparison State ---
        self.compare_mode_enabled = False
        self.compare_shot_number = None
        self.compare_shot_frames = []
        self.compare_frame_idx = 0  # Independent frame index for comparison shot

        # Connect controls
        self.shot_playback_controls.play_pause_btn.clicked.connect(self.toggle_shot_playback)
        self.shot_playback_controls.stop_btn.clicked.connect(self.reset_shot_playback)
        self.shot_playback_controls.speed_slider.valueChanged.connect(self.set_shot_playback_speed)
        self.shot_playback_controls.progress_slider.sliderMoved.connect(self.shot_playback_slider_moved)
        self.shot_playback_controls.progress_slider.valueChanged.connect(self.shot_playback_slider_value_changed)
        self.shot_playback_controls.forward_btn.clicked.connect(lambda: self.seek_shot_playback_relative(15))
        self.shot_playback_controls.backward_btn.clicked.connect(lambda: self.seek_shot_playback_relative(-15))
        
        # Connect comparison slider with debug prints
        print("Connecting comparison slider signals...")
        # Disconnect any existing connections from VideoControls
        try:
            self.shot_playback_controls.compare_progress_slider.sliderMoved.disconnect()
        except:
            pass
        try:
            self.shot_playback_controls.compare_progress_slider.valueChanged.disconnect()
        except:
            pass
        
        # Connect to our custom methods
        self.shot_playback_controls.compare_progress_slider.sliderMoved.connect(self.compare_shot_playback_slider_moved)
        self.shot_playback_controls.compare_progress_slider.valueChanged.connect(self.compare_shot_playback_slider_value_changed)
        print("Comparison slider signals connected to custom methods")

        # Connect canvas click events for navigation
        self.shot_overview_canvas.mpl_connect('button_press_event', self.on_shot_overview_click)
        
        # Add a test button to manually trigger comparison slider (for debugging)
        test_btn = QPushButton("Test Compare Slider")
        test_btn.clicked.connect(self.test_compare_slider)
        center_panel.addWidget(test_btn)
        
        # Add a test button to manually trigger virtual target visualization
        virtual_target_test_btn = QPushButton("Test Virtual Target")
        virtual_target_test_btn.clicked.connect(lambda: self.update_shot_playback_virtual_target(1))
        center_panel.addWidget(virtual_target_test_btn)

    def on_compare_shots_toggled(self, checked):
        self.compare_mode_enabled = checked
        self.compare_shot_dropdown.setEnabled(checked)
        self.compare_shot_dropdown.setVisible(checked)
        self.sync_sliders_checkbox.setEnabled(checked)
        self.sync_sliders_checkbox.setVisible(checked)
        if checked:
            self.shot_playback_controls.compare_slider_label.setVisible(True)
            self.shot_playback_controls.compare_progress_slider.setVisible(True)
            self.shot_playback_controls.compare_slider_label.setText("Comparison Shot")
            if self.compare_shot_frames:
                self.shot_playback_controls.compare_progress_slider.setRange(0, max(0, len(self.compare_shot_frames)-1))
                self.shot_playback_controls.compare_progress_slider.setValue(0)
        else:
            self.shot_playback_controls.compare_slider_label.setVisible(False)
            self.shot_playback_controls.compare_progress_slider.setVisible(False)
            self.sync_sliders_checkbox.setChecked(False)
            self.compare_shot_number = None
            self.compare_shot_frames = []
            self.compare_frame_idx = 0
            self.draw_shot_playback_current_frame()

    def on_compare_shot_selected(self, idx):
        if not self.compare_mode_enabled or self.shot_playback_csv_df is None:
            return
        shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
        if 0 <= idx < len(shot_numbers):
            self.compare_shot_number = shot_numbers[idx]
            self.compare_shot_frames = self.shot_playback_frames_by_shot.get(self.compare_shot_number, [])
            print(f"Selected comparison shot {self.compare_shot_number} with {len(self.compare_shot_frames)} frames")
            # Set the comparison slider range and reset frame index
            if self.compare_shot_frames:
                max_frame = max(0, len(self.compare_shot_frames)-1)
                self.shot_playback_controls.compare_progress_slider.setRange(0, max_frame)
                self.shot_playback_controls.compare_progress_slider.setValue(0)
                self.compare_frame_idx = 0
                print(f"Set comparison slider range: 0 to {max_frame}")
            self.draw_shot_playback_current_frame()
            # Update phase chart for both shots
            if self.shot_playback_current_shot is not None:
                self.update_shot_playback_phase_chart(self.shot_playback_current_shot, self.compare_shot_number)
            self.calculate_and_set_phase_markers()

    def load_shot_playback_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Skeleton JSON", "", "JSON Files (*.json)")
        if not filepath:
            return
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Accepts both {"frames": [...]} and just a list
        frames = data.get("frames", data) if isinstance(data, dict) else data
        # Group frames by shot number (if available), else by phase or just index
        frames_by_shot = {}
        for idx, frame in enumerate(frames):
            # Try to get shot number from frame metadata, fallback to index
            shot_num = None
            if "shot_number" in frame:
                shot_num = int(frame["shot_number"])
            elif "Shot_Number" in frame:
                shot_num = int(frame["Shot_Number"])
            elif "phase" in frame and frame["phase"] is not None:
                shot_num = str(frame["phase"])
            else:
                shot_num = int(idx // 100 + 1)  # crude fallback: every 100 frames = new shot
            frames_by_shot.setdefault(shot_num, []).append(frame)
        self.shot_playback_json_data = frames
        self.shot_playback_frames_by_shot = frames_by_shot
        self.update_shot_playback_shot_list()

    def load_shot_playback_csv(self):
        # First check if we have data from the main system that we can use
        if hasattr(self, 'combined_df') and self.combined_df is not None:
            # Check if the combined data has the required columns for shot playback
            required_columns = ['Shot_Number', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
            if all(col in self.combined_df.columns for col in required_columns):
                # Use the existing data
                self.shot_playback_csv_df = self.combined_df.copy()
                self.update_shot_playback_shot_list()
                QMessageBox.information(self, "Data Loaded", 
                                      f"Using existing data with {len(self.shot_playback_csv_df)} shots for shot playback analysis.")
                return
        
        # If no suitable data is available, prompt for file selection
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Shot CSV", "", "CSV Files (*.csv)")
        if not filepath:
            return
        import pandas as pd
        self.shot_playback_csv_df = pd.read_csv(filepath)
        self.update_shot_playback_shot_list()

    def update_shot_playback_shot_list(self):
        # Only update if both JSON and CSV are loaded
        if self.shot_playback_csv_df is None or not self.shot_playback_frames_by_shot:
            return
        
        # Create horizontal stacked bar chart showing all shots
        self.shot_overview_canvas.figure.clf()
        ax = self.shot_overview_canvas.figure.add_subplot(111)
        
        shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
        phases = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
        colors = PHASE_COLORS_LIST
        
        # Prepare data for stacked bars
        bottom = np.zeros(len(shot_numbers))
        
        for i, phase in enumerate(phases):
            values = []
            for shot_num in shot_numbers:
                row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == shot_num]
                if not row.empty:
                    value = float(row.iloc[0].get(phase, 0))
                else:
                    value = 0
                values.append(value)
            
            ax.barh(shot_numbers, values, left=bottom, 
                   color=colors[i], alpha=0.8, 
                   label=phase.replace("_Time(s)", ""))
            bottom += np.array(values)
        
        # Add score annotations
        for i, shot_num in enumerate(shot_numbers):
            row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == shot_num]
            if not row.empty:
                score = row.iloc[0].get('Score', '')
                if score:
                    total_width = bottom[i]
                    ax.text(total_width + 0.1, shot_num, f"Score: {score}", 
                           va='center', ha='left', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Shot Number")
        ax.set_title("Shot Overview - Phase Durations and Scores")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')
        
        self.shot_overview_canvas.figure.tight_layout()
        self.shot_overview_canvas.draw()
        
        # Auto-select first shot if available
        if shot_numbers and shot_numbers[0] in self.shot_playback_frames_by_shot:
            self.shot_playback_current_shot = shot_numbers[0]
            self.shot_playback_current_frames = self.shot_playback_frames_by_shot[shot_numbers[0]]
            self.shot_playback_current_frame_idx = 0
            self.shot_playback_controls.set_total_frames(len(self.shot_playback_current_frames))
            # Calculate and set phase markers for the first shot
            self.calculate_and_set_phase_markers(self.shot_playback_current_frames)
            self.draw_shot_playback_current_frame()
            # If compare mode, show both shots in phase chart
            if getattr(self, 'compare_mode_enabled', False) and self.compare_shot_number is not None:
                self.update_shot_playback_phase_chart(shot_numbers[0], self.compare_shot_number)
            else:
                self.update_shot_playback_phase_chart(shot_numbers[0])
            self.update_current_shot_info()
        
        # --- Populate compare shot dropdown ---
        if hasattr(self, 'compare_shots_toggle') and self.compare_shots_toggle.isChecked():
            self.populate_compare_shot_dropdown()

    def on_shot_overview_click(self, event):
        """Handle clicks on the shot overview chart to navigate to specific shots"""
        if event.inaxes != self.shot_overview_canvas.figure.axes[0]:
            return
        
        # Get the shot number from the y-axis position (since it's a horizontal bar chart)
        shot_number = int(round(event.ydata))
        
        # Verify the shot number exists in our data
        if shot_number in self.shot_playback_frames_by_shot:
            # Find the frame index for this shot
            frames = self.shot_playback_frames_by_shot[shot_number]
            if frames:
                self.shot_playback_current_frames = frames
                self.shot_playback_current_frame_idx = 0
                self.shot_playback_current_shot = shot_number
                self.shot_playback_controls.set_total_frames(len(frames))
                
                # Calculate and set phase markers
                self.calculate_and_set_phase_markers(frames)
                
                self.draw_shot_playback_current_frame()
                self.update_shot_playback_phase_chart(shot_number)
                self.update_current_shot_info()
                
                # Highlight the shot in the virtual target for bidirectional highlighting
                self.highlight_shot_in_virtual_target(shot_number)
                
                # Update compare shot dropdown
                if hasattr(self, 'compare_shots_toggle') and self.compare_shots_toggle.isChecked():
                    self.populate_compare_shot_dropdown()

    def populate_compare_shot_dropdown(self):
        if self.shot_playback_csv_df is None or self.shot_playback_current_shot is None:
            self.compare_shot_dropdown.clear()
            self.compare_shot_dropdown.setEnabled(False)
            self.compare_shot_dropdown.setVisible(False)
            return
        shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
        # Exclude the current primary shot
        compare_options = [str(s) for s in shot_numbers if s != self.shot_playback_current_shot]
        self.compare_shot_dropdown.blockSignals(True)
        self.compare_shot_dropdown.clear()
        self.compare_shot_dropdown.addItems(compare_options)
        self.compare_shot_dropdown.setEnabled(True)
        self.compare_shot_dropdown.setVisible(True)
        self.compare_shot_dropdown.blockSignals(False)
        # Optionally, auto-select the first available shot
        if compare_options:
            self.compare_shot_dropdown.setCurrentIndex(0)
            self.on_compare_shot_selected(0)
        else:
            self.compare_shot_number = None
            self.compare_shot_frames = []
            self.draw_shot_playback_current_frame()

    def on_shot_playback_shot_selected(self, item):
        # This method is no longer used - navigation is now handled by chart clicks
        pass
    def draw_shot_playback_current_frame(self):
        if not self.shot_playback_current_frames or not (0 <= self.shot_playback_current_frame_idx < len(self.shot_playback_current_frames)):
            self.shot_playback_display.set_image(None)
            self.shot_playback_controls.update_frame_display(0, len(self.shot_playback_current_frames))
            return
        
        frame = self.shot_playback_current_frames[self.shot_playback_current_frame_idx]
        
        # --- Comparison: get compare frame by independent position ---
        compare_frame = None
        compare_phase = None
        if getattr(self, 'compare_mode_enabled', False) and self.compare_shot_frames:
            if 0 <= self.compare_frame_idx < len(self.compare_shot_frames):
                compare_frame = self.compare_shot_frames[self.compare_frame_idx]
                compare_phase = compare_frame.get("phase", compare_frame.get("Phase", "Unknown"))
                print(f"Drawing comparison frame {self.compare_frame_idx} with phase {compare_phase}")
            else:
                print(f"Compare frame index {self.compare_frame_idx} out of range (0-{len(self.compare_shot_frames)-1})")
        
        # Update current phase information
        self.shot_playback_current_phase = frame.get("phase", frame.get("Phase", "Unknown"))
        self.update_current_shot_info()
        
        # Draw skeleton on a QImage
        img_width, img_height = self.shot_playback_display.width(), self.shot_playback_display.height()
        if img_width <= 0 or img_height <= 0:
            img_width, img_height = 640, 480
        
        from PyQt5.QtGui import QImage, QPainter, QPen, QColor
        import numpy as np
        image = QImage(img_width, img_height, QImage.Format_RGB888)
        image.fill(Qt.black)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate aspect ratio preservation for skeleton
        skeleton_aspect_ratio = 4/3  # Standard skeleton aspect ratio (width/height)
        widget_aspect_ratio = img_width / img_height if img_height > 0 else 1
        
        if widget_aspect_ratio > skeleton_aspect_ratio:
            # Widget is wider - fit to height
            x_offset = (img_width - img_height * skeleton_aspect_ratio) / 2
            y_offset = 0
            skeleton_width = img_height * skeleton_aspect_ratio
            skeleton_height = img_height
        else:
            # Widget is taller - fit to width  
            x_offset = 0
            y_offset = (img_height - img_width / skeleton_aspect_ratio) / 2
            skeleton_width = img_width
            skeleton_height = img_width / skeleton_aspect_ratio
        
        # Draw phase information at the top
        painter.setPen(QPen(Qt.white, 2))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        # Get current shot score
        current_score = "N/A"
        if self.shot_playback_csv_df is not None:
            row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == self.shot_playback_current_shot]
            if not row.empty:
                current_score = row.iloc[0].get('Score', 'N/A')
        
        phase_text = f"Shot {self.shot_playback_current_shot} | Phase: {self.shot_playback_current_phase} | Score: {current_score}"
        if compare_frame is not None and self.compare_shot_number is not None:
            # Get compare shot score
            compare_score = "N/A"
            if self.shot_playback_csv_df is not None:
                compare_row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == self.compare_shot_number]
                if not compare_row.empty:
                    compare_score = compare_row.iloc[0].get('Score', 'N/A')
            phase_text += f"    |    Shot {self.compare_shot_number} | Phase: {compare_phase} | Score: {compare_score}"
        painter.drawText(10, 30, phase_text)
        
        # Draw legend
        if compare_frame is not None and self.compare_shot_number is not None:
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.white)
            painter.drawRect(10, 40, 18, 18)
            painter.setPen(Qt.white)
            painter.drawText(32, 54, f"Shot {self.shot_playback_current_shot}")
            painter.setPen(QPen(QColor(0, 128, 255), 2))
            painter.setBrush(QColor(0, 128, 255, 180))
            painter.drawRect(10, 62, 18, 18)
            painter.setPen(QColor(0, 128, 255))
            painter.drawText(32, 76, f"Shot {self.compare_shot_number}")
        
        # Draw comparison skeleton (blue, semi-transparent)
        if compare_frame is not None:
            landmarks = compare_frame.get("landmarks") or compare_frame.get("Landmarks")
            if landmarks:
                points = []
                for lm in landmarks:
                    if isinstance(lm, dict):
                        x = lm.get("x", lm.get("X"))
                        y = lm.get("y", lm.get("Y"))
                    elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                        x, y = lm[0], lm[1]
                    else:
                        continue
                    if x is not None and y is not None:
                        px = int(float(x) * skeleton_width + x_offset)
                        py = int(float(y) * skeleton_height + y_offset)
                        points.append((px, py))
                # Draw connections (MediaPipe style)
                blue = QColor(0, 128, 255, 180)
                painter.setPen(QPen(blue, 3))
                for p1, p2 in POSE_CONNECTIONS:
                    if p1 < len(points) and p2 < len(points):
                        painter.drawLine(points[p1][0], points[p1][1], points[p2][0], points[p2][1])
                painter.setPen(QPen(blue, 1))
                painter.setBrush(blue)
                for px, py in points:
                    painter.drawEllipse(px-4, py-4, 8, 8)
        
        # Draw primary skeleton (white, as before)
        landmarks = frame.get("landmarks") or frame.get("Landmarks")
        if landmarks:
            points = []
            for lm in landmarks:
                if isinstance(lm, dict):
                    x = lm.get("x", lm.get("X"))
                    y = lm.get("y", lm.get("Y"))
                elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                    x, y = lm[0], lm[1]
                else:
                    continue
                if x is not None and y is not None:
                    px = int(float(x) * skeleton_width + x_offset)
                    py = int(float(y) * skeleton_height + y_offset)
                    points.append((px, py))
            phase_colors = UNIFIED_PHASE_QCOLORS
            current_phase_lower = self.shot_playback_current_phase.lower() if self.shot_playback_current_phase else 'unknown'
            skeleton_color = phase_colors.get(current_phase_lower, phase_colors['unknown'])
            painter.setPen(QPen(skeleton_color, 3))
            for p1, p2 in POSE_CONNECTIONS:
                if p1 < len(points) and p2 < len(points):
                    painter.drawLine(points[p1][0], points[p1][1], points[p2][0], points[p2][1])
            painter.setPen(QPen(skeleton_color, 1))
            painter.setBrush(skeleton_color)
            for px, py in points:
                painter.drawEllipse(px-4, py-4, 8, 8)
        
        # Draw small target visualization in top-right corner
        self.draw_small_target_visualization(painter, img_width, img_height, self.shot_playback_current_shot, compare_shot_number=self.compare_shot_number if compare_frame is not None else None)
        
        painter.end()
        self.shot_playback_display.set_image(image)
        self.shot_playback_controls.update_frame_display(self.shot_playback_current_frame_idx, len(self.shot_playback_current_frames))

    def toggle_shot_playback(self, checked):
        if not self.shot_playback_current_frames:
            self.shot_playback_controls.play_pause_btn.setChecked(False)
            self.shot_playback_controls.play_pause_btn.setText("▶")
            return
        if checked:
            self.shot_playback_controls.play_pause_btn.setText("⏸")
            speed = self.shot_playback_controls.speed_slider.value() / 100.0
            interval = int(1000 / (self.shot_playback_fps * speed))
            self.shot_playback_timer.start(max(20, interval))
        else:
            self.shot_playback_controls.play_pause_btn.setText("▶")
            self.shot_playback_timer.stop()

    def reset_shot_playback(self):
        self.shot_playback_timer.stop()
        self.shot_playback_controls.play_pause_btn.setChecked(False)
        self.shot_playback_controls.play_pause_btn.setText("▶")
        self.shot_playback_current_frame_idx = 0
        self.draw_shot_playback_current_frame()

    def set_shot_playback_speed(self, value):
        if self.shot_playback_timer.isActive():
            speed = value / 100.0
            interval = int(1000 / (self.shot_playback_fps * speed))
            self.shot_playback_timer.setInterval(max(20, interval))

    def shot_playback_slider_moved(self, frame_index):
        if self._syncing_sliders:
            return
        if 0 <= frame_index < len(self.shot_playback_current_frames):
            self.shot_playback_current_frame_idx = frame_index
            # Update comparison slider to keep in sync only if sync is enabled
            if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
                n1 = len(self.shot_playback_current_frames)
                n2 = len(self.compare_shot_frames)
                rel = frame_index / max(1, n1-1)
                compare_idx = int(round(rel * (n2-1)))
                self._syncing_sliders = True
                self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
                self._syncing_sliders = False
                self.compare_frame_idx = compare_idx
            # Update both phase bars
            self.calculate_and_set_phase_markers()
            self.draw_shot_playback_current_frame()
            if self.shot_playback_controls.play_pause_btn.isChecked():
                self.shot_playback_timer.stop()
                QTimer.singleShot(50, lambda: self.shot_playback_timer.start() if self.shot_playback_controls.play_pause_btn.isChecked() else None)

    def shot_playback_slider_value_changed(self, frame_index):
        if self._syncing_sliders:
            return
        if not self.shot_playback_controls.progress_slider.isSliderDown():
            if 0 <= frame_index < len(self.shot_playback_current_frames):
                self.shot_playback_current_frame_idx = frame_index
                # Update comparison slider to keep in sync only if sync is enabled
                if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
                    n1 = len(self.shot_playback_current_frames)
                    n2 = len(self.compare_shot_frames)
                    rel = frame_index / max(1, n1-1)
                    compare_idx = int(round(rel * (n2-1)))
                    self._syncing_sliders = True
                    self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
                    self._syncing_sliders = False
                    self.compare_frame_idx = compare_idx
                # Update both phase bars
                self.calculate_and_set_phase_markers()
                self.draw_shot_playback_current_frame()

    def seek_shot_playback_relative(self, frame_offset):
        if not self.shot_playback_current_frames:
            return
        new_idx = self.shot_playback_current_frame_idx + frame_offset
        self.shot_playback_current_frame_idx = max(0, min(new_idx, len(self.shot_playback_current_frames) - 1))
        # Update comparison slider to keep in sync only if sync is enabled
        if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
            n1 = len(self.shot_playback_current_frames)
            n2 = len(self.compare_shot_frames)
            rel = self.shot_playback_current_frame_idx / max(1, n1-1)
            compare_idx = int(round(rel * (n2-1)))
            self._syncing_sliders = True
            self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
            self._syncing_sliders = False
            self.compare_frame_idx = compare_idx
        self.draw_shot_playback_current_frame()
        self.shot_playback_controls.progress_slider.setValue(self.shot_playback_current_frame_idx)
        if self.shot_playback_controls.play_pause_btn.isChecked():
            self.shot_playback_timer.stop()
            QTimer.singleShot(50, lambda: self.shot_playback_timer.start() if self.shot_playback_controls.play_pause_btn.isChecked() else None)

    def advance_shot_playback_frame(self):
        if not self.shot_playback_current_frames:
            return
        self.shot_playback_current_frame_idx += 1
        if self.shot_playback_current_frame_idx >= len(self.shot_playback_current_frames):
            self.shot_playback_current_frame_idx = 0
        # Update comparison slider to keep in sync only if sync is enabled
        if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
            n1 = len(self.shot_playback_current_frames)
            n2 = len(self.compare_shot_frames)
            rel = self.shot_playback_current_frame_idx / max(1, n1-1)
            compare_idx = int(round(rel * (n2-1)))
            self._syncing_sliders = True
            self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
            self._syncing_sliders = False
            self.compare_frame_idx = compare_idx
        self.draw_shot_playback_current_frame()

    def on_sync_sliders_toggled(self, checked):
        """Handle sync sliders checkbox toggle - sync from current positions"""
        if checked and self.compare_shot_frames:
            # When sync is enabled, keep sliders at their current relative positions
            # Don't force them to start from beginning - let the existing sync logic handle it
            pass

    def update_shot_playback_phase_chart(self, shot_num, compare_shot_num=None):
        """
        Update the phase duration bar at the bottom. If compare_shot_num is provided, show both shots side by side.
        """
        # Check if old canvas exists, redirect if not
        if not hasattr(self, 'shot_phase_chart_canvas'):
            print("Redirecting to virtual target - old canvas not found")
            self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
            return
            
        if self.shot_playback_csv_df is None:
            return
        
        # New color scheme
        phase_labels = ["Preparing", "Aiming", "After Shot"]
        phases = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
        colors = PHASE_COLORS_LIST
        
        # Get primary shot data
        row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == int(shot_num)]
        if row.empty:
            try:
                self.shot_phase_chart_canvas.figure.clf()
                self.shot_phase_chart_canvas.draw()
            except AttributeError:
                print("Redirecting to virtual target - canvas error")
                self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
            return
        row = row.iloc[0]
        values = [float(row.get(phase, 0)) for phase in phases]
        
        # If compare_shot_num is provided, get its data
        compare_values = None
        if compare_shot_num is not None:
            compare_row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == int(compare_shot_num)]
            if not compare_row.empty:
                compare_row = compare_row.iloc[0]
                compare_values = [float(compare_row.get(phase, 0)) for phase in phases]
        
        try:
            self.shot_phase_chart_canvas.figure.clf()  # Clear the figure
            ax = self.shot_phase_chart_canvas.figure.add_subplot(111)  # Create a new axes
        except AttributeError:
            print("Redirecting to virtual target - canvas error")
            self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
            return
        
        if compare_values is not None:
            # Grouped bar chart: primary and comparison
            bar_width = 0.35
            x = np.arange(len(phases))
            bars1 = ax.bar(x - bar_width/2, values, width=bar_width, color=colors, label=f"Shot {shot_num}")
            bars2 = ax.bar(x + bar_width/2, compare_values, width=bar_width, color=colors, alpha=0.5, label=f"Shot {compare_shot_num}")
            for i, (bar, val) in enumerate(zip(bars1, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.2f}s", ha='center', va='center', color='white', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars2, compare_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.2f}s", ha='center', va='center', color='black', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(phase_labels)
            ax.legend()
            ax.set_title(f"Shot {shot_num} vs Shot {compare_shot_num} Phase Durations")
            ax.set_ylim(0, max(values + compare_values) * 1.2 if values + compare_values else 1)
        else:
            # Single shot
            bars = ax.bar(phase_labels, values, color=colors)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.2f}s", ha='center', va='center', color='white', fontsize=12)
            ax.set_ylabel("Time (s)")
            ax.set_ylim(0, max(values) * 1.2 if values else 1)
            ax.set_title(f"Shot {shot_num} Phase Durations")
        try:
            self.shot_phase_chart_canvas.figure.tight_layout()
            self.shot_phase_chart_canvas.draw()
        except AttributeError:
            print("Redirecting to virtual target - canvas error")
            self.update_shot_playback_virtual_target(shot_num, compare_shot_num)

    def on_score_edited(self, row, column):
        col_name = self.data_table.horizontalHeaderItem(column).text()
        if col_name == "Score":
            # Find the column index for Shot_Number
            shot_col = None
            for i in range(self.data_table.columnCount()):
                if self.data_table.horizontalHeaderItem(i).text() == "Shot_Number":
                    shot_col = i
                    break
            if shot_col is None:
                return  # Can't find Shot_Number column
            shot_number_item = self.data_table.item(row, shot_col)
            if shot_number_item is None:
                return
            try:
                shot_number = int(shot_number_item.text())
            except ValueError:
                return
            new_score = self.data_table.item(row, column).text()
            
            # Update the combined DataFrame
            self.combined_df.loc[self.combined_df['Shot_Number'] == shot_number, 'Score'] = new_score
            
            # Provide visual feedback that changes were made
            self.save_data_btn.setText("Save Changes *")
            self.save_data_btn.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")
            
            # Update the shot overview chart if we're in shot playback tab
            if hasattr(self, 'shot_playback_csv_df') and self.shot_playback_csv_df is not None:
                self.update_shot_playback_shot_list()
        
        elif col_name == "Direction":
            # Find the column index for Shot_Number
            shot_col = None
            for i in range(self.data_table.columnCount()):
                if self.data_table.horizontalHeaderItem(i).text() == "Shot_Number":
                    shot_col = i
                    break
            if shot_col is None:
                return  # Can't find Shot_Number column
            shot_number_item = self.data_table.item(row, shot_col)
            if shot_number_item is None:
                return
            try:
                shot_number = int(shot_number_item.text())
            except ValueError:
                return
            
            new_direction = self.data_table.item(row, column).text()
            
            # Validate direction - allow empty or one of the 8 directions
            valid_directions = ["", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]
            if new_direction not in valid_directions:
                # Reset to empty if invalid
                self.data_table.item(row, column).setText("")
                QMessageBox.warning(self, "Invalid Direction", 
                                  f"Please enter one of the valid directions: {', '.join(valid_directions[1:])} or leave empty")
                return
            
            # Update the combined DataFrame
            self.combined_df.loc[self.combined_df['Shot_Number'] == shot_number, 'Direction'] = new_direction
            
            # Provide visual feedback that changes were made
            self.save_data_btn.setText("Save Changes *")
            self.save_data_btn.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")

    def update_current_shot_info(self):
        """Update the current shot/phase information display"""
        if self.shot_playback_current_shot is not None and self.shot_playback_current_phase is not None:
            info_text = f"Shot {self.shot_playback_current_shot} | Phase: {self.shot_playback_current_phase}"
            if self.shot_playback_csv_df is not None:
                row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == self.shot_playback_current_shot]
                if not row.empty:
                    score = row.iloc[0].get('Score', 'N/A')
                    info_text += f" | Score: {score}"
            self.current_shot_info_label.setText(info_text)
        else:
            self.current_shot_info_label.setText("No data loaded")

    def save_data_table_changes(self):
        """Save the current data table changes to CSV files"""
        if self.combined_df is None:
            QMessageBox.warning(self, "No Data", "No data to save. Please load data first.")
            return
        
        try:
            # Apply current filters to get the data that should be saved
            filtered_df = self.combined_df.copy()
            
            # Apply player filter
            if self.player_combo.currentText() != "All Players":
                filtered_df = filtered_df[filtered_df['Player'] == self.player_combo.currentText()]
            
            # Apply session filter
            if self.session_combo.currentText() != "All Sessions":
                filtered_df = filtered_df[filtered_df['Session'] == self.session_combo.currentText()]
            
            # Apply shot range filter
            if 'Shot_Number' in filtered_df.columns:
                shot_min = self.shot_min_spin.value()
                shot_max = self.shot_max_spin.value()
                filtered_df = filtered_df[
                    (filtered_df['Shot_Number'] >= shot_min) & 
                    (filtered_df['Shot_Number'] <= shot_max)
                ]

            # Apply complete shot filter
            if 'Complete_Shot' in filtered_df.columns and self.complete_shot_checkbox.isChecked():
                filtered_df = filtered_df[filtered_df['Complete_Shot'].str.lower() == 'yes']
            
            # Ask user where to save
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Combined CSV File", "", "CSV Files (*.csv)")
            if file_path:
                # Save the filtered data
                filtered_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Combined data saved successfully to {file_path}")
                
                # Reset save button appearance
                self.save_data_btn.setText("Save Changes")
                self.save_data_btn.setStyleSheet("")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def calculate_and_set_phase_markers(self, frames=None):
        """
        Calculate phase boundaries and set markers on the slider (dual bar if compare mode).
        If in compare mode, use the current frame index for both primary and comparison shots.
        """
        # Use current frames if not provided
        if frames is None:
            frames = self.shot_playback_current_frames
        if not frames:
            self.shot_playback_controls.set_phase_markers([])
            return
        # --- Primary shot phase data ---
        phase_data = []
        current_phase = None
        phase_start_frame = 0
        for frame_idx, frame in enumerate(frames):
            frame_phase = frame.get("phase", frame.get("Phase", "unknown"))
            if current_phase is not None and frame_phase != current_phase:
                phase_data.append({
                    'start_frame': phase_start_frame,
                    'end_frame': frame_idx,
                    'phase_name': current_phase
                })
                phase_start_frame = frame_idx
            current_phase = frame_phase
        if current_phase is not None:
            phase_data.append({
                'start_frame': phase_start_frame,
                'end_frame': len(frames),
                'phase_name': current_phase
            })
        # --- Comparison shot phase data ---
        compare_phase_data = None
        compare_total_frames = None
        if getattr(self, 'compare_mode_enabled', False) and self.compare_shot_frames and len(self.compare_shot_frames) > 0:
            compare_frames = self.compare_shot_frames
            compare_phase_data = []
            compare_current_phase = None
            compare_phase_start = 0
            for frame_idx, frame in enumerate(compare_frames):
                frame_phase = frame.get("phase", frame.get("Phase", "unknown"))
                if compare_current_phase is not None and frame_phase != compare_current_phase:
                    compare_phase_data.append({
                        'start_frame': compare_phase_start,
                        'end_frame': frame_idx,
                        'phase_name': compare_current_phase
                    })
                    compare_phase_start = frame_idx
                compare_current_phase = frame_phase
            if compare_current_phase is not None:
                compare_phase_data.append({
                    'start_frame': compare_phase_start,
                    'end_frame': len(compare_frames),
                    'phase_name': compare_current_phase
                })
            compare_total_frames = len(compare_frames)
            self.shot_playback_controls.set_dual_phase_markers(
                phase_data, len(frames), compare_phase_data, compare_total_frames
            )
        else:
            self.shot_playback_controls.set_phase_markers(phase_data, len(frames))

    def compare_shot_playback_slider_moved(self, frame_index):
        if self._syncing_sliders:
            return
        if not self.compare_shot_frames:
            return
        if 0 <= frame_index < len(self.compare_shot_frames):
            self.compare_frame_idx = frame_index
            # Sync main slider by relative position only if sync is enabled
            if self.sync_sliders_checkbox.isChecked():
                n1 = len(self.shot_playback_current_frames)
                n2 = len(self.compare_shot_frames)
                rel = frame_index / max(1, n2-1)
                main_idx = int(round(rel * (n1-1)))
                self._syncing_sliders = True
                self.shot_playback_controls.progress_slider.setValue(main_idx)
                self._syncing_sliders = False
                self.shot_playback_current_frame_idx = main_idx
            # Update both phase bars
            self.calculate_and_set_phase_markers()
            self.draw_shot_playback_current_frame()
            if self.shot_playback_controls.play_pause_btn.isChecked():
                self.shot_playback_timer.stop()
                QTimer.singleShot(50, lambda: self.shot_playback_timer.start() if self.shot_playback_controls.play_pause_btn.isChecked() else None)

    def compare_shot_playback_slider_value_changed(self, frame_index):
        if self._syncing_sliders:
            return
        if not self.shot_playback_controls.compare_progress_slider.isSliderDown():
            if 0 <= frame_index < len(self.compare_shot_frames):
                self.compare_frame_idx = frame_index
                # Sync main slider by relative position only if sync is enabled
                if self.sync_sliders_checkbox.isChecked():
                    n1 = len(self.shot_playback_current_frames)
                    n2 = len(self.compare_shot_frames)
                    rel = frame_index / max(1, n2-1)
                    main_idx = int(round(rel * (n1-1)))
                    self._syncing_sliders = True
                    self.shot_playback_controls.progress_slider.setValue(main_idx)
                    self._syncing_sliders = False
                    self.shot_playback_current_frame_idx = main_idx
                # Update both phase bars
                self.calculate_and_set_phase_markers()
                self.draw_shot_playback_current_frame()

    def test_compare_slider(self):
        """Test method to manually trigger comparison slider movement"""
        print("=== TESTING COMPARE SLIDER ===")
        if self.compare_shot_frames:
            # Move to a different frame
            new_frame = (self.compare_frame_idx + 10) % len(self.compare_shot_frames)
            print(f"Manually setting comparison slider to frame {new_frame}")
            self.shot_playback_controls.compare_progress_slider.setValue(new_frame)
        else:
            print("No comparison frames available")

    def on_direction_changed(self, row, column, text):
        """Handle direction dropdown changes"""
        # Find the column index for Shot_Number
        shot_col = None
        for i in range(self.data_table.columnCount()):
            if self.data_table.horizontalHeaderItem(i).text() == "Shot_Number":
                shot_col = i
                break
        if shot_col is None:
            return  # Can't find Shot_Number column
        shot_number_item = self.data_table.item(row, shot_col)
        if shot_number_item is None:
            return
        try:
            shot_number = int(shot_number_item.text())
        except ValueError:
            return
        
        # Update the combined DataFrame
        self.combined_df.loc[self.combined_df['Shot_Number'] == shot_number, 'Direction'] = text
        
        # Provide visual feedback that changes were made
        self.save_data_btn.setText("Save Changes *")
        self.save_data_btn.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")

    def get_scaling_settings(self):
        """Get current scaling settings from UI controls"""
        return {
            'figsize': (self.fig_width_spin.value(), self.fig_height_spin.value()),
            'fontsize': self.font_size_spin.value(),
            'auto_scale': self.auto_scale_checkbox.isChecked(),
            'compact_mode': self.compact_mode_checkbox.isChecked(),
            'zoom_target': self.zoom_target_checkbox.isChecked()
        }

    def update_visualization(self):
        """Update visualization based on selection"""
        if self.chart_list.currentItem():
            self.generate_selected_chart()
    
    def generate_selected_chart(self):
        """Generate the selected chart type"""
        if not self.chart_list.currentItem() or self.visualizer is None:
            return
        
        chart_type = self.chart_list.currentItem().text()
        
        # Get scaling settings
        scaling_settings = self.get_scaling_settings()
        
        # Clear previous plots
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)
        
        try:
            if chart_type == "Small-Multiples Line Charts":
                fig = self.visualizer.small_multiples_line_charts(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Stacked Bar Composition":
                fig = self.visualizer.stacked_bars_shot_composition(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Box-Violin Distribution":
                fig = self.visualizer.box_violin_plots(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Phase Duration Heatmap":
                fig = self.visualizer.heatmap_phase_durations(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Side-by-Side Bar Chart":
                fig = self.visualizer.side_by_side_bar_chart(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Interactive Dashboard":
                fig = self.visualizer.create_interactive_plotly()
                html_string = pio.to_html(fig, include_plotlyjs='cdn')
                self.web_view.setHtml(html_string)
                self.tab_widget.setCurrentIndex(1)  # Switch to interactive tab
                
            elif chart_type == "Score vs Durations":
                fig = self.visualizer.score_vs_durations(scaling_settings=scaling_settings)
                self.add_matplotlib_figure(fig)
                
            elif chart_type == "Virtual Target Visualization":
                # Get zoom setting
                zoom_target = self.zoom_target_checkbox.isChecked()
                fig = self.visualizer.virtual_target_visualization(scaling_settings=scaling_settings, zoom_target=zoom_target)
                self.add_matplotlib_figure(fig)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate chart: {str(e)}")
    
    def add_matplotlib_figure(self, fig):
        """Add matplotlib figure to the layout"""
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(600)
        self.plot_layout.addWidget(canvas)
        self.tab_widget.setCurrentIndex(0)  # Switch to static charts tab
    
    def export_png(self):
        """Export current matplotlib figure as PNG"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG Files (*.png)")
        if file_path and self.plot_layout.count() > 0:
            try:
                canvas = self.plot_layout.itemAt(0).widget()
                if isinstance(canvas, FigureCanvas):
                    canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, "Success", "Chart exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def export_html(self):
        """Export interactive dashboard as HTML"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save HTML", "", "HTML Files (*.html)")
        if file_path and self.visualizer:
            try:
                fig = self.visualizer.create_interactive_plotly()
                pio.write_html(fig, file_path)
                QMessageBox.information(self, "Success", "Interactive dashboard exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    def setup_shot_playback_tab(self, tab_widget):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        main_layout = QHBoxLayout(tab_widget)

        # --- Left Panel: File Selection and Shot Overview ---
        left_panel = QVBoxLayout()
        self.shot_json_btn = QPushButton("Load Shot Skeleton JSON")
        self.shot_json_btn.clicked.connect(self.load_shot_playback_json)
        left_panel.addWidget(self.shot_json_btn)

        self.shot_csv_btn = QPushButton("Load Shot CSV")
        self.shot_csv_btn.clicked.connect(self.load_shot_playback_csv)
        left_panel.addWidget(self.shot_csv_btn)

        # Add shot overview chart
        left_panel.addWidget(QLabel("Shot Overview (Click to navigate):"))
        self.shot_overview_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        left_panel.addWidget(self.shot_overview_canvas, 1)

        # Add current shot/phase info
        self.current_shot_info_label = QLabel("No data loaded")
        self.current_shot_info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        left_panel.addWidget(self.current_shot_info_label)

        main_layout.addLayout(left_panel, 1)

        # --- Center Panel: Skeleton Playback and Virtual Target ---
        center_panel = QVBoxLayout()

        # --- Comparison Controls ---
        compare_controls_layout = QHBoxLayout()
        self.compare_shots_toggle = QCheckBox("Compare Shots")
        self.compare_shots_toggle.setChecked(False)
        self.compare_shots_toggle.toggled.connect(self.on_compare_shots_toggled)
        compare_controls_layout.addWidget(self.compare_shots_toggle)
        compare_controls_layout.addSpacing(10)
        self.compare_shot_dropdown = QComboBox()
        self.compare_shot_dropdown.setEnabled(False)
        self.compare_shot_dropdown.setVisible(False)
        self.compare_shot_dropdown.currentIndexChanged.connect(self.on_compare_shot_selected)
        compare_controls_layout.addWidget(QLabel("Compare with Shot:"))
        compare_controls_layout.addWidget(self.compare_shot_dropdown)
        compare_controls_layout.addSpacing(10)
        self.sync_sliders_checkbox = QCheckBox("Sync Sliders")
        self.sync_sliders_checkbox.setChecked(False)
        self.sync_sliders_checkbox.setEnabled(False)
        self.sync_sliders_checkbox.setVisible(False)
        self.sync_sliders_checkbox.stateChanged.connect(self.on_sync_sliders_toggled)
        compare_controls_layout.addWidget(self.sync_sliders_checkbox)
        compare_controls_layout.addStretch()
        center_panel.addLayout(compare_controls_layout)

        # Skeleton playback display
        self.shot_playback_display = DrawableDisplayWidget()
        self.shot_playback_display.setMinimumSize(640, 480)
        
        center_panel.addWidget(self.shot_playback_display, 4)

        # Playback controls
        self.shot_playback_controls = VideoControls()
        center_panel.addWidget(self.shot_playback_controls)

        # Virtual target visualization (replacing phase duration chart)
        virtual_target_label = QLabel("Virtual Target Visualization")
        virtual_target_label.setAlignment(Qt.AlignCenter)
        virtual_target_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        center_panel.addWidget(virtual_target_label)
        
        # Add filter controls for virtual target
        filter_layout = QHBoxLayout()
        
        # Shot range filter
        shot_range_layout = QHBoxLayout()
        shot_range_layout.addWidget(QLabel("Shot Range:"))
        self.shot_playback_min_spin = QSpinBox()
        self.shot_playback_min_spin.setRange(0, 1000)
        self.shot_playback_min_spin.setValue(0)
        self.shot_playback_min_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        shot_range_layout.addWidget(self.shot_playback_min_spin)
        shot_range_layout.addWidget(QLabel("to"))
        self.shot_playback_max_spin = QSpinBox()
        self.shot_playback_max_spin.setRange(0, 1000)
        self.shot_playback_max_spin.setValue(1000)
        self.shot_playback_max_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        shot_range_layout.addWidget(self.shot_playback_max_spin)
        filter_layout.addLayout(shot_range_layout)
        
        filter_layout.addSpacing(20)
        
        # Score range filter
        score_range_layout = QHBoxLayout()
        score_range_layout.addWidget(QLabel("Score Range:"))
        self.shot_playback_score_min_spin = QDoubleSpinBox()
        self.shot_playback_score_min_spin.setRange(0.0, 10.0)
        self.shot_playback_score_min_spin.setValue(0.0)
        self.shot_playback_score_min_spin.setDecimals(1)
        self.shot_playback_score_min_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        score_range_layout.addWidget(self.shot_playback_score_min_spin)
        score_range_layout.addWidget(QLabel("to"))
        self.shot_playback_score_max_spin = QDoubleSpinBox()
        self.shot_playback_score_max_spin.setRange(0.0, 10.0)
        self.shot_playback_score_max_spin.setValue(10.0)
        self.shot_playback_score_max_spin.setDecimals(1)
        self.shot_playback_score_max_spin.valueChanged.connect(self.update_shot_playback_virtual_target_filters)
        score_range_layout.addWidget(self.shot_playback_score_max_spin)
        filter_layout.addLayout(score_range_layout)
        
        filter_layout.addStretch()
        
        # Clear filters button
        self.clear_virtual_target_filters_btn = QPushButton("Clear Filters")
        self.clear_virtual_target_filters_btn.clicked.connect(self.clear_shot_playback_filters)
        filter_layout.addWidget(self.clear_virtual_target_filters_btn)
        
        center_panel.addLayout(filter_layout)
        
        # Add zoom controls for the virtual target
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Target Zoom:"))
        
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setToolTip("Zoom in on target")
        zoom_in_btn.clicked.connect(self.zoom_in_virtual_target)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setToolTip("Zoom out from target")
        zoom_out_btn.clicked.connect(self.zoom_out_virtual_target)
        zoom_layout.addWidget(zoom_out_btn)
        
        reset_zoom_btn = QPushButton("🔄")
        reset_zoom_btn.setToolTip("Reset zoom to default")
        reset_zoom_btn.clicked.connect(self.reset_virtual_target_zoom)
        zoom_layout.addWidget(reset_zoom_btn)
        
        zoom_layout.addStretch()
        center_panel.addLayout(zoom_layout)
        
        # Create a scroll area for the virtual target canvas to handle overflow
        virtual_target_scroll = QScrollArea()
        virtual_target_scroll.setWidgetResizable(True)
        virtual_target_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        virtual_target_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        virtual_target_scroll.setMinimumSize(640, 480)  # Same size as skeleton display
        
        # Create the virtual target canvas with larger size
        self.virtual_target_canvas = FigureCanvas(Figure(figsize=(12, 10)))
        virtual_target_scroll.setWidget(self.virtual_target_canvas)
        center_panel.addWidget(virtual_target_scroll, 4)  # Same stretch as skeleton display

        main_layout.addLayout(center_panel, 3)

        # --- State ---
        self.shot_playback_json_data = None
        self.shot_playback_csv_df = None
        self.shot_playback_frames_by_shot = {}
        self.shot_playback_timer = QTimer(self)
        self.shot_playback_timer.timeout.connect(self.advance_shot_playback_frame)
        self.shot_playback_current_frames = []
        self.shot_playback_current_frame_idx = 0
        self.shot_playback_fps = 15
        self.shot_playback_current_shot = None
        self.shot_playback_current_phase = None

        # --- Comparison State ---
        self.compare_mode_enabled = False
        self.compare_shot_number = None
        self.compare_shot_frames = []
        self.compare_frame_idx = 0  # Independent frame index for comparison shot

        # Connect controls
        self.shot_playback_controls.play_pause_btn.clicked.connect(self.toggle_shot_playback)
        self.shot_playback_controls.stop_btn.clicked.connect(self.reset_shot_playback)
        self.shot_playback_controls.speed_slider.valueChanged.connect(self.set_shot_playback_speed)
        self.shot_playback_controls.progress_slider.sliderMoved.connect(self.shot_playback_slider_moved)
        self.shot_playback_controls.progress_slider.valueChanged.connect(self.shot_playback_slider_value_changed)
        self.shot_playback_controls.forward_btn.clicked.connect(lambda: self.seek_shot_playback_relative(15))
        self.shot_playback_controls.backward_btn.clicked.connect(lambda: self.seek_shot_playback_relative(-15))
        
        # Connect comparison slider with debug prints
        print("Connecting comparison slider signals...")
        # Disconnect any existing connections from VideoControls
        try:
            self.shot_playback_controls.compare_progress_slider.sliderMoved.disconnect()
        except:
            pass
        try:
            self.shot_playback_controls.compare_progress_slider.valueChanged.disconnect()
        except:
            pass
        
        # Connect to our custom methods
        self.shot_playback_controls.compare_progress_slider.sliderMoved.connect(self.compare_shot_playback_slider_moved)
        self.shot_playback_controls.compare_progress_slider.valueChanged.connect(self.compare_shot_playback_slider_value_changed)
        print("Comparison slider signals connected to custom methods")

        # Connect canvas click events for navigation
        self.shot_overview_canvas.mpl_connect('button_press_event', self.on_shot_overview_click)
        
        # Add a test button to manually trigger comparison slider (for debugging)
        test_btn = QPushButton("Test Compare Slider")
        test_btn.clicked.connect(self.test_compare_slider)
        center_panel.addWidget(test_btn)
        
        # Add a test button to manually trigger virtual target visualization
        virtual_target_test_btn = QPushButton("Test Virtual Target")
        virtual_target_test_btn.clicked.connect(lambda: self.update_shot_playback_virtual_target(1))
        center_panel.addWidget(virtual_target_test_btn)

    def on_compare_shots_toggled(self, checked):
        self.compare_mode_enabled = checked
        self.compare_shot_dropdown.setEnabled(checked)
        self.compare_shot_dropdown.setVisible(checked)
        self.sync_sliders_checkbox.setEnabled(checked)
        self.sync_sliders_checkbox.setVisible(checked)
        if checked:
            self.shot_playback_controls.compare_slider_label.setVisible(True)
            self.shot_playback_controls.compare_progress_slider.setVisible(True)
            self.shot_playback_controls.compare_slider_label.setText("Comparison Shot")
            if self.compare_shot_frames:
                self.shot_playback_controls.compare_progress_slider.setRange(0, max(0, len(self.compare_shot_frames)-1))
                self.shot_playback_controls.compare_progress_slider.setValue(0)
        else:
            self.shot_playback_controls.compare_slider_label.setVisible(False)
            self.shot_playback_controls.compare_progress_slider.setVisible(False)
            self.sync_sliders_checkbox.setChecked(False)
            self.compare_shot_number = None
            self.compare_shot_frames = []
            self.compare_frame_idx = 0
            self.draw_shot_playback_current_frame()

    def on_compare_shot_selected(self, idx):
        if not self.compare_mode_enabled or self.shot_playback_csv_df is None:
            return
        shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
        if 0 <= idx < len(shot_numbers):
            self.compare_shot_number = shot_numbers[idx]
            self.compare_shot_frames = self.shot_playback_frames_by_shot.get(self.compare_shot_number, [])
            print(f"Selected comparison shot {self.compare_shot_number} with {len(self.compare_shot_frames)} frames")
            # Set the comparison slider range and reset frame index
            if self.compare_shot_frames:
                max_frame = max(0, len(self.compare_shot_frames)-1)
                self.shot_playback_controls.compare_progress_slider.setRange(0, max_frame)
                self.shot_playback_controls.compare_progress_slider.setValue(0)
                self.compare_frame_idx = 0
                print(f"Set comparison slider range: 0 to {max_frame}")
            self.draw_shot_playback_current_frame()
            # Update phase chart for both shots
            if self.shot_playback_current_shot is not None:
                self.update_shot_playback_phase_chart(self.shot_playback_current_shot, self.compare_shot_number)
            self.calculate_and_set_phase_markers()

    def load_shot_playback_json(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Skeleton JSON", "", "JSON Files (*.json)")
        if not filepath:
            return
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Accepts both {"frames": [...]} and just a list
        frames = data.get("frames", data) if isinstance(data, dict) else data
        # Group frames by shot number (if available), else by phase or just index
        frames_by_shot = {}
        for idx, frame in enumerate(frames):
            # Try to get shot number from frame metadata, fallback to index
            shot_num = None
            if "shot_number" in frame:
                shot_num = int(frame["shot_number"])
            elif "Shot_Number" in frame:
                shot_num = int(frame["Shot_Number"])
            elif "phase" in frame and frame["phase"] is not None:
                shot_num = str(frame["phase"])
            else:
                shot_num = int(idx // 100 + 1)  # crude fallback: every 100 frames = new shot
            frames_by_shot.setdefault(shot_num, []).append(frame)
        self.shot_playback_json_data = frames
        self.shot_playback_frames_by_shot = frames_by_shot
        self.update_shot_playback_shot_list()

    def load_shot_playback_csv(self):
        # First check if we have data from the main system that we can use
        if hasattr(self, 'combined_df') and self.combined_df is not None:
            # Check if the combined data has the required columns for shot playback
            required_columns = ['Shot_Number', 'Preparing_Time(s)', 'Aiming_Time(s)', 'After_Shot_Time(s)']
            if all(col in self.combined_df.columns for col in required_columns):
                # Use the existing data
                self.shot_playback_csv_df = self.combined_df.copy()
                self.update_shot_playback_shot_list()
                QMessageBox.information(self, "Data Loaded", 
                                      f"Using existing data with {len(self.shot_playback_csv_df)} shots for shot playback analysis.")
                return
        
        # If no suitable data is available, prompt for file selection
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Shot CSV", "", "CSV Files (*.csv)")
        if not filepath:
            return
        import pandas as pd
        self.shot_playback_csv_df = pd.read_csv(filepath)
        self.update_shot_playback_shot_list()

    def update_shot_playback_shot_list(self):
        # Only update if both JSON and CSV are loaded
        if self.shot_playback_csv_df is None or not self.shot_playback_frames_by_shot:
            return
        
        # Create horizontal stacked bar chart showing all shots
        self.shot_overview_canvas.figure.clf()
        ax = self.shot_overview_canvas.figure.add_subplot(111)
        
        shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
        phases = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
        colors = PHASE_COLORS_LIST
        
        # Prepare data for stacked bars
        bottom = np.zeros(len(shot_numbers))
        
        for i, phase in enumerate(phases):
            values = []
            for shot_num in shot_numbers:
                row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == shot_num]
                if not row.empty:
                    value = float(row.iloc[0].get(phase, 0))
                else:
                    value = 0
                values.append(value)
            
            ax.barh(shot_numbers, values, left=bottom, 
                   color=colors[i], alpha=0.8, 
                   label=phase.replace("_Time(s)", ""))
            bottom += np.array(values)
        
        # Add score annotations
        for i, shot_num in enumerate(shot_numbers):
            row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == shot_num]
            if not row.empty:
                score = row.iloc[0].get('Score', '')
                if score:
                    total_width = bottom[i]
                    ax.text(total_width + 0.1, shot_num, f"Score: {score}", 
                           va='center', ha='left', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Shot Number")
        ax.set_title("Shot Overview - Phase Durations and Scores")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')
        
        self.shot_overview_canvas.figure.tight_layout()
        self.shot_overview_canvas.draw()
        
        # Auto-select first shot if available
        if shot_numbers and shot_numbers[0] in self.shot_playback_frames_by_shot:
            self.shot_playback_current_shot = shot_numbers[0]
            self.shot_playback_current_frames = self.shot_playback_frames_by_shot[shot_numbers[0]]
            self.shot_playback_current_frame_idx = 0
            self.shot_playback_controls.set_total_frames(len(self.shot_playback_current_frames))
            # Calculate and set phase markers for the first shot
            self.calculate_and_set_phase_markers(self.shot_playback_current_frames)
            self.draw_shot_playback_current_frame()
            # If compare mode, show both shots in phase chart
            if getattr(self, 'compare_mode_enabled', False) and self.compare_shot_number is not None:
                self.update_shot_playback_phase_chart(shot_numbers[0], self.compare_shot_number)
            else:
                self.update_shot_playback_phase_chart(shot_numbers[0])
            self.update_current_shot_info()
        
        # --- Populate compare shot dropdown ---
        if hasattr(self, 'compare_shots_toggle') and self.compare_shots_toggle.isChecked():
            self.populate_compare_shot_dropdown()

    def on_shot_overview_click(self, event):
        """Handle clicks on the shot overview chart to navigate to specific shots"""
        if event.inaxes != self.shot_overview_canvas.figure.axes[0]:
            return
        
        # Get the shot number from the y-axis position (since it's a horizontal bar chart)
        shot_number = int(round(event.ydata))
        
        # Verify the shot number exists in our data
        if shot_number in self.shot_playback_frames_by_shot:
            # Find the frame index for this shot
            frames = self.shot_playback_frames_by_shot[shot_number]
            if frames:
                self.shot_playback_current_frames = frames
                self.shot_playback_current_frame_idx = 0
                self.shot_playback_current_shot = shot_number
                self.shot_playback_controls.set_total_frames(len(frames))
                
                # Calculate and set phase markers
                self.calculate_and_set_phase_markers(frames)
                
                self.draw_shot_playback_current_frame()
                self.update_shot_playback_phase_chart(shot_number)
                self.update_current_shot_info()
                
                # Highlight the shot in the virtual target for bidirectional highlighting
                self.highlight_shot_in_virtual_target(shot_number)
                
                # Update compare shot dropdown
                if hasattr(self, 'compare_shots_toggle') and self.compare_shots_toggle.isChecked():
                    self.populate_compare_shot_dropdown()

    def populate_compare_shot_dropdown(self):
        if self.shot_playback_csv_df is None or self.shot_playback_current_shot is None:
            self.compare_shot_dropdown.clear()
            self.compare_shot_dropdown.setEnabled(False)
            self.compare_shot_dropdown.setVisible(False)
            return
        shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
        # Exclude the current primary shot
        compare_options = [str(s) for s in shot_numbers if s != self.shot_playback_current_shot]
        self.compare_shot_dropdown.blockSignals(True)
        self.compare_shot_dropdown.clear()
        self.compare_shot_dropdown.addItems(compare_options)
        self.compare_shot_dropdown.setEnabled(True)
        self.compare_shot_dropdown.setVisible(True)
        self.compare_shot_dropdown.blockSignals(False)
        # Optionally, auto-select the first available shot
        if compare_options:
            self.compare_shot_dropdown.setCurrentIndex(0)
            self.on_compare_shot_selected(0)
        else:
            self.compare_shot_number = None
            self.compare_shot_frames = []
            self.draw_shot_playback_current_frame()

    def draw_shot_playback_current_frame(self):
        if not self.shot_playback_current_frames or not (0 <= self.shot_playback_current_frame_idx < len(self.shot_playback_current_frames)):
            self.shot_playback_display.set_image(None)
            self.shot_playback_controls.update_frame_display(0, len(self.shot_playback_current_frames))
            return
        
        frame = self.shot_playback_current_frames[self.shot_playback_current_frame_idx]
        
        # --- Comparison: get compare frame by independent position ---
        compare_frame = None
        compare_phase = None
        if getattr(self, 'compare_mode_enabled', False) and self.compare_shot_frames:
            if 0 <= self.compare_frame_idx < len(self.compare_shot_frames):
                compare_frame = self.compare_shot_frames[self.compare_frame_idx]
                compare_phase = compare_frame.get("phase", compare_frame.get("Phase", "Unknown"))
                print(f"Drawing comparison frame {self.compare_frame_idx} with phase {compare_phase}")
            else:
                print(f"Compare frame index {self.compare_frame_idx} out of range (0-{len(self.compare_shot_frames)-1})")
        
        # Update current phase information
        self.shot_playback_current_phase = frame.get("phase", frame.get("Phase", "Unknown"))
        self.update_current_shot_info()
        
        # Draw skeleton on a QImage
        img_width, img_height = self.shot_playback_display.width(), self.shot_playback_display.height()
        if img_width <= 0 or img_height <= 0:
            img_width, img_height = 640, 480
        
        from PyQt5.QtGui import QImage, QPainter, QPen, QColor
        import numpy as np
        image = QImage(img_width, img_height, QImage.Format_RGB888)
        image.fill(Qt.black)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate aspect ratio preservation for skeleton
        skeleton_aspect_ratio = 4/3  # Standard skeleton aspect ratio (width/height)
        widget_aspect_ratio = img_width / img_height if img_height > 0 else 1
        
        if widget_aspect_ratio > skeleton_aspect_ratio:
            # Widget is wider - fit to height
            x_offset = (img_width - img_height * skeleton_aspect_ratio) / 2
            y_offset = 0
            skeleton_width = img_height * skeleton_aspect_ratio
            skeleton_height = img_height
        else:
            # Widget is taller - fit to width  
            x_offset = 0
            y_offset = (img_height - img_width / skeleton_aspect_ratio) / 2
            skeleton_width = img_width
            skeleton_height = img_width / skeleton_aspect_ratio
        
        # Draw phase information at the top
        painter.setPen(QPen(Qt.white, 2))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        # Get current shot score
        current_score = "N/A"
        if self.shot_playback_csv_df is not None:
            row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == self.shot_playback_current_shot]
            if not row.empty:
                current_score = row.iloc[0].get('Score', 'N/A')
        
        phase_text = f"Shot {self.shot_playback_current_shot} | Phase: {self.shot_playback_current_phase} | Score: {current_score}"
        if compare_frame is not None and self.compare_shot_number is not None:
            # Get compare shot score
            compare_score = "N/A"
            if self.shot_playback_csv_df is not None:
                compare_row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == self.compare_shot_number]
                if not compare_row.empty:
                    compare_score = compare_row.iloc[0].get('Score', 'N/A')
            phase_text += f"    |    Shot {self.compare_shot_number} | Phase: {compare_phase} | Score: {compare_score}"
        painter.drawText(10, 30, phase_text)
        
        # Draw legend
        if compare_frame is not None and self.compare_shot_number is not None:
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.white)
            painter.drawRect(10, 40, 18, 18)
            painter.setPen(Qt.white)
            painter.drawText(32, 54, f"Shot {self.shot_playback_current_shot}")
            painter.setPen(QPen(QColor(0, 128, 255), 2))
            painter.setBrush(QColor(0, 128, 255, 180))
            painter.drawRect(10, 62, 18, 18)
            painter.setPen(QColor(0, 128, 255))
            painter.drawText(32, 76, f"Shot {self.compare_shot_number}")
        
        # Draw comparison skeleton (blue, semi-transparent)
        if compare_frame is not None:
            landmarks = compare_frame.get("landmarks") or compare_frame.get("Landmarks")
            if landmarks:
                points = []
                for lm in landmarks:
                    if isinstance(lm, dict):
                        x = lm.get("x", lm.get("X"))
                        y = lm.get("y", lm.get("Y"))
                    elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                        x, y = lm[0], lm[1]
                    else:
                        continue
                    if x is not None and y is not None:
                        px = int(float(x) * skeleton_width + x_offset)
                        py = int(float(y) * skeleton_height + y_offset)
                        points.append((px, py))
                # Draw connections (MediaPipe style)
                blue = QColor(0, 128, 255, 180)
                painter.setPen(QPen(blue, 3))
                for p1, p2 in POSE_CONNECTIONS:
                    if p1 < len(points) and p2 < len(points):
                        painter.drawLine(points[p1][0], points[p1][1], points[p2][0], points[p2][1])
                painter.setPen(QPen(blue, 1))
                painter.setBrush(blue)
                for px, py in points:
                    painter.drawEllipse(px-4, py-4, 8, 8)
        
        # Draw primary skeleton (white, as before)
        landmarks = frame.get("landmarks") or frame.get("Landmarks")
        if landmarks:
            points = []
            for lm in landmarks:
                if isinstance(lm, dict):
                    x = lm.get("x", lm.get("X"))
                    y = lm.get("y", lm.get("Y"))
                elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                    x, y = lm[0], lm[1]
                else:
                    continue
                if x is not None and y is not None:
                    px = int(float(x) * skeleton_width + x_offset)
                    py = int(float(y) * skeleton_height + y_offset)
                    points.append((px, py))
            phase_colors = UNIFIED_PHASE_QCOLORS
            current_phase_lower = self.shot_playback_current_phase.lower() if self.shot_playback_current_phase else 'unknown'
            skeleton_color = phase_colors.get(current_phase_lower, phase_colors['unknown'])
            painter.setPen(QPen(skeleton_color, 3))
            for p1, p2 in POSE_CONNECTIONS:
                if p1 < len(points) and p2 < len(points):
                    painter.drawLine(points[p1][0], points[p1][1], points[p2][0], points[p2][1])
            painter.setPen(QPen(skeleton_color, 1))
            painter.setBrush(skeleton_color)
            for px, py in points:
                painter.drawEllipse(px-4, py-4, 8, 8)
        
        # Draw small target visualization in top-right corner
        self.draw_small_target_visualization(painter, img_width, img_height, self.shot_playback_current_shot, compare_shot_number=self.compare_shot_number if compare_frame is not None else None)
        
        painter.end()
        self.shot_playback_display.set_image(image)
        self.shot_playback_controls.update_frame_display(self.shot_playback_current_frame_idx, len(self.shot_playback_current_frames))

    def toggle_shot_playback(self, checked):
        if not self.shot_playback_current_frames:
            self.shot_playback_controls.play_pause_btn.setChecked(False)
            self.shot_playback_controls.play_pause_btn.setText("▶")
            return
        if checked:
            self.shot_playback_controls.play_pause_btn.setText("⏸")
            speed = self.shot_playback_controls.speed_slider.value() / 100.0
            interval = int(1000 / (self.shot_playback_fps * speed))
            self.shot_playback_timer.start(max(20, interval))
        else:
            self.shot_playback_controls.play_pause_btn.setText("▶")
            self.shot_playback_timer.stop()

    def reset_shot_playback(self):
        self.shot_playback_timer.stop()
        self.shot_playback_controls.play_pause_btn.setChecked(False)
        self.shot_playback_controls.play_pause_btn.setText("▶")
        self.shot_playback_current_frame_idx = 0
        self.draw_shot_playback_current_frame()

    def set_shot_playback_speed(self, value):
        if self.shot_playback_timer.isActive():
            speed = value / 100.0
            interval = int(1000 / (self.shot_playback_fps * speed))
            self.shot_playback_timer.setInterval(max(20, interval))

    def shot_playback_slider_moved(self, frame_index):
        if self._syncing_sliders:
            return
        if 0 <= frame_index < len(self.shot_playback_current_frames):
            self.shot_playback_current_frame_idx = frame_index
            # Update comparison slider to keep in sync only if sync is enabled
            if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
                n1 = len(self.shot_playback_current_frames)
                n2 = len(self.compare_shot_frames)
                rel = frame_index / max(1, n1-1)
                compare_idx = int(round(rel * (n2-1)))
                self._syncing_sliders = True
                self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
                self._syncing_sliders = False
                self.compare_frame_idx = compare_idx
            # Update both phase bars
            self.calculate_and_set_phase_markers()
            self.draw_shot_playback_current_frame()
            if self.shot_playback_controls.play_pause_btn.isChecked():
                self.shot_playback_timer.stop()
                QTimer.singleShot(50, lambda: self.shot_playback_timer.start() if self.shot_playback_controls.play_pause_btn.isChecked() else None)

    def shot_playback_slider_value_changed(self, frame_index):
        if self._syncing_sliders:
            return
        if not self.shot_playback_controls.progress_slider.isSliderDown():
            if 0 <= frame_index < len(self.shot_playback_current_frames):
                self.shot_playback_current_frame_idx = frame_index
                # Update comparison slider to keep in sync only if sync is enabled
                if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
                    n1 = len(self.shot_playback_current_frames)
                    n2 = len(self.compare_shot_frames)
                    rel = frame_index / max(1, n1-1)
                    compare_idx = int(round(rel * (n2-1)))
                    self._syncing_sliders = True
                    self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
                    self._syncing_sliders = False
                    self.compare_frame_idx = compare_idx
                # Update both phase bars
                self.calculate_and_set_phase_markers()
                self.draw_shot_playback_current_frame()

    def seek_shot_playback_relative(self, frame_offset):
        if not self.shot_playback_current_frames:
            return
        new_idx = self.shot_playback_current_frame_idx + frame_offset
        self.shot_playback_current_frame_idx = max(0, min(new_idx, len(self.shot_playback_current_frames) - 1))
        # Update comparison slider to keep in sync only if sync is enabled
        if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
            n1 = len(self.shot_playback_current_frames)
            n2 = len(self.compare_shot_frames)
            rel = self.shot_playback_current_frame_idx / max(1, n1-1)
            compare_idx = int(round(rel * (n2-1)))
            self._syncing_sliders = True
            self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
            self._syncing_sliders = False
            self.compare_frame_idx = compare_idx
        self.draw_shot_playback_current_frame()
        self.shot_playback_controls.progress_slider.setValue(self.shot_playback_current_frame_idx)
        if self.shot_playback_controls.play_pause_btn.isChecked():
            self.shot_playback_timer.stop()
            QTimer.singleShot(50, lambda: self.shot_playback_timer.start() if self.shot_playback_controls.play_pause_btn.isChecked() else None)

    def advance_shot_playback_frame(self):
        if not self.shot_playback_current_frames:
            return
        self.shot_playback_current_frame_idx += 1
        if self.shot_playback_current_frame_idx >= len(self.shot_playback_current_frames):
            self.shot_playback_current_frame_idx = 0
        # Update comparison slider to keep in sync only if sync is enabled
        if self.compare_shot_frames and self.sync_sliders_checkbox.isChecked():
            n1 = len(self.shot_playback_current_frames)
            n2 = len(self.compare_shot_frames)
            rel = self.shot_playback_current_frame_idx / max(1, n1-1)
            compare_idx = int(round(rel * (n2-1)))
            self._syncing_sliders = True
            self.shot_playback_controls.compare_progress_slider.setValue(compare_idx)
            self._syncing_sliders = False
            self.compare_frame_idx = compare_idx
        self.draw_shot_playback_current_frame()

    def on_sync_sliders_toggled(self, checked):
        """Handle sync sliders checkbox toggle - sync from current positions"""
        if checked and self.compare_shot_frames:
            # When sync is enabled, keep sliders at their current relative positions
            # Don't force them to start from beginning - let the existing sync logic handle it
            pass

    def update_shot_playback_phase_chart(self, shot_num, compare_shot_num=None):
        """
        Update the phase duration bar at the bottom. If compare_shot_num is provided, show both shots side by side.
        """
        # Check if old canvas exists, redirect if not
        if not hasattr(self, 'shot_phase_chart_canvas'):
            print("Redirecting to virtual target - old canvas not found")
            self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
            return
            
        if self.shot_playback_csv_df is None:
            return
        
        # New color scheme
        phase_labels = ["Preparing", "Aiming", "After Shot"]
        phases = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
        colors = PHASE_COLORS_LIST
        
        # Get primary shot data
        row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == int(shot_num)]
        if row.empty:
            try:
                self.shot_phase_chart_canvas.figure.clf()
                self.shot_phase_chart_canvas.draw()
            except AttributeError:
                print("Redirecting to virtual target - canvas error")
                self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
            return
        row = row.iloc[0]
        values = [float(row.get(phase, 0)) for phase in phases]
        
        # If compare_shot_num is provided, get its data
        compare_values = None
        if compare_shot_num is not None:
            compare_row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == int(compare_shot_num)]
            if not compare_row.empty:
                compare_row = compare_row.iloc[0]
                compare_values = [float(compare_row.get(phase, 0)) for phase in phases]
        
        try:
            self.shot_phase_chart_canvas.figure.clf()  # Clear the figure
            ax = self.shot_phase_chart_canvas.figure.add_subplot(111)  # Create a new axes
        except AttributeError:
            print("Redirecting to virtual target - canvas error")
            self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
            return
        
        if compare_values is not None:
            # Grouped bar chart: primary and comparison
            bar_width = 0.35
            x = np.arange(len(phases))
            bars1 = ax.bar(x - bar_width/2, values, width=bar_width, color=colors, label=f"Shot {shot_num}")
            bars2 = ax.bar(x + bar_width/2, compare_values, width=bar_width, color=colors, alpha=0.5, label=f"Shot {compare_shot_num}")
            for i, (bar, val) in enumerate(zip(bars1, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.2f}s", ha='center', va='center', color='white', fontsize=12)
            for i, (bar, val) in enumerate(zip(bars2, compare_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.2f}s", ha='center', va='center', color='black', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(phase_labels)
            ax.legend()
            ax.set_title(f"Shot {shot_num} vs Shot {compare_shot_num} Phase Durations")
            ax.set_ylim(0, max(values + compare_values) * 1.2 if values + compare_values else 1)
        else:
            # Single shot
            bars = ax.bar(phase_labels, values, color=colors)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{val:.2f}s", ha='center', va='center', color='white', fontsize=12)
            ax.set_ylabel("Time (s)")
            ax.set_ylim(0, max(values) * 1.2 if values else 1)
            ax.set_title(f"Shot {shot_num} Phase Durations")
        try:
            self.shot_phase_chart_canvas.figure.tight_layout()
            self.shot_phase_chart_canvas.draw()
        except AttributeError:
            print("Redirecting to virtual target - canvas error")
            self.update_shot_playback_virtual_target(shot_num, compare_shot_num)
    def on_score_edited(self, row, column):
        col_name = self.data_table.horizontalHeaderItem(column).text()
        if col_name == "Score":
            # Find the column index for Shot_Number
            shot_col = None
            for i in range(self.data_table.columnCount()):
                if self.data_table.horizontalHeaderItem(i).text() == "Shot_Number":
                    shot_col = i
                    break
            if shot_col is None:
                return  # Can't find Shot_Number column
            shot_number_item = self.data_table.item(row, shot_col)
            if shot_number_item is None:
                return
            try:
                shot_number = int(shot_number_item.text())
            except ValueError:
                return
            new_score = self.data_table.item(row, column).text()
            
            # Update the combined DataFrame
            self.combined_df.loc[self.combined_df['Shot_Number'] == shot_number, 'Score'] = new_score
            
            # Provide visual feedback that changes were made
            self.save_data_btn.setText("Save Changes *")
            self.save_data_btn.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")
            
            # Update the shot overview chart if we're in shot playback tab
            if hasattr(self, 'shot_playback_csv_df') and self.shot_playback_csv_df is not None:
                self.update_shot_playback_shot_list()
        
        elif col_name == "Direction":
            # Find the column index for Shot_Number
            shot_col = None
            for i in range(self.data_table.columnCount()):
                if self.data_table.horizontalHeaderItem(i).text() == "Shot_Number":
                    shot_col = i
                    break
            if shot_col is None:
                return  # Can't find Shot_Number column
            shot_number_item = self.data_table.item(row, shot_col)
            if shot_number_item is None:
                return
            try:
                shot_number = int(shot_number_item.text())
            except ValueError:
                return
            
            new_direction = self.data_table.item(row, column).text()
            
            # Validate direction - allow empty or one of the 8 directions
            valid_directions = ["", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]
            if new_direction not in valid_directions:
                # Reset to empty if invalid
                self.data_table.item(row, column).setText("")
                QMessageBox.warning(self, "Invalid Direction", 
                                  f"Please enter one of the valid directions: {', '.join(valid_directions[1:])} or leave empty")
                return
            
            # Update the combined DataFrame
            self.combined_df.loc[self.combined_df['Shot_Number'] == shot_number, 'Direction'] = new_direction
            
            # Provide visual feedback that changes were made
            self.save_data_btn.setText("Save Changes *")
            self.save_data_btn.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")

    def update_current_shot_info(self):
        """Update the current shot/phase information display"""
        if self.shot_playback_current_shot is not None and self.shot_playback_current_phase is not None:
            info_text = f"Shot {self.shot_playback_current_shot} | Phase: {self.shot_playback_current_phase}"
            if self.shot_playback_csv_df is not None:
                row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == self.shot_playback_current_shot]
                if not row.empty:
                    score = row.iloc[0].get('Score', 'N/A')
                    info_text += f" | Score: {score}"
            self.current_shot_info_label.setText(info_text)
        else:
            self.current_shot_info_label.setText("No data loaded")

    def save_data_table_changes(self):
        """Save the current data table changes to CSV files"""
        if self.combined_df is None:
            QMessageBox.warning(self, "No Data", "No data to save. Please load data first.")
            return
        
        try:
            # Apply current filters to get the data that should be saved
            filtered_df = self.combined_df.copy()
            
            # Apply player filter
            if self.player_combo.currentText() != "All Players":
                filtered_df = filtered_df[filtered_df['Player'] == self.player_combo.currentText()]
            
            # Apply session filter
            if self.session_combo.currentText() != "All Sessions":
                filtered_df = filtered_df[filtered_df['Session'] == self.session_combo.currentText()]
            
            # Apply shot range filter
            if 'Shot_Number' in filtered_df.columns:
                shot_min = self.shot_min_spin.value()
                shot_max = self.shot_max_spin.value()
                filtered_df = filtered_df[
                    (filtered_df['Shot_Number'] >= shot_min) & 
                    (filtered_df['Shot_Number'] <= shot_max)
                ]

            # Apply complete shot filter
            if 'Complete_Shot' in filtered_df.columns and self.complete_shot_checkbox.isChecked():
                filtered_df = filtered_df[filtered_df['Complete_Shot'].str.lower() == 'yes']
            
            # Ask user where to save
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Combined CSV File", "", "CSV Files (*.csv)")
            if file_path:
                # Save the filtered data
                filtered_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Combined data saved successfully to {file_path}")
                
                # Reset save button appearance
                self.save_data_btn.setText("Save Changes")
                self.save_data_btn.setStyleSheet("")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def calculate_and_set_phase_markers(self, frames=None):
        """
        Calculate phase boundaries and set markers on the slider (dual bar if compare mode).
        If in compare mode, use the current frame index for both primary and comparison shots.
        """
        # Use current frames if not provided
        if frames is None:
            frames = self.shot_playback_current_frames
        if not frames:
            self.shot_playback_controls.set_phase_markers([])
            return
        # --- Primary shot phase data ---
        phase_data = []
        current_phase = None
        phase_start_frame = 0
        for frame_idx, frame in enumerate(frames):
            frame_phase = frame.get("phase", frame.get("Phase", "unknown"))
            if current_phase is not None and frame_phase != current_phase:
                phase_data.append({
                    'start_frame': phase_start_frame,
                    'end_frame': frame_idx,
                    'phase_name': current_phase
                })
                phase_start_frame = frame_idx
            current_phase = frame_phase
        if current_phase is not None:
            phase_data.append({
                'start_frame': phase_start_frame,
                'end_frame': len(frames),
                'phase_name': current_phase
            })
        # --- Comparison shot phase data ---
        compare_phase_data = None
        compare_total_frames = None
        if getattr(self, 'compare_mode_enabled', False) and self.compare_shot_frames and len(self.compare_shot_frames) > 0:
            compare_frames = self.compare_shot_frames
            compare_phase_data = []
            compare_current_phase = None
            compare_phase_start = 0
            for frame_idx, frame in enumerate(compare_frames):
                frame_phase = frame.get("phase", frame.get("Phase", "unknown"))
                if compare_current_phase is not None and frame_phase != compare_current_phase:
                    compare_phase_data.append({
                        'start_frame': compare_phase_start,
                        'end_frame': frame_idx,
                        'phase_name': compare_current_phase
                    })
                    compare_phase_start = frame_idx
                compare_current_phase = frame_phase
            if compare_current_phase is not None:
                compare_phase_data.append({
                    'start_frame': compare_phase_start,
                    'end_frame': len(compare_frames),
                    'phase_name': compare_current_phase
                })
            compare_total_frames = len(compare_frames)
            self.shot_playback_controls.set_dual_phase_markers(
                phase_data, len(frames), compare_phase_data, compare_total_frames
            )
        else:
            self.shot_playback_controls.set_phase_markers(phase_data, len(frames))

    def compare_shot_playback_slider_moved(self, frame_index):
        if self._syncing_sliders:
            return
        if not self.compare_shot_frames:
            return
        if 0 <= frame_index < len(self.compare_shot_frames):
            self.compare_frame_idx = frame_index
            # Sync main slider by relative position only if sync is enabled
            if self.sync_sliders_checkbox.isChecked():
                n1 = len(self.shot_playback_current_frames)
                n2 = len(self.compare_shot_frames)
                rel = frame_index / max(1, n2-1)
                main_idx = int(round(rel * (n1-1)))
                self._syncing_sliders = True
                self.shot_playback_controls.progress_slider.setValue(main_idx)
                self._syncing_sliders = False
                self.shot_playback_current_frame_idx = main_idx
            # Update both phase bars
            self.calculate_and_set_phase_markers()
            self.draw_shot_playback_current_frame()
            if self.shot_playback_controls.play_pause_btn.isChecked():
                self.shot_playback_timer.stop()
                QTimer.singleShot(50, lambda: self.shot_playback_timer.start() if self.shot_playback_controls.play_pause_btn.isChecked() else None)

    def compare_shot_playback_slider_value_changed(self, frame_index):
        if self._syncing_sliders:
            return
        if not self.shot_playback_controls.compare_progress_slider.isSliderDown():
            if 0 <= frame_index < len(self.compare_shot_frames):
                self.compare_frame_idx = frame_index
                # Sync main slider by relative position only if sync is enabled
                if self.sync_sliders_checkbox.isChecked():
                    n1 = len(self.shot_playback_current_frames)
                    n2 = len(self.compare_shot_frames)
                    rel = frame_index / max(1, n2-1)
                    main_idx = int(round(rel * (n1-1)))
                    self._syncing_sliders = True
                    self.shot_playback_controls.progress_slider.setValue(main_idx)
                    self._syncing_sliders = False
                    self.shot_playback_current_frame_idx = main_idx
                # Update both phase bars
                self.calculate_and_set_phase_markers()
                self.draw_shot_playback_current_frame()

    def test_compare_slider(self):
        """Test method to manually trigger comparison slider movement"""
        print("=== TESTING COMPARE SLIDER ===")
        if self.compare_shot_frames:
            # Move to a different frame
            new_frame = (self.compare_frame_idx + 10) % len(self.compare_shot_frames)
            print(f"Manually setting comparison slider to frame {new_frame}")
            self.shot_playback_controls.compare_progress_slider.setValue(new_frame)
        else:
            print("No comparison frames available")

    def on_direction_changed(self, row, column, text):
        """Handle direction dropdown changes"""
        # Find the column index for Shot_Number
        shot_col = None
        for i in range(self.data_table.columnCount()):
            if self.data_table.horizontalHeaderItem(i).text() == "Shot_Number":
                shot_col = i
                break
        if shot_col is None:
            return  # Can't find Shot_Number column
        shot_number_item = self.data_table.item(row, shot_col)
        if shot_number_item is None:
            return
        try:
            shot_number = int(shot_number_item.text())
        except ValueError:
            return
        
        # Update the combined DataFrame
        self.combined_df.loc[self.combined_df['Shot_Number'] == shot_number, 'Direction'] = text
        
        # Provide visual feedback that changes were made
        self.save_data_btn.setText("Save Changes *")
        self.save_data_btn.setStyleSheet("background-color: #ffeb3b; font-weight: bold;")

    def draw_small_target_visualization(self, painter, img_width, img_height, current_shot, compare_shot_number=None):
        """Draw an enhanced target visualization showing ALL shots with current shot highlighted"""
        if self.shot_playback_csv_df is None or current_shot is None:
            return
            
        # Enhanced target visualization size and position (top-right corner) - MUCH LARGER
        target_size = 200  # Increased from 120 to 200
        target_x = img_width - target_size - 15
        target_y = 60
        
        # Get current shot data
        current_shot_data = None
        if self.shot_playback_csv_df is not None:
            row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == current_shot]
            if not row.empty:
                current_shot_data = row.iloc[0]
        
        # Get compare shot data if available
        compare_shot_data = None
        if compare_shot_number is not None and self.shot_playback_csv_df is not None:
            compare_row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == compare_shot_number]
            if not compare_row.empty:
                compare_shot_data = compare_row.iloc[0]
        
        # Draw target background
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(QColor(30, 30, 30, 200))  # Semi-transparent dark background
        painter.drawRect(target_x - 5, target_y - 5, target_size + 10, target_size + 10)
        
        # Draw target rings
        center_x = target_x + target_size // 2
        center_y = target_y + target_size // 2
        
        # Draw proper archery target with accurate scoring rings
        # Each scoring ring has equal width - 10 rings total (1-10 points)
        max_radius = (target_size // 2) - 5
        
        # Define 10 individual scoring rings with proper colors and proportions
        scoring_rings = [
            # (score, radius_ratio, color, is_outer_ring)
            (1, 1.0, QColor(255, 255, 255), True),    # 1 - White outer
            (2, 0.9, QColor(240, 240, 240), False),   # 2 - White inner  
            (3, 0.8, QColor(50, 50, 50), True),       # 3 - Black outer
            (4, 0.7, QColor(100, 100, 100), False),   # 4 - Black inner
            (5, 0.6, QColor(0, 120, 255), True),      # 5 - Blue outer
            (6, 0.5, QColor(100, 180, 255), False),   # 6 - Blue inner
            (7, 0.4, QColor(220, 0, 0), True),        # 7 - Red outer
            (8, 0.3, QColor(255, 100, 100), False),   # 8 - Red inner
            (9, 0.2, QColor(255, 215, 0), True),      # 9 - Gold outer
            (10, 0.1, QColor(255, 255, 150), False),  # 10 - Gold inner (center)
        ]
        
        # Draw rings from outside to inside
        for score, ratio, color, is_outer in scoring_rings:
            radius = int(max_radius * ratio)
            if radius > 0:
                painter.setPen(QPen(Qt.black, 2 if is_outer else 1))
                painter.setBrush(color)
                painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Add score numbers on rings for clarity
        painter.setPen(QPen(Qt.black, 1))
        painter.setFont(QFont("Arial", 7, QFont.Bold))
        for score, ratio, color, is_outer in scoring_rings:
            if score in [10, 8, 6, 4, 2]:  # Only show even numbers to avoid clutter
                radius = int(max_radius * ratio)
                if radius > 8:  # Only if there's enough space
                    # Place number at top of ring
                    text_x = center_x - 4
                    text_y = center_y - radius + 8
                    painter.drawText(text_x, text_y, str(score))
        
        # Direction mappings
        direction_angles = {
            'N': 90,    'NE': 45,   'E': 0,     'SE': 315,
            'S': 270,   'SW': 225,  'W': 180,   'NW': 135
        }
        
        # Draw ALL shots with current shot highlighted
        all_shots_data = self.shot_playback_csv_df.copy()
        
        # Apply shot range filter if available
        if hasattr(self, 'shot_playback_min_spin') and hasattr(self, 'shot_playback_max_spin'):
            shot_min = self.shot_playback_min_spin.value()
            shot_max = self.shot_playback_max_spin.value()
            if 'Shot_Number' in all_shots_data.columns:
                all_shots_data = all_shots_data[
                    (all_shots_data['Shot_Number'] >= shot_min) & 
                    (all_shots_data['Shot_Number'] <= shot_max)
                ]
        
        # Apply score range filter if available
        if hasattr(self, 'shot_playback_score_min_spin') and hasattr(self, 'shot_playback_score_max_spin'):
            score_min = self.shot_playback_score_min_spin.value()
            score_max = self.shot_playback_score_max_spin.value()
            if 'Score' in all_shots_data.columns:
                all_shots_data['Score'] = pd.to_numeric(all_shots_data['Score'], errors='coerce')
                all_shots_data = all_shots_data.dropna(subset=['Score'])
                all_shots_data = all_shots_data[
                    (all_shots_data['Score'] >= score_min) & 
                    (all_shots_data['Score'] <= score_max)
                ]
        
        # First draw all non-current shots with smaller markers
        for _, shot_row in all_shots_data.iterrows():
            shot_num_val = int(shot_row.get('Shot_Number', 0))
            if shot_num_val != int(current_shot):
                # Determine color based on score
                try:
                    score = float(shot_row.get('Score', 0))
                    if score >= 9:
                        color = QColor(0, 255, 0)  # Green for high scores
                    elif score >= 7:
                        color = QColor(255, 165, 0)  # Orange for medium scores
                    else:
                        color = QColor(100, 149, 237)  # Cornflower blue for lower scores
                except (ValueError, TypeError):
                    color = QColor(128, 128, 128)  # Gray for invalid scores
                
                self._draw_shot_on_target(painter, center_x, center_y, shot_row, 
                                        direction_angles, color, target_size, "Other")
        
        # Draw current shot with enhanced red color and larger size for better visibility
        if current_shot_data is not None:
            self._draw_shot_on_target(painter, center_x, center_y, current_shot_data, 
                                    direction_angles, QColor(255, 0, 0), target_size, "Current")
        
        # Draw comparison shot position if available (on top)
        if compare_shot_data is not None:
            self._draw_shot_on_target(painter, center_x, center_y, compare_shot_data, 
                                    direction_angles, QColor(0, 128, 255), target_size, "Compare")
        
        # Draw enhanced target label with shot count
        painter.setPen(QPen(Qt.white, 2))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        total_shots = len(all_shots_data)
        label_text = f"All Shots ({total_shots}) - Current: {current_shot}"
        painter.drawText(target_x, target_y - 12, label_text)

    def _draw_shot_on_target(self, painter, center_x, center_y, shot_data, direction_angles, color, target_size, label_prefix=""):
        """Helper method to draw a shot on the target visualization"""
        try:
            score = float(shot_data.get('Score', 0))
            direction = str(shot_data.get('Direction', '')).strip().upper()
            
            # Calculate position based on exact score - matching new accurate target rings
            max_radius = (target_size // 2) - 10
            
            # Each ring has equal width (10% of max radius)
            # Position shots accurately within their scoring zone
            if score >= 10:
                distance = max_radius * 0.05  # Ring 10 - center gold
            elif score >= 9:
                distance = max_radius * 0.15  # Ring 9 - outer gold
            elif score >= 8:
                distance = max_radius * 0.25  # Ring 8 - inner red
            elif score >= 7:
                distance = max_radius * 0.35  # Ring 7 - outer red
            elif score >= 6:
                distance = max_radius * 0.45  # Ring 6 - inner blue
            elif score >= 5:
                distance = max_radius * 0.55  # Ring 5 - outer blue
            elif score >= 4:
                distance = max_radius * 0.65  # Ring 4 - inner black
            elif score >= 3:
                distance = max_radius * 0.75  # Ring 3 - outer black
            elif score >= 2:
                distance = max_radius * 0.85  # Ring 2 - inner white
            elif score >= 1:
                distance = max_radius * 0.95  # Ring 1 - outer white
            else:
                distance = max_radius * 1.0   # Miss - outside target
            
            # Calculate angle based on direction
            if direction in direction_angles:
                angle_deg = direction_angles[direction]
                angle_rad = math.radians(angle_deg)
                
                # Calculate position
                shot_x = center_x + distance * math.cos(angle_rad)
                shot_y = center_y - distance * math.sin(angle_rad)  # Y is inverted in screen coordinates
                
                # Draw shot marker with better visibility - different sizes for different shot types
                if label_prefix == "Current":
                    marker_size = 8
                    pen_width = 3
                elif label_prefix == "Compare":
                    marker_size = 6
                    pen_width = 2
                else:  # "Other" shots
                    marker_size = 4
                    pen_width = 1
                
                # Draw outer ring for contrast
                painter.setPen(QPen(Qt.black, pen_width))
                painter.setBrush(Qt.black)
                painter.drawEllipse(int(shot_x - marker_size - 1), int(shot_y - marker_size - 1), 
                                  (marker_size + 1) * 2, (marker_size + 1) * 2)
                # Draw inner marker
                painter.setPen(QPen(color, pen_width))
                painter.setBrush(color)
                painter.drawEllipse(int(shot_x - marker_size), int(shot_y - marker_size), 
                                  marker_size * 2, marker_size * 2)
                
                # Draw direction arrow - smaller for "Other" shots to reduce clutter
                if label_prefix == "Other":
                    arrow_length = 8
                    arrowhead_length = 3
                    arrow_pen_width = 1
                else:
                    arrow_length = 15
                    arrowhead_length = 5
                    arrow_pen_width = 2
                    
                arrow_end_x = shot_x + arrow_length * math.cos(angle_rad)
                arrow_end_y = shot_y - arrow_length * math.sin(angle_rad)
                painter.setPen(QPen(color, arrow_pen_width))
                painter.drawLine(int(shot_x), int(shot_y), int(arrow_end_x), int(arrow_end_y))
                
                # Draw arrowhead
                arrow_angle1 = angle_rad + math.radians(150)
                arrow_angle2 = angle_rad - math.radians(150)
                
                ah1_x = arrow_end_x + arrowhead_length * math.cos(arrow_angle1)
                ah1_y = arrow_end_y - arrowhead_length * math.sin(arrow_angle1)
                ah2_x = arrow_end_x + arrowhead_length * math.cos(arrow_angle2)
                ah2_y = arrow_end_y - arrowhead_length * math.sin(arrow_angle2)
                
                painter.drawLine(int(arrow_end_x), int(arrow_end_y), int(ah1_x), int(ah1_y))
                painter.drawLine(int(arrow_end_x), int(arrow_end_y), int(ah2_x), int(ah2_y))
                
        except (ValueError, TypeError):
            # If score or direction is invalid, just mark center
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(center_x - 3, center_y - 3, 6, 6)

    def update_shot_playback_virtual_target(self, shot_num, compare_shot_num=None):
        """
        Update the virtual target visualization for the shot playback tab.
        Now shows all shots by default with improved compact mode for many shots.
        """
        # Prevent infinite recursion with a counter
        if not hasattr(self, '_virtual_target_update_count'):
            self._virtual_target_update_count = 0
        
        if self._virtual_target_update_count > 5:  # Max 5 updates in a row
            print("Preventing infinite loop - too many consecutive updates")
            self._virtual_target_update_count = 0
            return
        
        self._virtual_target_update_count += 1
        
        print(f"Virtual target method called with shot_num={shot_num}, compare_shot_num={compare_shot_num}")
        
        # Check if virtual target canvas exists
        if not hasattr(self, 'virtual_target_canvas'):
            print("ERROR: virtual_target_canvas does not exist!")
            self._virtual_target_update_count = 0
            return
            
        if self.shot_playback_csv_df is None:
            print("No CSV data available")
            self._virtual_target_update_count = 0
            return
        
        # Capture current zoom state BEFORE clearing the figure
        current_xlim = None
        current_ylim = None
        if hasattr(self, 'virtual_target_canvas') and self.virtual_target_canvas.figure.axes:
            current_xlim = self.virtual_target_canvas.figure.axes[0].get_xlim()
            current_ylim = self.virtual_target_canvas.figure.axes[0].get_ylim()
            print(f"Captured zoom state: xlim={current_xlim}, ylim={current_ylim}")
            
            # Also capture the title to check if it was zoomed
            current_title = self.virtual_target_canvas.figure.axes[0].get_title()
            was_zoomed = "(Zoomed)" in current_title
            print(f"Was zoomed: {was_zoomed}, Title: {current_title}")
        
        # Create virtual target visualization
        print("Creating virtual target visualization...")
        self.virtual_target_canvas.figure.clf()
        ax = self.virtual_target_canvas.figure.add_subplot(111)
        
        # Make the target larger and more detailed
        ax.set_facecolor('#f8f9fa')  # Light background
        
        # Draw target rings with larger scale for better visibility
        target_center = (0, 0)
        ring_radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Larger ring radii for better visibility
        ring_colors = ['gold', 'red', 'blue', 'black', 'white', 'white']  # Ring colors
        ring_scores = [10, 9, 8, 7, 6, 5]  # Score for each ring
        
        # Draw target rings with enhanced styling and larger linewidth
        for i, (radius, color, score) in enumerate(zip(ring_radii, ring_colors, ring_scores)):
            circle = plt.Circle(target_center, radius, fill=False, color=color, linewidth=4)
            ax.add_patch(circle)
        
        # Add ring reference legend on the right side
        legend_x = 6.5
        legend_y_start = 6.0
        legend_spacing = 0.8
        
        # Add title for the legend
        ax.text(legend_x, legend_y_start + 1.0, "Ring Reference", ha='center', va='bottom', 
               fontsize=14, fontweight='bold', color='black')
        
        # Add ring references with colored circles and labels
        for i, (radius, color, score) in enumerate(zip(ring_radii, ring_colors, ring_scores)):
            y_pos = legend_y_start - i * legend_spacing
            # Draw small reference circle
            ref_circle = plt.Circle((legend_x - 0.5, y_pos), 0.3, fill=False, color=color, linewidth=3)
            ax.add_patch(ref_circle)
            # Add score label
            ax.text(legend_x + 0.2, y_pos, f"Score {score}", ha='left', va='center', 
                   fontsize=12, fontweight='bold', color=color)
        
        # Apply filters to the data
        all_shots_data = self.shot_playback_csv_df.copy()
        
        # Apply shot range filter
        if hasattr(self, 'shot_playback_min_spin') and hasattr(self, 'shot_playback_max_spin'):
            shot_min = self.shot_playback_min_spin.value()
            shot_max = self.shot_playback_max_spin.value()
            if 'Shot_Number' in all_shots_data.columns:
                all_shots_data = all_shots_data[
                    (all_shots_data['Shot_Number'] >= shot_min) & 
                    (all_shots_data['Shot_Number'] <= shot_max)
                ]
        
        # Apply score range filter
        if hasattr(self, 'shot_playback_score_min_spin') and hasattr(self, 'shot_playback_score_max_spin'):
            score_min = self.shot_playback_score_min_spin.value()
            score_max = self.shot_playback_score_max_spin.value()
            if 'Score' in all_shots_data.columns:
                all_shots_data['Score'] = pd.to_numeric(all_shots_data['Score'], errors='coerce')
                all_shots_data = all_shots_data.dropna(subset=['Score'])
                all_shots_data = all_shots_data[
                    (all_shots_data['Score'] >= score_min) & 
                    (all_shots_data['Score'] <= score_max)
                ]
        
        shot_positions = []
        
        # Clear shot positions data for click detection
        self.shot_positions_data = []
        
        # Determine if we need compact mode based on number of shots
        total_shots = len(all_shots_data)
        compact_mode = total_shots >= 30
        
        # Adjust marker sizes based on compact mode - larger for better clickability
        current_shot_size = 200 if compact_mode else 400
        other_shot_size = 100 if compact_mode else 200
        label_fontsize = 6 if compact_mode else 10
        
        for _, shot_row in all_shots_data.iterrows():
            shot_num_val = int(shot_row.get('Shot_Number', 0))
            
            # Get direction and handle score extraction
            direction = str(shot_row.get('Direction', '')).strip().upper()
            try:
                score = float(shot_row.get('Score', 0))
                score_valid = True
            except (ValueError, TypeError):
                score = 0
                score_valid = False
                print(f"Warning: Non-numeric score '{shot_row.get('Score', 'N/A')}' for shot {shot_num_val}")
            
            # Calculate position for this shot based on direction first
            direction_angles = {
                'N': 90,    # North = up
                'NE': 45,   # Northeast
                'E': 0,     # East = right
                'SE': 315,  # Southeast
                'S': 270,   # South = down
                'SW': 225,  # Southwest
                'W': 180,   # West = left
                'NW': 135   # Northwest
            }
            
            import math
            import random
            if direction in direction_angles:
                # Use direction-based placement
                angle_deg = direction_angles[direction]
                angle_rad = math.radians(angle_deg)
                
                # Place shot in the correct scoring ring based on score
                if score_valid:
                    # Find the appropriate ring for this score
                    target_ring_idx = None
                    for i, ring_score in enumerate(ring_scores):
                        if score >= ring_score:
                            target_ring_idx = i
                            break
                    
                    if target_ring_idx is None:
                        target_ring_idx = len(ring_radii) - 1
                    
                    # Get the ring radius for this score
                    ring_radius = ring_radii[target_ring_idx]
                    if target_ring_idx > 0:
                        inner_radius = ring_radii[target_ring_idx - 1]
                    else:
                        inner_radius = 0
                        
                    # Place shot randomly within the correct ring
                    random.seed(shot_num_val)
                    final_distance = random.uniform(inner_radius, ring_radius)
                else:
                    # If no valid score, place in outer ring
                    final_distance = random.uniform(ring_radii[-2], ring_radii[-1])
                
                # Calculate coordinates
                shot_x = final_distance * math.cos(angle_rad)
                shot_y = final_distance * math.sin(angle_rad)
                print(f"Shot {shot_num_val} placed by direction '{direction}' at ({shot_x:.2f}, {shot_y:.2f})")
            else:
                # Fallback to score-based placement if no direction
                target_ring_idx = None
                if score_valid:
                    for i, ring_score in enumerate(ring_scores):
                        if score >= ring_score:
                            target_ring_idx = i
                            break
                
                if target_ring_idx is None:
                    target_ring_idx = len(ring_radii) - 1
                
                ring_radius = ring_radii[target_ring_idx]
                if target_ring_idx > 0:
                    inner_radius = ring_radii[target_ring_idx - 1]
                else:
                    inner_radius = 0
                
                random.seed(shot_num_val)
                distance = random.uniform(inner_radius, ring_radius)
                angle = random.uniform(0, 2 * math.pi)
                
                shot_x = distance * math.cos(angle)
                shot_y = distance * math.sin(angle)
                print(f"Shot {shot_num_val} fallback placement at ({shot_x:.2f}, {shot_y:.2f})")
            
            # Store position and data
            shot_data = {
                'x': shot_x,
                'y': shot_y,
                'shot_num': shot_num_val,
                'score': score,
                'score_valid': score_valid,
                'is_current': shot_num_val == int(shot_num)
            }
            shot_positions.append(shot_data)
            
            # Store for click detection
            self.shot_positions_data.append(shot_data)
        
        # Plot all shots with improved visibility and contrast
        # First plot all non-current shots
        for pos_data in shot_positions:
            if not pos_data['is_current']:
                # Other shots - smaller with better contrast
                # Use different colors based on score for better distinction
                score = pos_data['score']
                try:
                    # Try to convert score to numeric for color assignment
                    numeric_score = float(score) if score is not None else 0
                    if numeric_score >= 9:
                        color = '#00FF00'  # Bright green for high scores
                        edge_color = '#006400'
                    elif numeric_score >= 7:
                        color = '#FFA500'  # Orange for medium scores
                        edge_color = '#FF8C00'
                    else:
                        color = '#4169E1'  # Royal blue for lower scores
                        edge_color = '#000080'
                except (ValueError, TypeError):
                    # If score is not numeric, use default color
                    color = '#808080'  # Gray for non-numeric scores
                    edge_color = '#404040'
                
                ax.scatter(pos_data['x'], pos_data['y'], color=color, s=other_shot_size, zorder=4, 
                          edgecolor=edge_color, linewidth=2, alpha=0.8)
                
                # Only show labels for first 15 shots in compact mode to avoid clutter
                if not compact_mode or len([p for p in shot_positions if p['shot_num'] <= pos_data['shot_num']]) <= 15:
                    ax.annotate(f'{pos_data["shot_num"]}', 
                               (pos_data['x'], pos_data['y']), xytext=(pos_data['x'] + 0.3, pos_data['y'] + 0.3),
                               fontsize=max(6, label_fontsize-2), fontweight='bold', color=edge_color,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=edge_color))
        
        # Now plot the current shot on top with enhanced visibility
        for pos_data in shot_positions:
            if pos_data['is_current']:
                # Current shot - larger and bright red with thick border and extra emphasis
                # Add a glowing effect by plotting multiple circles
                ax.scatter(pos_data['x'], pos_data['y'], color='#FF6B6B', s=current_shot_size * 1.5, zorder=7, 
                          edgecolor='none', linewidth=0, alpha=0.3)  # Outer glow
                ax.scatter(pos_data['x'], pos_data['y'], color='#FF0000', s=current_shot_size, zorder=8, 
                          edgecolor='#FFFFFF', linewidth=4, alpha=1.0)  # Main marker with white border
                ax.scatter(pos_data['x'], pos_data['y'], color='#8B0000', s=current_shot_size * 0.6, zorder=9, 
                          edgecolor='none', linewidth=0, alpha=1.0)  # Inner dark core
                
                # Remove the text annotation - keep only the visual marker
                break  # Only one current shot
        
        # Update title to show all shots with compact mode info
        title_text = f"Virtual Target - All Shots (Current: Shot {shot_num})"
        if compact_mode:
            title_text += " (Compact View)"
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25)
        
        # Add summary statistics with filter information
        avg_score = 0
        if 'Score' in all_shots_data.columns:
            try:
                # Try to convert Score column to numeric, handling non-numeric values
                numeric_scores = pd.to_numeric(all_shots_data['Score'], errors='coerce')
                # Calculate mean only on valid numeric values
                valid_scores = numeric_scores.dropna()
                if len(valid_scores) > 0:
                    avg_score = valid_scores.mean()
                else:
                    print("Warning: No valid numeric scores found in Score column")
            except Exception as e:
                print(f"Error processing Score column: {e}")
                avg_score = 0
        
        summary_text = f'Total Shots: {total_shots}\nAverage Score: {avg_score:.1f}'
        
        # Add filter information
        if hasattr(self, 'shot_playback_min_spin') and hasattr(self, 'shot_playback_max_spin'):
            shot_min = self.shot_playback_min_spin.value()
            shot_max = self.shot_playback_max_spin.value()
            if shot_min > 0 or shot_max < 1000:
                summary_text += f'\nShot Range: {shot_min}-{shot_max}'
        
        if hasattr(self, 'shot_playback_score_min_spin') and hasattr(self, 'shot_playback_score_max_spin'):
            score_min = self.shot_playback_score_min_spin.value()
            score_max = self.shot_playback_score_max_spin.value()
            if score_min > 0.0 or score_max < 10.0:
                summary_text += f'\nScore Range: {score_min:.1f}-{score_max:.1f}'
        
        if compact_mode:
            summary_text += f'\nCompact Mode: Enabled'
        
        ax.text(0.02, 0.98, summary_text, 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
               fontsize=10)
        
        # Set up the plot with larger range for better visibility
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(-8.0, 8.0)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # Add color legend for shot scores in bottom-left
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', 
                      markersize=10, label='Current Shot', markeredgecolor='#8B0000', markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00', 
                      markersize=8, label='Score 9-10', markeredgecolor='#006400'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', 
                      markersize=8, label='Score 7-8', markeredgecolor='#FF8C00'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', 
                      markersize=8, label='Score <7', markeredgecolor='#000080')
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc='lower left', title='Shot Colors')
        
                    # Add zoom controls in the purple circle area (bottom-right)
        zoom_controls_text = "🔍 Zoom Controls:\n• Click shots to view video\n• Use buttons below for zoom"
        ax.text(5.5, -6.5, zoom_controls_text, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6E6FA', alpha=0.8, edgecolor='#9370DB'),
               color='#4B0082', fontweight='bold')
        

        
        # Add interactive functionality for shot selection and zoom buttons
        self.setup_virtual_target_interactivity(ax)
        
        # Restore zoom state if it was zoomed (compare with default view limits)
        default_xlim = (-8.0, 8.0)
        default_ylim = (-8.0, 8.0)
        
        # Check if we should restore zoom state
        should_restore_zoom = False
        if current_xlim and current_ylim:
            # Check if the previous view was significantly different from default
            x_diff = abs(current_xlim[0] - default_xlim[0]) + abs(current_xlim[1] - default_xlim[1])
            y_diff = abs(current_ylim[0] - default_ylim[0]) + abs(current_ylim[1] - default_ylim[1])
            should_restore_zoom = (x_diff > 0.1 or y_diff > 0.1)
        
        if should_restore_zoom:
            # Set the zoom limits
            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)
            
            # Update title to indicate zoomed state if it was zoomed before
            if was_zoomed:
                current_title = ax.get_title()
                if "(Zoomed)" not in current_title:
                    ax.set_title(current_title + " (Zoomed)", fontsize=18, fontweight='bold', pad=25)
            
            # Force a redraw to apply the zoom
            self.virtual_target_canvas.draw()
            print(f"Preserved zoom state during shot selection: xlim={current_xlim}, ylim={current_ylim}")
        else:
            # Ensure we're using the default view
            ax.set_xlim(default_xlim)
            ax.set_ylim(default_ylim)
            print("No zoom to preserve - using default view")
        
        # Add axis labels with larger font
        ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=14, fontweight='bold')
        
        # Add target center marker
        ax.scatter(0, 0, color='black', s=50, zorder=6, marker='+', linewidth=3)
        
        self.virtual_target_canvas.figure.tight_layout()
        self.virtual_target_canvas.draw()
        print("Virtual target visualization completed and drawn")
        
        # Reset the counter when done
        self._virtual_target_update_count = 0

    def setup_virtual_target_interactivity(self, ax):
        """Setup interactive functionality for shot selection and zoom controls"""
        # Store default view limits for reset functionality
        self.original_xlim = (-8.0, 8.0)  # Default view limits
        self.original_ylim = (-8.0, 8.0)
        
        # Connect mouse events for shot selection using matplotlib events
        print("Setting up virtual target click handler...")
        print(f"Canvas type: {type(self.virtual_target_canvas)}")
        print(f"Canvas has figure: {hasattr(self.virtual_target_canvas, 'figure')}")
        print(f"Canvas has axes: {len(self.virtual_target_canvas.figure.axes) if hasattr(self.virtual_target_canvas, 'figure') else 'No figure'}")
        
        # Store the axes for later use
        self.virtual_target_ax = ax
        
        # Connect matplotlib button_press_event
        self.virtual_target_canvas.mpl_connect('button_press_event', self.on_virtual_target_shot_click)
        
        # Add zoom buttons to the canvas
        self.add_zoom_buttons_to_canvas()
        
        print("Matplotlib-based click handler connected successfully")

    def update_virtual_target_shot_selection(self, shot_num):
        """Update shot selection visually without redrawing the entire target"""
        if not hasattr(self, 'virtual_target_canvas') or not self.virtual_target_canvas.figure.axes:
            return
        
        ax = self.virtual_target_canvas.figure.axes[0]
        
        # Clear existing shot markers
        for artist in ax.get_children():
            if hasattr(artist, '_shot_marker'):
                artist.remove()
        
        # Redraw all shots with updated selection
        if hasattr(self, 'shot_positions_data') and self.shot_positions_data:
            for shot_data in self.shot_positions_data:
                x, y = shot_data['x'], shot_data['y']
                shot_num_data = shot_data['shot_num']
                score = shot_data['score']
                
                # Determine color based on selection and score
                if shot_num_data == shot_num:
                    # Selected shot - red with black border
                    color = 'red'
                    edgecolor = 'black'
                    markersize = 300  # Increased from 200 to 300
                    zorder = 10
                else:
                    # Non-selected shots - color based on score
                    if score >= 9:
                        color = 'green'
                    elif score >= 7:
                        color = 'orange'
                    else:
                        color = 'blue'
                    edgecolor = 'black'
                    markersize = 200  # Increased from 150 to 200
                    zorder = 5
                
                # Plot the shot
                marker = ax.scatter(x, y, c=color, s=markersize, edgecolors=edgecolor, 
                                  linewidth=2, zorder=zorder, alpha=0.8)
                marker._shot_marker = True  # Mark as shot marker for easy removal
                
                # Add shot number annotation
                ax.annotate(str(shot_num_data), (x, y), xytext=(0, 0), 
                           textcoords='offset points', ha='center', va='center',
                           fontsize=10, fontweight='bold', color='white')
        
        # Update the title to show current shot
        current_title = ax.get_title()
        if "(Current: Shot" in current_title:
            # Replace existing shot number
            title_parts = current_title.split("(Current: Shot")
            if len(title_parts) > 1:
                new_title = title_parts[0] + f"(Current: Shot {shot_num})"
                if "(Zoomed)" in current_title:
                    new_title += " (Zoomed)"
                ax.set_title(new_title, fontsize=18, fontweight='bold', pad=25)
        else:
            # Add shot number to title
            new_title = current_title + f" (Current: Shot {shot_num})"
            ax.set_title(new_title, fontsize=18, fontweight='bold', pad=25)
        
        # Redraw the canvas
        self.virtual_target_canvas.draw()
        print(f"Updated shot selection to Shot {shot_num} without redrawing entire target")

    def on_virtual_target_shot_click(self, event):
        """Handle mouse clicks on shots to select them for video playback"""
        print(f"Click event received: button={event.button}, inaxes={event.inaxes}")
        
        # Check if click is in the virtual target axes
        if not hasattr(self, 'virtual_target_ax') or event.inaxes != self.virtual_target_ax:
            print(f"Click not in target axes: {event.inaxes}")
            return
        
        if event.button == 1:  # Left click only
            click_x, click_y = event.xdata, event.ydata
            
            print(f"Left click detected at ({click_x:.2f}, {click_y:.2f})")
            print(f"Shot positions data length: {len(self.shot_positions_data) if hasattr(self, 'shot_positions_data') else 'No data'}")
            
            # Find the closest shot to the click
            closest_shot = None
            min_distance = float('inf')
            
            if not hasattr(self, 'shot_positions_data') or not self.shot_positions_data:
                print("No shot positions data available!")
                return
            
            for shot_data in self.shot_positions_data:
                shot_x, shot_y = shot_data['x'], shot_data['y']
                distance = ((click_x - shot_x) ** 2 + (click_y - shot_y) ** 2) ** 0.5
                
                print(f"  Shot {shot_data['shot_num']} at ({shot_x:.2f}, {shot_y:.2f}) - distance: {distance:.2f}")
                
                if distance < min_distance and distance < 3.0:  # Within 3 unit radius for easier clicking
                    min_distance = distance
                    closest_shot = shot_data
            
            if closest_shot:
                shot_num = closest_shot['shot_num']
                score = closest_shot['score']
                print(f"Selected Shot {shot_num} (Score: {score}) for video playback")
                
                # Update the current shot
                self.shot_playback_current_shot = shot_num
                
                # Update shot selection visually without redrawing entire target
                self.update_virtual_target_shot_selection(shot_num)
                
                # Highlight the shot in the bar chart for bidirectional highlighting
                self.highlight_shot_in_bar_chart(shot_num)
                
                # Trigger skeleton video playback
                self.trigger_skeleton_video_for_shot(shot_num)
                
                # Trigger shot playback video if available
                if hasattr(self, 'shot_playback_frames_by_shot') and shot_num in self.shot_playback_frames_by_shot:
                    self.draw_shot_playback_current_frame()
                    print(f"Starting shot playback video for Shot {shot_num}")
            else:
                print("No shot found near click position")
    def add_zoom_buttons_to_canvas(self):
        """Add zoom in/out buttons to the virtual target canvas"""
        from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
        
        # Create a widget to hold the zoom buttons
        zoom_widget = QWidget()
        zoom_layout = QVBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create zoom in button
        zoom_in_btn = QPushButton("🔍 Zoom In")
        zoom_in_btn.setMaximumWidth(100)
        zoom_in_btn.clicked.connect(self.zoom_in_virtual_target)
        zoom_layout.addWidget(zoom_in_btn)
        
        # Create zoom out button
        zoom_out_btn = QPushButton("🔍 Zoom Out")
        zoom_out_btn.setMaximumWidth(100)
        zoom_out_btn.clicked.connect(self.zoom_out_virtual_target)
        zoom_layout.addWidget(zoom_out_btn)
        
        # Create reset button
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setMaximumWidth(100)
        reset_btn.clicked.connect(self.reset_virtual_target_zoom)
        zoom_layout.addWidget(reset_btn)
        
        # Position the widget in the bottom-right corner of the canvas
        zoom_widget.setParent(self.virtual_target_canvas)
        zoom_widget.move(self.virtual_target_canvas.width() - 120, self.virtual_target_canvas.height() - 120)
        zoom_widget.show()

    def zoom_in_virtual_target(self):
        """Zoom in on the virtual target"""
        try:
            if not hasattr(self, 'virtual_target_canvas') or self.virtual_target_canvas is None:
                print("Virtual target canvas not available")
                return
                
            if not self.virtual_target_canvas.figure.axes:
                print("No axes found in virtual target canvas")
                return
                
            ax = self.virtual_target_canvas.figure.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Zoom in by 0.5x from center
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            
            x_range = (xlim[1] - xlim[0]) * 0.5
            y_range = (ylim[1] - ylim[0]) * 0.5
            
            new_xlim = (center_x - x_range/2, center_x + x_range/2)
            new_ylim = (center_y - y_range/2, center_y + y_range/2)
            
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            
            # Update title
            current_title = ax.get_title()
            if "(Zoomed)" not in current_title:
                ax.set_title(current_title + " (Zoomed)", fontsize=18, fontweight='bold', pad=25)
            
            self.virtual_target_canvas.draw()
            print("Zoomed in on virtual target")
        except Exception as e:
            print(f"Error zooming in: {e}")

    def zoom_out_virtual_target(self):
        """Zoom out on the virtual target"""
        try:
            if not hasattr(self, 'virtual_target_canvas') or self.virtual_target_canvas is None:
                print("Virtual target canvas not available")
                return
        
            if not self.virtual_target_canvas.figure.axes:
                print("No axes found in virtual target canvas")
                return
                
            ax = self.virtual_target_canvas.figure.axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Zoom out by 2x from center
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            
            x_range = (xlim[1] - xlim[0]) * 2.0
            y_range = (ylim[1] - ylim[0]) * 2.0
            
            new_xlim = (center_x - x_range/2, center_x + x_range/2)
            new_ylim = (center_y - y_range/2, center_y + y_range/2)
            
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            
            # Update title
            current_title = ax.get_title()
            if "(Zoomed)" not in current_title:
                ax.set_title(current_title + " (Zoomed)", fontsize=18, fontweight='bold', pad=25)
            
            self.virtual_target_canvas.draw()
            print("Zoomed out on virtual target")
        except Exception as e:
            print(f"Error zooming out: {e}")

    def reset_virtual_target_zoom(self):
        """Reset zoom to original view"""
        try:
            if not hasattr(self, 'virtual_target_canvas') or self.virtual_target_canvas is None:
                print("Virtual target canvas not available")
                return
        
            if not self.virtual_target_canvas.figure.axes:
                print("No axes found in virtual target canvas")
                return
                
            if not hasattr(self, 'original_xlim') or not hasattr(self, 'original_ylim'):
                print("Original zoom limits not set")
                return
                
            ax = self.virtual_target_canvas.figure.axes[0]
            ax.set_xlim(self.original_xlim)
            ax.set_ylim(self.original_ylim)
            
            # Remove zoom indicator from title
            current_title = ax.get_title()
            if "(Zoomed)" in current_title:
                ax.set_title(current_title.replace(" (Zoomed)", ""), fontsize=18, fontweight='bold', pad=25)
            
            self.virtual_target_canvas.draw()
            print("Reset zoom to original view")
        except Exception as e:
            print(f"Error resetting zoom: {e}")

    def trigger_skeleton_video_for_shot(self, shot_num):
        """Trigger skeleton video playback for the selected shot"""
        print(f"Triggering skeleton video for Shot {shot_num}")
        
        # First, try to use shot playback data if available
        if hasattr(self, 'shot_playback_frames_by_shot') and self.shot_playback_frames_by_shot:
            if shot_num in self.shot_playback_frames_by_shot:
                print(f"Found shot {shot_num} in shot playback data")
                
                # Set the current frames for the skeleton system
                self.shot_playback_current_frames = self.shot_playback_frames_by_shot[shot_num]
                self.shot_playback_current_shot = shot_num
                
                # Update skeleton shot number if available
                if hasattr(self, 'skeleton_shot_number_spin'):
                    self.skeleton_shot_number_spin.setValue(shot_num)
                    print(f"Updated skeleton shot number to {shot_num}")
                
                # Draw the first frame
                self.draw_shot_playback_current_frame()
                
                # Start playback if not already playing
                if hasattr(self, 'shot_playback_timer') and not self.shot_playback_timer.isActive():
                    self.toggle_shot_playback(True)
                    print(f"Started shot playback video for Shot {shot_num}")
                else:
                    print(f"Shot playback video ready for Shot {shot_num}")
            return
        
        # Fallback to skeleton data if available
        if hasattr(self, 'skeleton_data_by_shot') and self.skeleton_data_by_shot:
            # Find the skeleton data for this shot - try different naming conventions
            possible_keys = [
                f"Shot_{shot_num}",
                f"shot_{shot_num}",
                str(shot_num),
                f"Shot{shot_num}",
                f"shot{shot_num}",
                f"shot_{shot_num:02d}",  # Try with zero padding
                f"Shot_{shot_num:02d}",
                f"shot{shot_num:02d}",
                f"Shot{shot_num:02d}"
            ]
            
            found_key = None
            for key in possible_keys:
                if key in self.skeleton_data_by_shot:
                    found_key = key
                    break
            
            if found_key:
                print(f"Found skeleton data for {found_key}")
                
                # Update skeleton shot number
                if hasattr(self, 'skeleton_shot_number_spin'):
                    self.skeleton_shot_number_spin.setValue(shot_num)
                    print(f"Updated skeleton shot number to {shot_num}")
                
                # Prepare skeleton data for playback
                self.prepare_skeleton_data_for_playback()
                
                # Draw the first frame
                self.draw_current_skeleton_frame()
                
                # Start playback if not already playing
                if hasattr(self, 'skeleton_playback_timer') and not self.skeleton_playback_timer.isActive():
                    self.toggle_skeleton_playback(True)
                    print(f"Started skeleton video playback for Shot {shot_num}")
                else:
                    print(f"Skeleton video ready for Shot {shot_num}")
            else:
                print(f"No skeleton data found for Shot {shot_num}")
                # Try to find the shot in available data
                available_shots = list(self.skeleton_data_by_shot.keys())
                print(f"Available skeleton shots: {available_shots}")
                
                # Try to find a shot with similar number
                for key in available_shots:
                    if str(shot_num) in key:
                        print(f"Found similar shot: {key}")
                        # Update to this shot
                        if hasattr(self, 'skeleton_shot_number_spin'):
                            # Extract shot number from key
                            import re
                            match = re.search(r'\d+', key)
                            if match:
                                found_shot_num = int(match.group())
                                self.skeleton_shot_number_spin.setValue(found_shot_num)
                                print(f"Updated to similar shot: {found_shot_num}")
                                self.prepare_skeleton_data_for_playback()
                                self.draw_current_skeleton_frame()
                                break
        else:
            print("No skeleton data or shot playback data available")
        
        # Note: All shots are now displayed by default in the shot playback tab
        # The current shot is highlighted in red, while other shots are shown in blue
        # Compact mode is automatically enabled for 30+ shots to improve visibility

    def update_shot_playback_virtual_target_filters(self):
        """Update virtual target visualization when filters change"""
        # Use debounced update to prevent rapid successive calls
        if hasattr(self, 'virtual_target_debounce_timer'):
            self.virtual_target_debounce_timer.stop()
            self.virtual_target_debounce_timer.start(300)  # 300ms delay
        
        # Also trigger the debounced update method
        if hasattr(self, '_debounced_virtual_target_update'):
            self._debounced_virtual_target_update()

    def _debounced_virtual_target_update(self):
        """Debounced virtual target update to prevent infinite loops"""
        if hasattr(self, 'shot_playback_current_shot') and self.shot_playback_current_shot is not None:
            print(f"Debounced virtual target update for shot {self.shot_playback_current_shot}")
            # Add a longer delay to ensure the previous update is complete and prevent rapid successive calls
            QTimer.singleShot(500, lambda: self.update_shot_playback_virtual_target(self.shot_playback_current_shot))

    def clear_shot_playback_filters(self):
        """Clear all filters for shot playback virtual target"""
        if hasattr(self, 'shot_playback_min_spin'):
            self.shot_playback_min_spin.setValue(0)
        if hasattr(self, 'shot_playback_max_spin'):
            self.shot_playback_max_spin.setValue(1000)
        if hasattr(self, 'shot_playback_score_min_spin'):
            self.shot_playback_score_min_spin.setValue(0.0)
        if hasattr(self, 'shot_playback_score_max_spin'):
            self.shot_playback_score_max_spin.setValue(10.0)
        
        # Clear highlights when filters are cleared
        self.clear_all_highlights()
        
        # Update the virtual target visualization
        if hasattr(self, 'shot_playback_current_shot') and self.shot_playback_current_shot is not None:
            self.update_shot_playback_virtual_target(self.shot_playback_current_shot)

    def clear_skeleton_filters(self):
        """Clear all filters for skeleton virtual target"""
        if hasattr(self, 'skeleton_min_spin'):
            self.skeleton_min_spin.setValue(0)
        if hasattr(self, 'skeleton_max_spin'):
            self.skeleton_max_spin.setValue(1000)
        if hasattr(self, 'skeleton_score_min_spin'):
            self.skeleton_score_min_spin.setValue(0.0)
        if hasattr(self, 'skeleton_score_max_spin'):
            self.skeleton_score_max_spin.setValue(10.0)
        
        # Clear highlights when filters are cleared
        self.clear_all_highlights()
        
        # Update the skeleton virtual target visualization
        self.update_skeleton_virtual_target()
        if hasattr(self, 'shot_playback_score_max_spin'):
            self.shot_playback_score_max_spin.setValue(10.0)
        self.update_shot_playback_virtual_target_filters()

    def highlight_shot_in_bar_chart(self, shot_num):
        """Highlight the specified shot in the shot timing bar chart"""
        if not hasattr(self, 'shot_overview_canvas') or self.shot_overview_canvas is None:
            return
        
        try:
            # Clear previous highlights
            self.clear_bar_chart_highlights()
            
            # Get the axes
            ax = self.shot_overview_canvas.figure.axes[0]
            
            # Find the shot in the bar chart data
            if hasattr(self, 'shot_playback_csv_df') and self.shot_playback_csv_df is not None:
                shot_numbers = sorted(self.shot_playback_csv_df["Shot_Number"].unique())
                if shot_num in shot_numbers:
                    # Get the y-position (shot number position in horizontal bar chart)
                    y_pos = shot_numbers.index(shot_num) + 1  # +1 because shot numbers start from 1
                    
                    # Add a highlight rectangle around the selected shot
                    phases = ["Preparing_Time(s)", "Aiming_Time(s)", "After_Shot_Time(s)"]
                    total_width = 0
                    
                    # Calculate total width for this shot
                    row = self.shot_playback_csv_df[self.shot_playback_csv_df["Shot_Number"] == shot_num]
                    if not row.empty:
                        for phase in phases:
                            value = float(row.iloc[0].get(phase, 0))
                            total_width += value
                    
                    # Create highlight rectangle
                    highlight_rect = patches.Rectangle(
                        (0, y_pos - 0.4),  # x, y position
                        total_width + 0.2,  # width (add some padding)
                        0.8,  # height
                        linewidth=3,
                        edgecolor='#FF0000',  # Red color
                        facecolor='none',
                        alpha=0.8,
                        zorder=10
                    )
                    ax.add_patch(highlight_rect)
                    
                    # Add shot number label with highlight
                    ax.text(-0.5, y_pos, f"→ Shot {shot_num}", 
                           va='center', ha='right', fontsize=10, fontweight='bold',
                           color='#000000', bbox=dict(boxstyle="round,pad=0.3", 
                                                    facecolor='#FF0000', alpha=0.2))
                    
                    # Redraw the canvas
                    self.shot_overview_canvas.draw()
                    print(f"Highlighted Shot {shot_num} in bar chart")
        except Exception as e:
            print(f"Error highlighting shot in bar chart: {e}")

    def clear_bar_chart_highlights(self):
        """Clear all highlights from the bar chart"""
        if not hasattr(self, 'shot_overview_canvas') or self.shot_overview_canvas is None:
            return
        
        try:
            ax = self.shot_overview_canvas.figure.axes[0]
            
            # Remove all patches (highlights)
            for patch in ax.patches[:]:
                if hasattr(patch, 'get_edgecolor') and patch.get_edgecolor() == '#FFD700':
                    patch.remove()
            
            # Remove highlight text labels
            for text in ax.texts[:]:
                if hasattr(text, 'get_text') and '→ Shot' in text.get_text():
                    text.remove()
            
            # Redraw the canvas
            self.shot_overview_canvas.draw()
        except Exception as e:
            print(f"Error clearing bar chart highlights: {e}")

    def highlight_shot_in_virtual_target(self, shot_num):
        """Highlight the specified shot in the virtual target"""
        if not hasattr(self, 'virtual_target_ax') or self.virtual_target_ax is None:
            return
        
        try:
            # Clear previous highlights
            self.clear_virtual_target_highlights()
            
            # Find the shot in the shot positions data
            if hasattr(self, 'shot_positions_data') and self.shot_positions_data:
                for shot_data in self.shot_positions_data:
                    if shot_data['shot_num'] == shot_num:
                        # Add a highlight circle around the selected shot
                        highlight_circle = patches.Circle(
                            (shot_data['x'], shot_data['y']),
                            radius=1.5,  # Larger radius for highlight
                            linewidth=3,
                            edgecolor='#FFD700',  # Gold color
                            facecolor='none',
                            alpha=0.8,
                            zorder=10
                        )
                        self.virtual_target_ax.add_patch(highlight_circle)
                        
                        # Add shot number label with highlight
                        self.virtual_target_ax.text(
                            shot_data['x'] + 1.5, shot_data['y'] + 1.5,
                            f"Shot {shot_num}",
                            fontsize=12, fontweight='bold',
                            color='#FFD700',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor='#FFD700', alpha=0.2),
                            ha='left', va='bottom'
                        )
                        
                        # Redraw the virtual target
                        if hasattr(self, 'virtual_target_canvas'):
                            self.virtual_target_canvas.draw()
                        print(f"Highlighted Shot {shot_num} in virtual target")
                        break
        except Exception as e:
            print(f"Error highlighting shot in virtual target: {e}")

    def clear_virtual_target_highlights(self):
        """Clear all highlights from the virtual target"""
        if not hasattr(self, 'virtual_target_ax') or self.virtual_target_ax is None:
            return
        
        try:
            # Remove highlight circles
            for patch in self.virtual_target_ax.patches[:]:
                if hasattr(patch, 'get_edgecolor') and patch.get_edgecolor() == '#FFD700':
                    patch.remove()
            
            # Remove highlight text labels
            for text in self.virtual_target_ax.texts[:]:
                if hasattr(text, 'get_text') and 'Shot ' in text.get_text():
                    if hasattr(text, 'get_bbox_patch') and text.get_bbox_patch():
                        bbox_color = text.get_bbox_patch().get_facecolor()
                        if bbox_color == '#FFD700':
                            text.remove()
            
            # Redraw the virtual target
            if hasattr(self, 'virtual_target_canvas'):
                self.virtual_target_canvas.draw()
        except Exception as e:
            print(f"Error clearing virtual target highlights: {e}")

    def clear_all_highlights(self):
        """Clear highlights from both virtual target and bar chart"""
        self.clear_virtual_target_highlights()
        self.clear_bar_chart_highlights()
        print("Cleared all highlights from both components")

    def highlight_shot_in_both_components(self, shot_num):
        """Highlight a shot in both the virtual target and bar chart simultaneously"""
        # Clear previous highlights from both components
        self.clear_all_highlights()
        
        # Highlight in virtual target
        self.highlight_shot_in_virtual_target(shot_num)
        
        # Highlight in bar chart
        self.highlight_shot_in_bar_chart(shot_num)
        
        print(f"Highlighted Shot {shot_num} in both components")

    def on_show_all_shots_toggled(self):
        """Handle the show all shots checkbox toggle"""
        # Use debounced update to prevent rapid successive calls
        if hasattr(self, 'virtual_target_debounce_timer'):
            self.virtual_target_debounce_timer.stop()
            self.virtual_target_debounce_timer.start(300)  # 300ms delay

class PhaseMarkerBar(QWidget):
    phase_clicked = pyqtSignal(int)  # Emits the start_frame of the clicked phase

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase_data = []
        self.total_frames = 1
        self.phase_colors = UNIFIED_PHASE_COLORS
        self.setMinimumHeight(24)
        self.setMouseTracking(True)

    def set_phases(self, phase_data, total_frames):
        self.phase_data = phase_data or []
        self.total_frames = max(1, total_frames)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        font = QFont("Arial", 8, QFont.Bold)
        painter.setFont(font)
        for phase in self.phase_data:
            start = phase.get('start_frame', 0)
            end = phase.get('end_frame', 0)
            name = phase.get('phase_name', 'unknown')
            color = QColor(self.phase_colors.get(str(name).lower(), '#CCCCCC'))
            x1 = int((start / self.total_frames) * w)
            x2 = int((end / self.total_frames) * w)
            rect_w = max(1, x2 - x1)
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(color)
            painter.drawRect(x1, 0, rect_w, h)
            painter.setPen(Qt.black)
            painter.drawText(x1, 0, rect_w, h, Qt.AlignCenter, str(name).upper())
        painter.end()

    def mousePressEvent(self, event):
        if not self.phase_data or self.total_frames <= 0:
            return
        x = event.x()
        w = self.width()
        for phase in self.phase_data:
            start = phase.get('start_frame', 0)
            end = phase.get('end_frame', 0)
            x1 = int((start / self.total_frames) * w)
            x2 = int((end / self.total_frames) * w)
            if x1 <= x < x2:
                self.phase_clicked.emit(start)
                break

# --- Add after PhaseMarkerBar ---
class DualPhaseBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)  # More compact
        self.primary_bar = PhaseMarkerBar()
        self.compare_bar = PhaseMarkerBar()
        self.primary_bar.setMinimumHeight(18)
        self.primary_bar.setMaximumHeight(24)
        self.compare_bar.setMinimumHeight(18)
        self.compare_bar.setMaximumHeight(24)
        self.layout.addWidget(self.primary_bar)
        self.layout.addWidget(self.compare_bar)
        self.compare_bar.setVisible(False)
        self.setMinimumHeight(36)
        self.setMaximumHeight(48)
        


    def set_phases(self, primary_phases, primary_total_frames, compare_phases=None, compare_total_frames=None):
        self.primary_bar.set_phases(primary_phases, primary_total_frames)
        if compare_phases is not None and compare_total_frames is not None:
            self.compare_bar.set_phases(compare_phases, compare_total_frames)
            self.compare_bar.setVisible(True)
        else:
            self.compare_bar.setVisible(False)
    


    def set_phase_clicked_callback(self, callback):
        self.primary_bar.phase_clicked.connect(callback)
        self.compare_bar.phase_clicked.connect(callback)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = ShotTimingGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()