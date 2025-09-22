# AE AnalyzerGUI: Integrated GUI with Full Pose and Shot Logic
# ---
# This is the starting structure. We'll integrate all functional pieces step-by-step.

import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import threading
import pyaudio
import audioop
import os
import json
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
                             QTabWidget, QGroupBox, QGridLayout, QListWidget,
                             QMessageBox, QFrame, QLineEdit, QDoubleSpinBox,
                             QSpinBox, QDialog, QComboBox, QTextEdit, QFileDialog,
                             QListWidgetItem, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class VideoControls(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                min-width: 40px;
                padding: 5px;
            }
            QLabel {
                color: white;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_pause_btn = QPushButton("⏸")
        self.stop_btn = QPushButton("⏹")
        self.backward_btn = QPushButton("⏪")
        self.forward_btn = QPushButton("⏩")
        
        for btn in [self.backward_btn, self.play_pause_btn, self.stop_btn, self.forward_btn]:
            controls_layout.addWidget(btn)
            btn.setFixedWidth(40)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(25, 200)  # 0.25x to 2x
        self.speed_slider.setValue(100)  # 1x default
        self.speed_value = QLabel("1.0x")
        
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        self.time_label = QLabel("0:00 / 0:00")
        self.progress_slider = QSlider(Qt.Horizontal)
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.time_label)
        
        # Add all layouts
        layout.addLayout(controls_layout)
        layout.addLayout(speed_layout)
        layout.addLayout(progress_layout)

class ThresholdInput(QWidget):
    def __init__(self, label, min_val, max_val, default_val, decimals=6, step=0.000001, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        self.label = QLabel(label)
        layout.addWidget(self.label)
        
        # Determine if this is for movement threshold (small decimals) or sound threshold (larger integers)
        self.is_movement_threshold = decimals > 0 and max_val < 1.0
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        if self.is_movement_threshold:
            # For movement threshold (small decimals)
            self.slider.setMinimum(0)
            self.slider.setMaximum(1000)  # This will map to 0.0000 to 0.0010
            self.slider.setValue(int(default_val * 1000000))  # Scale default value for slider
        else:
            # For sound threshold (larger integers)
            self.slider.setMinimum(min_val)
            self.slider.setMaximum(max_val)
            self.slider.setValue(default_val)
        layout.addWidget(self.slider)
        
        # Input field
        if decimals > 0:
            self.input = QDoubleSpinBox()
            self.input.setDecimals(decimals)
            self.input.setSingleStep(step)
            if self.is_movement_threshold:
                self.input.setMinimum(0.0000)
                self.input.setMaximum(0.0010)
            else:
                self.input.setMinimum(min_val)
                self.input.setMaximum(max_val)
            self.input.setValue(default_val)
        else:
            self.input = QSpinBox()
            self.input.setSingleStep(1)
            self.input.setMinimum(min_val)
            self.input.setMaximum(max_val)
            self.input.setValue(default_val)
        
        self.input.setFixedWidth(100)
        layout.addWidget(self.input)
        
        # Store parameters
        self.decimals = decimals
        
        # Connect signals
        self.slider.valueChanged.connect(self._slider_changed)
        self.input.valueChanged.connect(self._input_changed)
        
        # Style
        self.setStyleSheet("""
            QLabel { color: white; }
            QDoubleSpinBox, QSpinBox {
                background-color: #444444;
                color: #ff0000;
                font-weight: bold;
                border: 1px solid #555555;
                padding: 2px;
            }
        """)

    def _slider_changed(self, value):
        if self.is_movement_threshold:
            self.input.setValue(float(value) / 1000000.0)  # Scale slider value back down for movement threshold
        else:
            self.input.setValue(value)  # No scaling needed for sound threshold
        
    def _input_changed(self, value):
        if self.is_movement_threshold:
            self.slider.setValue(int(value * 1000000))  # Scale input value up for movement threshold
        else:
            self.slider.setValue(int(value))  # No scaling needed for sound threshold

    def value(self):
        return self.input.value()

    def on_threshold_change(self, val):
        """Handle movement threshold slider change"""
        value = val / 1000000.0  # Scale from slider value to actual threshold
        self.movement_threshold = value
        # Don't reset any timers or states when threshold changes
        self.log_message(f"Movement threshold changed to: {value:.6f}")

    def on_threshold_input_change(self, value):
        """Handle movement threshold direct input change"""
        self.movement_threshold = value
        # Don't reset any timers or states when threshold changes
        self.log_message(f"Movement threshold changed to: {value:.6f}")

class PlayerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Player Management")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Player Selection
        player_group = QGroupBox("Select Player")
        player_layout = QVBoxLayout(player_group)
        
        self.player_combo = QComboBox()
        self.player_combo.setEditable(True)
        self.player_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
        player_layout.addWidget(self.player_combo)
        
        # Session Management
        session_group = QGroupBox("Session Management")
        session_layout = QVBoxLayout(session_group)
        
        session_label = QLabel("Session Number:")
        self.session_spin = QSpinBox()
        self.session_spin.setRange(1, 9999)
        session_layout.addWidget(session_label)
        session_layout.addWidget(self.session_spin)
        
        # Notes
        notes_group = QGroupBox("Session Notes")
        notes_layout = QVBoxLayout(notes_group)
        self.notes_text = QTextEdit()
        notes_layout.addWidget(self.notes_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Start Session")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all widgets to main layout
        layout.addWidget(player_group)
        layout.addWidget(session_group)
        layout.addWidget(notes_group)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        # Style
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QLabel {
                color: white;
            }
            QComboBox, QSpinBox {
                background-color: #444444;
                color: white;
                border: 1px solid #555555;
                padding: 5px;
                min-height: 25px;
            }
            QTextEdit {
                background-color: #444444;
                color: white;
                border: 1px solid #555555;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
    
    def get_session_info(self):
        return {
            'player': self.player_combo.currentText(),
            'session': self.session_spin.value(),
            'notes': self.notes_text.toPlainText()
        }

class ArcheryAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AEYE Shooting Pro")
        self.setMinimumSize(1200, 800)

        self.all_shot_details = []  # persist shot details across recordings
        # Test mode flag
        self.test_mode = False
        self.test_shot_counter = 0
        
        # Camera state
        self.camera_enabled = True
        
        # Player and Session Management
        self.current_player = None
        self.current_session = None
        self.session_notes = ""
        
        # Show directory selection dialog first
        self.base_folder = self.select_data_directory()
        if not self.base_folder:
            sys.exit()
            
        self.session_data_file = None  # Initialize the file path variable
        
        # Show player dialog before initializing the rest
        if not self.show_player_dialog():
            sys.exit()

        # Initialize session data file with headers if it doesn't exist
        if not os.path.exists(self.session_data_file):
            self.create_session_data_file()
            
        # Initialize shot counters
        self.combo_counter = 0  # Start at 0, will be incremented to 1 on first shot
        self.last_saved_shot = 0
        self.ready_for_next = True
        self.up_time_start = None
        self.up_time_seconds = 0
        self.display_up_time = 0
        
        # Initialize arm stages
        self.left_stage = None
        self.right_stage = None

        # Set the stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #333333;
            }
            QTabWidget::tab-bar {
                left: 5px;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: white;
                padding: 8px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444444;
                height: 8px;
                background: #2b2b2b;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #4a90e2;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

        # Core Data Variables
        self.baseline_skeleton = None
        self.baseline_anchor = None
        self.baseline_angles = {}
        self.current_status = "free"
        self.last_status = "free"
        self.last_movement_time = None
        self.movement_threshold = 0.0005
        self.stable_time_threshold = 3.0
        self.last_landmarks = None
        self.stability_progress = 0
        self.phase_start_time = time.time()
        self.preparing_duration = 0
        self.aiming_duration = 0
        self.after_shot_duration = 0
        self.is_in_shooting_position = False

        self.data_lock = threading.Lock()
        self.shot_events = []
        self.Shot_events = []

        # Pose & Drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Video Capture Setup
        self.cap = cv2.VideoCapture(0)

        # Add video recording variables
        self.video_writer = None
        self.recording = False
        

        # Add video state variables
        self.is_playing_video = False
        self.timer_enabled = True
        self.video_cap = None
        self.video_paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.playback_speed = 1.0

        # Skeleton recording variables
        self.skeleton_recording = False
        self.skeleton_frames = []
        self.skeleton_metadata = {}
        self.skeleton_start_time = None
        self.skeleton_frame_count = 0
        self.skeleton_fps = 15  # Record at 15fps to save space
        self.skeleton_last_frame_time = 0

        # UI Setup
        self.init_ui()

        # QTimer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Start Audio Detection Thread
        threading.Thread(target=self.detect_shot_audio, daemon=True).start()

        # Initialize session data
        self.save_session_data(0, 0, "free", 0, 0, 0, "No")

    def select_data_directory(self):
        """Show a dialog to select the data directory"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setWindowTitle("Select Players Data Directory")
        
        # Set default directory to user's home directory
        default_dir = os.path.expanduser("~")
        dialog.setDirectory(default_dir)
        
        if dialog.exec_():
            selected_dir = dialog.selectedFiles()[0]
            # Create players_data subdirectory if it doesn't exist
            data_dir = os.path.join(selected_dir, "Players_data")
            if not os.path.exists(data_dir):
                try:
                    os.makedirs(data_dir)
                    QMessageBox.information(self, "Directory Created", 
                        f"Created Players data directory at:\n{data_dir}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", 
                        f"Failed to create directory:\n{str(e)}")
                    return None
            return data_dir
        return None

    def show_player_dialog(self):
        dialog = PlayerDialog(self)
        
        # Load existing players
        players_path = os.path.join(self.base_folder, "players.txt")
        if os.path.exists(players_path):
            with open(players_path, 'r', encoding='utf-8') as f:
                players = [line.strip() for line in f.readlines()]
                dialog.player_combo.addItems(players)
        
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_session_info()
            self.current_player = info['player']
            self.current_session = info['session']
            self.session_notes = info['notes']
            
            # Create directory structure
            self.setup_session_directories()
            
            # Save session info
            self.save_session_info()
            
            # Save player name if new
            self.save_player_name()
            
            # Update window title
            self.setWindowTitle(f"AEYE SHOOTING Pro - {self.current_player} - Session {self.current_session}")
            
            return True
        return False

    def create_session_data_file(self):
        """Create the session data file with headers"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.session_data_file), exist_ok=True)
            
            headers = [
                'Player', 'Session', 'Shot_Number', 'Timestamp', 
                'Up_Time(s)', 'Preparing_Time(s)', 
                'Aiming_Time(s)', 'After_Shot_Time(s)', 
                'Complete_Shot'
            ]
            
            # Use Windows line endings
            with open(self.session_data_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"Created session data file at: {self.session_data_file}")
        except Exception as e:
            print(f"Error creating session data file: {e}")
            QMessageBox.critical(self, "Error", 
                f"Failed to create session data file:\n{str(e)}")

    def setup_session_directories(self):
        """Create the directory structure for the current session"""
        try:
            # Create base directory if it doesn't exist
            if not os.path.exists(self.base_folder):
                os.makedirs(self.base_folder)
                print(f"Created base folder: {self.base_folder}")

            # Create player directory using normalized path
            self.player_folder = os.path.normpath(os.path.join(self.base_folder, self.current_player))
            if not os.path.exists(self.player_folder):
                os.makedirs(self.player_folder)
                print(f"Created player folder: {self.player_folder}")

            # Create session directory with formatted number
            session_name = f"session_{self.current_session:03d}"  # Format: session_001
            self.session_folder = os.path.normpath(os.path.join(self.player_folder, session_name))
            if not os.path.exists(self.session_folder):
                os.makedirs(self.session_folder)
                print(f"Created session folder: {self.session_folder}")

            # Create subdirectories
            self.shots_folder = os.path.normpath(os.path.join(self.session_folder, "videos"))
            if not os.path.exists(self.shots_folder):
                os.makedirs(self.shots_folder)
                print(f"Created videos folder: {self.shots_folder}")

            # Set the session data file path using normalized path with player and session in filename
            self.session_data_file = os.path.normpath(os.path.join(
                self.session_folder, 
                f"{self.current_player}_session_{self.current_session:03d}_shots_data.csv"
            ))
            print(f"Session data file will be created at: {self.session_data_file}")

            return True

        except Exception as e:
            print(f"Error creating directory structure: {e}")
            QMessageBox.critical(self, "Error", 
                f"Failed to create session directories:\n{str(e)}")
            return False

    def save_player_name(self):
        players_path = os.path.join(self.base_folder, "players.txt")
        players = set()
        if os.path.exists(players_path):
            with open(players_path, 'r', encoding='utf-8') as f:
                players = set(line.strip() for line in f.readlines())
        
        players.add(self.current_player)
        with open(players_path, 'w', encoding='utf-8') as f:
            for player in sorted(players):
                f.write(f"{player}\n")
    
    def save_session_info(self):
        """Save session information to a text file"""
        try:
            info_file = os.path.join(self.session_folder, "session_info.txt")
            with open(info_file, 'w') as f:
                f.write(f"Player: {self.current_player}\n")
                f.write(f"Session: {self.current_session:03d}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Notes:\n{self.session_notes}\n")
            print(f"Session info saved to {info_file}")
        except Exception as e:
            print(f"Error saving session info: {e}")
            QMessageBox.warning(self, "Warning", 
                f"Failed to save session info:\n{str(e)}")

    def init_ui(self):
        """Initialize the user interface with improved organization"""
        # Main layout structure
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ========== LEFT PANEL: TABBED INTERFACE ==========
        left_panel = QTabWidget()
        left_panel.setFixedWidth(250)
        left_panel.setTabPosition(QTabWidget.North)
        left_panel.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #444444;
                border-radius: 4px;
                background-color: #333333;
                padding: 5px;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: white;
                font-weight: bold;
            }
        """)

        # ----- SESSION TAB -----
        session_tab = QWidget()
        session_layout = QVBoxLayout(session_tab)
        session_layout.setSpacing(10)
        
        # Player info display
        player_group = QGroupBox("Player & Session")
        player_layout = QGridLayout(player_group)
        player_layout.addWidget(QLabel("Player:"), 0, 0)
        player_name = QLabel(self.current_player)
        player_name.setStyleSheet("font-weight: bold; color: #4a90e2;")
        player_layout.addWidget(player_name, 0, 1)
        player_layout.addWidget(QLabel("Session:"), 1, 0)
        session_num = QLabel(f"{self.current_session}")
        session_num.setStyleSheet("font-weight: bold; color: #4a90e2;")
        player_layout.addWidget(session_num, 1, 1)
        session_layout.addWidget(player_group)
        
        # Test mode group
        test_mode_group = QGroupBox("Test Mode")
        test_mode_layout = QVBoxLayout(test_mode_group)
        
        self.test_mode_btn = QPushButton("Enable Test Mode")
        self.test_mode_btn.setCheckable(True)
        self.test_mode_btn.clicked.connect(self.toggle_test_mode)
        self.test_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked:hover {
                background-color: #218838;
            }
        """)
        
        self.test_mode_label = QLabel("Test mode is OFF")
        self.test_mode_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        self.test_mode_label.setAlignment(Qt.AlignCenter)
        
        test_mode_layout.addWidget(self.test_mode_btn)
        test_mode_layout.addWidget(self.test_mode_label)
        session_layout.addWidget(test_mode_group)
        
        # Camera control group
        camera_group = QGroupBox("Camera Control")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_btn = QPushButton("Disable Camera")
        self.camera_btn.setCheckable(True)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked:hover {
                background-color: #c82333;
            }
        """)
        
        self.camera_label = QLabel("Camera is ON")
        self.camera_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        camera_layout.addWidget(self.camera_btn)
        camera_layout.addWidget(self.camera_label)
        session_layout.addWidget(camera_group)
        
        # Dashboard button
        self.dashboard_btn = QPushButton("Open Analysis Dashboard")
        self.dashboard_btn.clicked.connect(self.launch_dashboard)
        self.dashboard_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
                margin-top: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        session_layout.addWidget(self.dashboard_btn)
        
        session_layout.addStretch()
        left_panel.addTab(session_tab, "Session")

        # ----- PLAYBACK TAB -----
        playback_tab = QWidget()
        playback_layout = QVBoxLayout(playback_tab)
        playback_layout.setSpacing(10)
        
        # Video list group
        video_group = QGroupBox("Recorded Shots")
        video_layout = QVBoxLayout(video_group)
        
        # Buttons for video management
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_video_list)
        delete_btn = QPushButton("🗑 Delete")
        delete_btn.clicked.connect(self.delete_selected_video)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(delete_btn)
        video_layout.addLayout(btn_layout)
        
        # Add video list
        self.video_list = QListWidget()
        self.video_list.itemDoubleClicked.connect(self.play_selected_video)
        self.video_list.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #4a90e2;
                color: white;
            }
        """)
        video_layout.addWidget(self.video_list)
        
        # Return to live button
        return_live_btn = QPushButton("📹 Return to Live")
        return_live_btn.clicked.connect(self.return_to_live)
        return_live_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        video_layout.addWidget(return_live_btn)
        
        playback_layout.addWidget(video_group)
        
        # Shot Information Group
        info_group = QGroupBox("Shot Information")
        info_layout = QGridLayout(info_group)
        
        self.shot_info_labels = {}
        info_fields = [
            ("shot_number", "SHOT #"),
            ("up_time", "UP TIME"),
            ("shot_count", "SHOT COUNT"),
            ("status", "STATUS"),
            ("prep_time", "PREP TIME"),
            ("aim_time", "AIM TIME"),
            ("after_time", "AFTER TIME"),
            ("position", "POSITION"),
            ("stability", "STABILITY")
        ]
        
        for i, (key, text) in enumerate(info_fields):
            label = QLabel(text)
            value = QLabel("-")
            value.setStyleSheet("color: #4a90e2; font-weight: bold;")
            info_layout.addWidget(label, i, 0)
            info_layout.addWidget(value, i, 1)
            self.shot_info_labels[key] = value
            
        playback_layout.addWidget(info_group)
        
        # Video controls
        self.video_controls = VideoControls()
        self.video_controls.play_pause_btn.clicked.connect(self.toggle_pause)
        self.video_controls.stop_btn.clicked.connect(self.stop_playback)
        self.video_controls.forward_btn.clicked.connect(lambda: self.seek_relative(30))
        self.video_controls.backward_btn.clicked.connect(lambda: self.seek_relative(-30))
        self.video_controls.speed_slider.valueChanged.connect(self.update_speed)
        self.video_controls.progress_slider.sliderMoved.connect(self.seek_to_position)
        self.video_controls.setVisible(False)
        playback_layout.addWidget(self.video_controls)
        
        playback_layout.addStretch()
        left_panel.addTab(playback_tab, "Playback")

        # ----- ANALYSIS TAB -----
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setSpacing(10)
        
        # CoG Analysis Group
        cog_analysis_group = QGroupBox("Center of Gravity Analysis")
        cog_analysis_layout = QVBoxLayout(cog_analysis_group)

        self.track_cog_btn = QPushButton("Track Balance")
        self.track_cog_btn.setCheckable(True)
        self.track_cog_btn.clicked.connect(self.toggle_cog_tracking)
        self.track_cog_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
        """)
        cog_analysis_layout.addWidget(self.track_cog_btn)

        cog_buttons_layout = QHBoxLayout()
        
        self.reset_cog_btn = QPushButton("Reset History")
        self.reset_cog_btn.clicked.connect(self.reset_cog_history)
        cog_buttons_layout.addWidget(self.reset_cog_btn)

        self.calculate_cog_btn = QPushButton("Calculate CoG")
        self.calculate_cog_btn.clicked.connect(self.calculate_cog)
        cog_buttons_layout.addWidget(self.calculate_cog_btn)
        
        self.save_cog_data_btn = QPushButton("Save Analysis")
        self.save_cog_data_btn.clicked.connect(self.save_cog_data)
        cog_buttons_layout.addWidget(self.save_cog_data_btn)
        
        cog_analysis_layout.addLayout(cog_buttons_layout)
        analysis_layout.addWidget(cog_analysis_group)
        
        # Skeleton Recording Group
        skeleton_group = QGroupBox("Skeleton Recording")
        skeleton_layout = QVBoxLayout(skeleton_group)

        # Record buttons layout
        recording_btn_layout = QHBoxLayout()
        self.start_skeleton_btn = QPushButton("▶️ Start")
        self.start_skeleton_btn.clicked.connect(self.start_skeleton_recording)
        self.stop_skeleton_btn = QPushButton("⏹️ Stop")
        self.stop_skeleton_btn.clicked.connect(self.stop_skeleton_recording)
        self.stop_skeleton_btn.setEnabled(False)
        recording_btn_layout.addWidget(self.start_skeleton_btn)
        recording_btn_layout.addWidget(self.stop_skeleton_btn)
        skeleton_layout.addLayout(recording_btn_layout)

        # Playback button
        self.view_skeleton_btn = QPushButton("🎬 View Recordings")
        self.view_skeleton_btn.clicked.connect(self.view_skeleton_recordings)
        skeleton_layout.addWidget(self.view_skeleton_btn)

        # Debug button to reload CSV data
        self.reload_csv_btn = QPushButton("🔄 Reload CSV Data")
        self.reload_csv_btn.clicked.connect(self.reload_csv_data)
        self.reload_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: black;
                padding: 5px;
                border: none;
                border-radius: 3px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        skeleton_layout.addWidget(self.reload_csv_btn)

        # Recording status
        self.skeleton_status_label = QLabel("Not recording")
        self.skeleton_status_label.setAlignment(Qt.AlignCenter)
        self.skeleton_status_label.setStyleSheet("color: gray; font-style: italic;")
        skeleton_layout.addWidget(self.skeleton_status_label)

        analysis_layout.addWidget(skeleton_group)
        
        # Baseline Controls Group
        baseline_group = QGroupBox("Baseline Controls")
        baseline_layout = QGridLayout(baseline_group)
        
        self.set_baseline_btn = QPushButton("📌 Set Baseline")
        self.set_baseline_btn.clicked.connect(self.set_baseline)
        baseline_layout.addWidget(self.set_baseline_btn, 0, 0)
        
        self.save_baseline_btn = QPushButton("💾 Save")
        self.save_baseline_btn.clicked.connect(self.save_baseline)
        baseline_layout.addWidget(self.save_baseline_btn, 0, 1)
        
        self.clear_baseline_btn = QPushButton("❌ Clear")
        self.clear_baseline_btn.clicked.connect(self.clear_baseline)
        baseline_layout.addWidget(self.clear_baseline_btn, 1, 0)
        
        self.upload_baseline_btn = QPushButton("📤 Upload")
        self.upload_baseline_btn.clicked.connect(self.upload_baseline)
        baseline_layout.addWidget(self.upload_baseline_btn, 1, 1)
        
        analysis_layout.addWidget(baseline_group)
        
        analysis_layout.addStretch()
        left_panel.addTab(analysis_tab, "Analysis")

        # ----- SETTINGS TAB -----
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setSpacing(10)
        
        # Thresholds Group
        thresholds_group = QGroupBox("Detection Thresholds")
        thresholds_layout = QVBoxLayout(thresholds_group)

        # Movement Threshold
        self.movement_threshold_input = ThresholdInput(
            "Movement:", 0.0000, 0.0100, 0.0005, decimals=6, step=0.0001)
        self.movement_threshold = 0.0005
        self.movement_threshold_input.input.setValue(0.0005)
        self.movement_threshold_input.slider.setValue(500)
        self.movement_threshold_input.slider.valueChanged.connect(self.on_threshold_change)
        self.movement_threshold_input.input.valueChanged.connect(self.on_threshold_input_change)
        thresholds_layout.addWidget(self.movement_threshold_input)

        # Sound Threshold
        self.sound_threshold_input = ThresholdInput(
            "Sound:", 1000, 15000, 8000, decimals=0)
        self.sound_threshold_input.slider.valueChanged.connect(self.on_sound_threshold_change)
        self.sound_threshold_input.input.valueChanged.connect(self.on_sound_threshold_input_change)
        thresholds_layout.addWidget(self.sound_threshold_input)

        # Stability Time Threshold
        stability_layout = QHBoxLayout()
        stability_threshold_label = QLabel("Stability Time:")
        stability_layout.addWidget(stability_threshold_label)
        
        self.stability_threshold_input = QDoubleSpinBox()
        self.stability_threshold_input.setRange(1.0, 10.0)
        self.stability_threshold_input.setValue(3.0)
        self.stability_threshold_input.setSingleStep(0.1)
        self.stability_threshold_input.valueChanged.connect(self.update_stability_threshold)
        self.stability_threshold_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #444444;
                color: #4a90e2;
                font-weight: bold;
                border: 1px solid #555555;
                padding: 4px;
                border-radius: 3px;
            }
        """)
        stability_layout.addWidget(self.stability_threshold_input)
        stability_layout.addWidget(QLabel("seconds"))
        
        thresholds_layout.addLayout(stability_layout)
        settings_layout.addWidget(thresholds_group)
        
        # Clear log button in a separate group
        log_group = QGroupBox("Log Controls")
        log_layout = QVBoxLayout(log_group)
        
        clear_log_btn = QPushButton("Clear Log Messages")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        log_layout.addWidget(clear_log_btn)
        
        settings_layout.addWidget(log_group)
        settings_layout.addStretch()
        left_panel.addTab(settings_tab, "Settings")

        # ========== CENTER PANEL: VIDEO AND LOG ==========
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(12)
        
        # Video display with border
        video_container = QFrame()
        video_container.setFrameShape(QFrame.StyledPanel)
        video_container.setStyleSheet("""
            QFrame {
                border: 2px solid #444444;
                border-radius: 6px;
                background-color: #222222;
                padding: 4px;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(8, 8, 8, 8)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        
        center_layout.addWidget(video_container, stretch=4)
        
        # Log message area
        log_group = QGroupBox("Log Messages")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #00ff00;
                font-family: Consolas, monospace;
                font-size: 10pt;
                border: none;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        center_layout.addWidget(log_group, stretch=1)

        # ========== RIGHT PANEL: STATS AND CONTROLS ==========
        right_panel = QWidget()
        right_panel.setFixedWidth(250)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        
        # Shot information with improved style
        shot_group = QGroupBox("Current Shot Information")
        shot_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12pt;
                border: 2px solid #4a90e2;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 24px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-position: top center;
                color: #4a90e2;
            }
        """)
        shot_layout = QGridLayout(shot_group)
        shot_layout.setVerticalSpacing(8)
        shot_layout.setHorizontalSpacing(12)

        self.stat_labels = {}
        stats = [
            ("shot_number", "SHOT #"),
            ("up_time", "UP TIME"),
            ("shot_count", "SHOT COUNT"),
            ("status", "STATUS"),
            ("prep_time", "PREP TIME"),
            ("aim_time", "AIM TIME"),
            ("after_time", "AFTER TIME"),
            ("position", "POSITION"),
            ("stability", "STABILITY")
        ]

        for i, (key, text) in enumerate(stats):
            label = QLabel(text)
            label.setStyleSheet("color: #cccccc; font-weight: bold;")
            value = QLabel("0")
            value.setStyleSheet("color: #4a90e2; font-weight: bold; font-size: 11pt;")
            shot_layout.addWidget(label, i, 0)
            shot_layout.addWidget(value, i, 1)
            self.stat_labels[key] = value

        right_layout.addWidget(shot_group)
        
        # Phase indicator (visual timeline)
        phase_group = QGroupBox("Shot Phases")
        phase_layout = QVBoxLayout(phase_group)
        
        phase_row = QHBoxLayout()
        phases = [
            ("Preparing", "#ffc107"),
            ("Aiming", "#28a745"),
            ("After Shot", "#17a2b8")
        ]
        
        for phase_name, color in phases:
            phase_label = QLabel(phase_name)
            phase_label.setAlignment(Qt.AlignCenter)
            phase_label.setStyleSheet(f"""
                background-color: {color};
                color: white;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
            """)
            phase_row.addWidget(phase_label)
        
        phase_layout.addLayout(phase_row)
        right_layout.addWidget(phase_group)

        # Add all panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, stretch=1)
        main_layout.addWidget(right_panel)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel: Settings and Video Replay
        left_panel = QTabWidget()
        left_panel.setFixedWidth(200)

        # Settings Tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Test Mode Group
        test_mode_group = QGroupBox("Test Mode")
        test_mode_layout = QVBoxLayout(test_mode_group)
        
        # Test mode toggle button
        self.test_mode_btn = QPushButton("Enable Test Mode")
        self.test_mode_btn.setCheckable(True)
        self.test_mode_btn.clicked.connect(self.toggle_test_mode)
        self.test_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked:hover {
                background-color: #218838;
            }
        """)
        
        # Test mode info label
        self.test_mode_label = QLabel("Test mode is OFF")
        self.test_mode_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        self.test_mode_label.setAlignment(Qt.AlignCenter)
        
        test_mode_layout.addWidget(self.test_mode_btn)
        test_mode_layout.addWidget(self.test_mode_label)
        
        # Camera Control Group
        camera_group = QGroupBox("Camera Control")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera toggle button
        self.camera_btn = QPushButton("Disable Camera")
        self.camera_btn.setCheckable(True)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked:hover {
                background-color: #c82333;
            }
        """)
        
        # Camera status label
        self.camera_label = QLabel("Camera is ON")
        self.camera_label.setStyleSheet("color: #28a745; font-weight: bold;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        camera_layout.addWidget(self.camera_btn)
        camera_layout.addWidget(self.camera_label)
        
        settings_layout.addWidget(test_mode_group)
        settings_layout.addWidget(camera_group)
        settings_layout.addStretch()
        # Add Calculate CoG Button
        self.calculate_cog_btn = QPushButton("Calculate CoG")
        self.calculate_cog_btn.clicked.connect(self.calculate_cog)
        settings_layout.addWidget(self.calculate_cog_btn)
        left_panel.addTab(settings_tab, "Settings")
       
          
        # Video Replay Tab
        replay_tab = QWidget()
        replay_layout = QVBoxLayout(replay_tab)
        
        # Video list group
        video_group = QGroupBox("Recorded Shots")
        video_group_layout = QVBoxLayout(video_group)
        
        # Add refresh and management buttons in a horizontal layout
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_video_list)
        btn_layout.addWidget(refresh_btn)
        
        delete_btn = QPushButton("🗑 Delete")
        delete_btn.clicked.connect(self.delete_selected_video)
        btn_layout.addWidget(delete_btn)
        
        video_group_layout.addLayout(btn_layout)
        
        # Add video list
        self.video_list = QListWidget()
        self.video_list.itemDoubleClicked.connect(self.play_selected_video)
        video_group_layout.addWidget(self.video_list)
        
        replay_layout.addWidget(video_group)
        
        # Shot Information Group
        info_group = QGroupBox("Shot Information")
        info_layout = QGridLayout(info_group)
        
        self.shot_info_labels = {}
        info_fields = [
            ("shot_number", "SHOT #"),
            ("up_time", "UP TIME"),
            ("shot_count", "SHOT COUNT"),
            ("status", "STATUS"),
            ("prep_time", "PREP TIME"),
            ("aim_time", "AIM TIME"),
            ("after_time", "AFTER TIME"),
            ("position", "POSITION"),
            ("stability", "STABILITY")
        ]
        
        for i, (key, text) in enumerate(info_fields):
            label = QLabel(text)
            value = QLabel("-")
            value.setStyleSheet("color: #4a90e2; font-weight: bold; ")
            info_layout.addWidget(label, i, 0)
            info_layout.addWidget(value, i, 1)
            self.shot_info_labels[key] = value
            
        replay_layout.addWidget(info_group)
        
        # Video Controls
        self.video_controls = VideoControls()
        self.video_controls.play_pause_btn.clicked.connect(self.toggle_pause)
        self.video_controls.stop_btn.clicked.connect(self.stop_playback)
        self.video_controls.forward_btn.clicked.connect(lambda: self.seek_relative(30))
        self.video_controls.backward_btn.clicked.connect(lambda: self.seek_relative(-30))
        self.video_controls.speed_slider.valueChanged.connect(self.update_speed)
        self.video_controls.progress_slider.sliderMoved.connect(self.seek_to_position)
        self.video_controls.setVisible(False)
        
        replay_layout.addWidget(self.video_controls)
        
        # Return to live button
        return_live_btn = QPushButton("📹 Return to Live")
        return_live_btn.clicked.connect(self.return_to_live)
        return_live_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        replay_layout.addWidget(return_live_btn)
        
        left_panel.addTab(replay_tab, "Video Replay")

        # Center Panel: Video and Log
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.video_label, stretch=4)  # Give video more space
        
        # Log message area
        log_group = QGroupBox("Log Messages")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #00ff00;
                font-family: Consolas, monospace;
                font-size: 10pt;
                border: none;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        log_layout.addWidget(clear_log_btn)
        
        center_layout.addWidget(log_group, stretch=1)

        # Right Panel: Stats
        right_panel = QWidget()
        right_panel.setFixedWidth(250)
        right_layout = QVBoxLayout(right_panel)
        shot_group = QGroupBox("Shot Information")
        shot_layout = QGridLayout(shot_group)

        self.stat_labels = {}
        stats = [
            ("shot_number", "SHOT #"),
            ("up_time", "UP TIME"),
            ("shot_count", "SHOT COUNT"),
            ("status", "STATUS"),
            ("prep_time", "PREP TIME"),
            ("aim_time", "AIM TIME"),
            ("after_time", "AFTER TIME"),
            ("position", "POSITION"),
            ("stability", "STABILITY")
        ]

        for i, (key, text) in enumerate(stats):
            label = QLabel(text)
            value = QLabel("0")
            value.setStyleSheet("color: #4a90e2; font-weight: bold;")
            shot_layout.addWidget(label, i, 0)
            shot_layout.addWidget(value, i, 1)
            self.stat_labels[key] = value

        right_layout.addWidget(shot_group)

        # Thresholds Group
        thresholds_group = QGroupBox("Thresholds")
        thresholds_layout = QVBoxLayout(thresholds_group)

        # Movement Threshold (uses decimals)
        self.movement_threshold_input = ThresholdInput(
            "Movement:", 0.0000, 0.0100, 0.0005, decimals=6, step=0.0001)
        self.movement_threshold = 0.0005  # Set initial movement threshold
        self.movement_threshold_input.input.setValue(0.0005)  # Set initial value in input
        self.movement_threshold_input.slider.setValue(500)  # Set initial value in slider
        self.movement_threshold_input.slider.valueChanged.connect(self.on_threshold_change)
        self.movement_threshold_input.input.valueChanged.connect(self.on_threshold_input_change)
        thresholds_layout.addWidget(self.movement_threshold_input)

        # Sound Threshold (integer values)
        self.sound_threshold_input = ThresholdInput(
            "Sound:", 1000, 15000, 8000, decimals=0)
        self.sound_threshold_input.slider.valueChanged.connect(self.on_sound_threshold_change)
        self.sound_threshold_input.input.valueChanged.connect(self.on_sound_threshold_input_change)
        thresholds_layout.addWidget(self.sound_threshold_input)

        # Stability Threshold Input
        stability_threshold_label = QLabel("Stability Time Threshold:")
        self.stability_threshold_input = QDoubleSpinBox()
        self.stability_threshold_input.setRange(1.0, 10.0)
        self.stability_threshold_input.setValue(3.0)
        self.stability_threshold_input.setSingleStep(0.1)
        self.stability_threshold_input.valueChanged.connect(self.update_stability_threshold)
        thresholds_layout.addWidget(stability_threshold_label)
        thresholds_layout.addWidget(self.stability_threshold_input)

        right_layout.addWidget(thresholds_group)
        # NOW ADD THE COG ANALYSIS GROUP HERE (after right_layout is defined):
        cog_analysis_group = QGroupBox("CoG Analysis")
        cog_analysis_layout = QVBoxLayout(cog_analysis_group)

        self.track_cog_btn = QPushButton("Track Balance")
        self.track_cog_btn.setCheckable(True)
        self.track_cog_btn.clicked.connect(self.toggle_cog_tracking)
        cog_analysis_layout.addWidget(self.track_cog_btn)

        self.reset_cog_btn = QPushButton("Reset CoG History")
        self.reset_cog_btn.clicked.connect(self.reset_cog_history)
        cog_analysis_layout.addWidget(self.reset_cog_btn)

        self.save_cog_data_btn = QPushButton("Save CoG Analysis")
        self.save_cog_data_btn.clicked.connect(self.save_cog_data)
        cog_analysis_layout.addWidget(self.save_cog_data_btn)
        
        

        right_layout.addWidget(cog_analysis_group)

        # Instead of adding baseline controls to right_layout,
        # add them to the settings_layout in the left panel

        # Add baseline controls to the settings tab
        baseline_group = QGroupBox("Baseline Controls")
        baseline_layout = QVBoxLayout(baseline_group)

        # Add Set Baseline button
        self.set_baseline_btn = QPushButton("📌 Set Baseline")
        self.set_baseline_btn.clicked.connect(self.set_baseline)
        self.set_baseline_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        baseline_layout.addWidget(self.set_baseline_btn)

        # Add Save Baseline button
        self.save_baseline_btn = QPushButton("💾 Save Baseline")
        self.save_baseline_btn.clicked.connect(self.save_baseline)
        self.save_baseline_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        baseline_layout.addWidget(self.save_baseline_btn)

        # Add Clear Baseline button
        self.clear_baseline_btn = QPushButton("❌ Clear Baseline")
        self.clear_baseline_btn.clicked.connect(self.clear_baseline)
        self.clear_baseline_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        baseline_layout.addWidget(self.clear_baseline_btn)

        # Add Upload Baseline button
        self.upload_baseline_btn = QPushButton("📤 Upload Baseline")
        self.upload_baseline_btn.clicked.connect(self.upload_baseline)
        self.upload_baseline_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: black;
                padding: 8px;
                border: none;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        baseline_layout.addWidget(self.upload_baseline_btn)

        # Add to the settings layout in left panel instead of right_layout
        settings_layout.addWidget(baseline_group)

        # Add Dashboard button
        self.dashboard_btn = QPushButton("Open Analysis Dashboard")
        self.dashboard_btn.clicked.connect(self.launch_dashboard)
        self.dashboard_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        right_layout.addWidget(self.dashboard_btn)

        # Add all panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, stretch=1)
        main_layout.addWidget(right_panel)
        
        skeleton_group = QGroupBox("Skeleton Recording")
        skeleton_layout = QVBoxLayout(skeleton_group)

        # Record buttons layout
        recording_btn_layout = QHBoxLayout()
        self.start_skeleton_btn = QPushButton("▶️ Start Recording")
        self.start_skeleton_btn.clicked.connect(self.start_skeleton_recording)
        self.stop_skeleton_btn = QPushButton("⏹️ Stop Recording")
        self.stop_skeleton_btn.clicked.connect(self.stop_skeleton_recording)
        self.stop_skeleton_btn.setEnabled(False)
        recording_btn_layout.addWidget(self.start_skeleton_btn)
        recording_btn_layout.addWidget(self.stop_skeleton_btn)
        skeleton_layout.addLayout(recording_btn_layout)

        # Playback button
        self.view_skeleton_btn = QPushButton("🎬 View Recordings")
        self.view_skeleton_btn.clicked.connect(self.view_skeleton_recordings)
        skeleton_layout.addWidget(self.view_skeleton_btn)

        # Debug button to reload CSV data
        self.reload_csv_btn = QPushButton("🔄 Reload CSV Data")
        self.reload_csv_btn.clicked.connect(self.reload_csv_data)
        self.reload_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: black;
                padding: 5px;
                border: none;
                border-radius: 3px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        skeleton_layout.addWidget(self.reload_csv_btn)

        # Recording status
        self.skeleton_status_label = QLabel("Not recording")
        self.skeleton_status_label.setAlignment(Qt.AlignCenter)
        self.skeleton_status_label.setStyleSheet("color: gray; font-style: italic;")
        skeleton_layout.addWidget(self.skeleton_status_label)

        right_layout.addWidget(skeleton_group)
    def toggle_cog_tracking(self):
        """Toggle continuous CoG tracking."""
        self.cog_tracking_enabled = self.track_cog_btn.isChecked()
        if self.cog_tracking_enabled:
            self.track_cog_btn.setText("Stop Tracking")
            self.log_message("CoG tracking started")
            # Initialize history and reference position
            self.cog_history = []
            self.cog_reference_position = None
            
            # Start continuous tracking
            self.track_cog()
        else:
            self.track_cog_btn.setText("Track Balance")
            self.log_message("CoG tracking stopped")

    def toggle_cog_tracking(self):
        """Toggle continuous CoG tracking."""
        self.cog_tracking_enabled = self.track_cog_btn.isChecked()
        if self.cog_tracking_enabled:
            self.track_cog_btn.setText("Stop Tracking")
            self.log_message("CoG tracking started")
            # Initialize history and reference position
            self.cog_history = []
            self.cog_reference_position = None
        else:
            self.track_cog_btn.setText("Track Balance")
            self.log_message("CoG tracking stopped")

    def reset_cog_history(self):
        """Reset the CoG history data."""
        self.cog_history = []
        self.cog_reference_position = None
        self.log_message("CoG history reset")

    def save_cog_data(self):
        """Save CoG tracking data to CSV."""
        if not hasattr(self, 'cog_history') or len(self.cog_history) < 2:
            QMessageBox.warning(self, "Warning", "No CoG data available to save.")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(
            self.session_folder, 
            f"{self.current_player}_session{self.current_session:03d}_cog_analysis_{timestamp}.csv"
        )
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Frame', 'X_Pixel', 'Y_Pixel', 'X_Normalized', 'Y_Normalized'])
                
                h, w = self.cap.read()[1].shape[:2] if self.cap.isOpened() else (480, 640)
                
                for i, (px, py) in enumerate(self.cog_history):
                    # Convert pixels back to normalized coordinates
                    norm_x = px / w
                    norm_y = py / h
                    writer.writerow([i, px, py, norm_x, norm_y])
                    
            self.log_message(f"CoG data saved to {filename}")
            QMessageBox.information(self, "Success", f"CoG data saved successfully to:\n{filename}")
        except Exception as e:
            self.log_message(f"Error saving CoG data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save CoG data:\n{str(e)}")

    def on_threshold_change(self, val):
        """Handle movement threshold slider change"""
        value = val / 1000000.0  # Scale from slider value to actual threshold
        self.movement_threshold = value
        # Don't reset any timers or states when threshold changes
        self.log_message(f"Movement threshold changed to: {value:.6f}")

    def on_threshold_input_change(self, value):
        """Handle movement threshold direct input change"""
        self.movement_threshold = value
        # Don't reset any timers or states when threshold changes
        self.log_message(f"Movement threshold changed to: {value:.6f}")

    def on_sound_threshold_change(self, val):
        """Handle sound threshold slider change"""
        self.sound_threshold = val
        self.log_message(f"Sound threshold changed to: {val}")

    def on_sound_threshold_input_change(self, value):
        """Handle sound threshold direct input change"""
        self.sound_threshold = int(value)
        self.log_message(f"Sound threshold changed to: {value}")

    def on_stability_threshold_change(self, val):
        """Handle stability threshold slider change"""
        value = val / 10.0
        self.stable_time_threshold = value
        self.log_message(f"Stability time threshold changed to: {value:.1f}s")

    def on_stability_threshold_input_change(self, value):
        """Handle stability threshold direct input change"""
        self.stable_time_threshold = value
        self.log_message(f"Stability time threshold changed to: {value:.1f}s")

    def detect_shot_audio(self):
        CHUNK = 1024
        RATE = 44100
        COOLDOWN = 0.5
        
        # Initialize sound threshold
        self.sound_threshold = 8000

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

        self.log_message("🎧 Shot detection started...")
        last_Shot_time = 0

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                rms = audioop.rms(data, 2)
                current_time = time.time()

                if rms > self.sound_threshold and (current_time - last_Shot_time) > COOLDOWN:
                    self.log_message("👏 Shot detected")
                    last_Shot_time = current_time
                    with self.data_lock:
                        shot = self.combo_counter
                        time_up = self.up_time_seconds
                        if self.current_status == "aiming":
                            self.log_message(f"Transitioning from {self.current_status} to after_shot")
                            self.current_status = "after_shot"
                            self.stability_progress = 0
                            self.last_movement_time = None
                            # Reset and start the after_shot timer
                            self.after_shot_duration = 0
                            self.phase_start_time = current_time
                            self.log_message(f"After shot timer started at: {current_time}")
                            self.save_Shot_info(shot, time_up)
            except Exception as e:
                self.log_message(f"Audio error: {e}")
                break

    def update_frame(self, current_time=None):
        if not self.timer_enabled:
            return

        # Skip frame processing if camera is disabled
        if not self.camera_enabled:
            # Create a black frame
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add "Camera Disabled" text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Camera Disabled"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (640 - text_size[0]) // 2
            text_y = (480 + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            
            # Convert to QImage and display
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return

        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                current_time = time.time() if current_time is None else current_time
                height, width, _ = image.shape

                # Define phase_colors before use
                phase_colors = {
                    "free": (200, 200, 200),
                    "preparing": (0, 255, 255),
                    "aiming": (0, 255, 0),
                    "after_shot": (255, 0, 0)
                }

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                line_height = 22

                phase_lines = [
                    (f"Phase: {self.current_status.upper()}", phase_colors.get(self.current_status, (200, 200, 200))),
                    (f"Preparing: {self.preparing_duration:.1f}s", phase_colors["preparing"]),
                    (f"Aiming: {self.aiming_duration:.1f}s", phase_colors["aiming"]),
                    (f"After Shot: {self.after_shot_duration:.1f}s", phase_colors["after_shot"]),
                ]
                y_pos = 30
                for text, color in phase_lines:
                    (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    x = width - text_width - 10
                    cv2.putText(image, text, (x, y_pos), font, font_scale, color, thickness)
                    y_pos += line_height

                # --- TEST MODE OVERLAY (MAX RIGHT) ---
                if self.test_mode:
                    test_values = [
                        f"TEST MODE - Shot #{self.test_shot_counter}",
                        f"Status: {self.current_status.upper()}",
                        f"Position: {'READY' if self.is_in_shooting_position else 'NOT READY'}",
                        f"Up Time: {self.up_time_seconds}s",
                        f"Prep Time: {self.preparing_duration:.1f}s",
                        f"Aim Time: {self.aiming_duration:.1f}s",
                        f"After Time: {self.after_shot_duration:.1f}s",
                        f"Stability: {int(self.stability_progress * 100)}%",
                        f"Movement: {self.movement_threshold:.6f}"
                    ]
                    y0 = 140
                    current_y = y0 + 20
                    for i, text in enumerate(test_values):
                        if i == 0:
                            color = (50, 255, 50)
                        elif "Status" in text:
                            color = {
                                'FREE': (200, 200, 200),
                                'PREPARING': (50, 255, 255),
                                'AIMING': (50, 255, 50),
                                'AFTER_SHOT': (255, 200, 50)
                            }.get(self.current_status.upper(), (200, 200, 200))
                        elif "Position" in text:
                            color = (50, 255, 50) if self.is_in_shooting_position else (50, 50, 255)
                        elif "Stability" in text:
                            progress = self.stability_progress
                            color = (
                                50,
                                int(255 * progress),
                                int(255 * (1 - progress))
                            )
                        else:
                            color = (255, 255, 255)
                        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        x = width - text_width - 10
                        cv2.putText(image, text, (x, current_y), font, font_scale, color, thickness)
                        current_y += line_height

                # Movement detection visualization
                if self.last_landmarks is not None and self.current_status != "after_shot":
                    right_shoulder_idx = 12
                    right_elbow_idx = 14
                    
                    last_shoulder = self.last_landmarks[right_shoulder_idx]
                    current_shoulder = landmarks[right_shoulder_idx]
                    last_elbow = self.last_landmarks[right_elbow_idx]
                    current_elbow = landmarks[right_elbow_idx]
                    
                    shoulder_y_diff = abs(current_shoulder.y - last_shoulder.y)
                    elbow_y_diff = abs(current_elbow.y - last_elbow.y)
                    combined_y_diff = (shoulder_y_diff + elbow_y_diff) / 2
                    
                    # Draw movement indicators
                    shoulder_cx = int(current_shoulder.x * width)
                    shoulder_cy = int(current_shoulder.y * height)
                    elbow_cx = int(current_elbow.x * width)
                    elbow_cy = int(current_elbow.y * height)
                    
                    movement_detected = combined_y_diff > self.movement_threshold
                    
                    if movement_detected:
                        cv2.circle(image, (shoulder_cx, shoulder_cy), 10, (0, 0, 255), -1)
                        cv2.circle(image, (elbow_cx, elbow_cy), 10, (0, 0, 255), -1)
                        if self.current_status == "free" and self.is_in_shooting_position:
                            self.current_status = "preparing"
                            self.phase_start_time = current_time
                            self.preparing_duration = 0  # Initialize preparation time
                        # Update movement time but don't reset preparation timer
                        self.last_movement_time = current_time
                        self.stability_progress = 0
                    else:
                        cv2.circle(image, (shoulder_cx, shoulder_cy), 10, (0, 255, 0), -1)
                        cv2.circle(image, (elbow_cx, elbow_cy), 10, (0, 255, 0), -1)
                        if self.current_status == "preparing" and self.last_movement_time is not None:
                            elapsed_time = current_time - self.last_movement_time
                            self.stability_progress = min(elapsed_time / self.stable_time_threshold, 1.0)
                            if self.stability_progress >= 1.0:
                                self.current_status = "aiming"
                                self.phase_start_time = current_time

                self.last_landmarks = landmarks
                self.check_shooting_position(landmarks)
                self.update_timers(current_time)
                
                # Only update stats panel if not in test mode
                if not self.test_mode:
                    self.update_stats()

                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(66, 245, 221), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(66, 100, 245), thickness=2, circle_radius=2)
                )

                if self.baseline_skeleton and self.baseline_anchor and not self.test_mode:
                    hip_left = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                    hip_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                    current_anchor = ((hip_left.x + hip_right.x) / 2, (hip_left.y + hip_right.y) / 2)
                    highlight_joints = []
                    if self.baseline_angles:
                        current_angles = self.get_current_angles()
                        angles_to_check = {
                            "left_elbow": (11, 13, 15),
                            "right_elbow": (12, 14, 16),
                            "left_shoulder": (23, 11, 13),
                            "right_shoulder": (24, 12, 14),
                        }
                        for name, ids in angles_to_check.items():
                            if name in self.baseline_angles:
                                baseline = self.baseline_angles[name]
                                current = current_angles[name]
                                if abs(current - baseline) > 1:  # ANGLE_THRESHOLD
                                    highlight_joints.extend(ids)
                    self.draw_baseline_skeleton(
                        image, width, height, self.baseline_skeleton,
                        current_anchor, self.baseline_anchor, set(highlight_joints)
                    )
                    # Check if CoG tracking is enabled and process it
                if hasattr(self, 'cog_tracking_enabled') and self.cog_tracking_enabled:
                    # Use BGR copy of the frame for CoG visualization
                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Calculate CoG on the current landmarks
                    segments = [
                        ((0, 1),   0.081, 0.50),  # head→neck
                        ((11,23),  0.432, 0.43),  # shoulders→hips
                        ((23,25),  0.105, 0.36),  # left thigh
                        ((24,26),  0.105, 0.36),  # right thigh
                        ((25,27),  0.046, 0.42),  # left shank
                        ((26,28),  0.046, 0.42),  # right shank
                        ((27,31),  0.015, 0.50),  # left foot
                        ((28,32),  0.015, 0.50),  # right foot
                        ((11,13),  0.027, 0.436), # left upper arm
                        ((12,14),  0.027, 0.436), # right upper arm
                        ((13,15),  0.023, 0.43),  # left forearm+hand
                        ((14,16),  0.023, 0.43),  # right forearm+hand
                    ]

                    missing_indices = [seg for seg,_,_ in segments
                            if seg[0] >= len(landmarks) or seg[1] >= len(landmarks)]
                            
                    if not missing_indices:
                        # Check visibility of all landmarks
                        VISIBILITY_THRESHOLD = 0.5
                        used_landmarks = set()
                        for (a, b), _, _ in segments:
                            used_landmarks.add(a)
                            used_landmarks.add(b)
                        
                        low_visibility = [idx for idx in used_landmarks 
                                        if landmarks[idx].visibility < VISIBILITY_THRESHOLD]
                        
                        if not low_visibility:
                            # Calculate CoG
                            total_x = total_y = total_mass = 0.0
                            for (a, b), mass_frac, cog_frac in segments:
                                p1, p2 = landmarks[a], landmarks[b]
                                cog_x = p1.x + cog_frac * (p2.x - p1.x)
                                cog_y = p1.y + cog_frac * (p2.y - p1.y)
                                total_x += cog_x * mass_frac
                                total_y += cog_y * mass_frac
                                total_mass += mass_frac

                            if total_mass > 0:
                                final_x = total_x / total_mass
                                final_y = total_y / total_mass
                                
                                # Draw CoG on this frame
                                self.draw_cog(final_x, final_y, bgr_image)
                                
                                # Skip the normal frame display since draw_cog handles it
                                return

            # Normal frame display path (reached if CoG tracking isn't active or there's an issue)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Record frame if recording is active and not in test mode
            if self.recording and self.video_writer and not self.test_mode:
                self.video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Add in update_frame method after results.pose_landmarks processing block
            if results.pose_landmarks and self.skeleton_recording:
                # Capture skeleton frame if recording is active
                self.capture_skeleton_frame(results.pose_landmarks.landmark)

        # ────────────────────────────────────────────────────────────────
    # 1)  CENTER-OF-GRAVITY  CALCULATION
    # ────────────────────────────────────────────────────────────────
    def calculate_cog(self, continuous: bool = False):
        """Compute the 2-D CoG (Dempster fractions) and hand it to draw_cog()."""
        if not continuous:
            print("► Calc CoG called")
            self.log_message("Calculating Center of Gravity (CoG)…")

        landmarks = getattr(self, "last_landmarks", None)
        if not landmarks:
            if not continuous:
                QMessageBox.warning(
                    self, "Warning",
                    "No pose detected! Make sure you are fully visible."
                )
            return

        # Dempster mass fractions (segment start → end, mass %, local-CoG fraction)
        segments = [
            ((0, 1),   0.081, 0.50),   # head
            ((11, 23), 0.432, 0.43),   # trunk
            ((23, 25), 0.105, 0.36),   # left thigh
            ((24, 26), 0.105, 0.36),   # right thigh
            ((25, 27), 0.046, 0.42),   # left shank
            ((26, 28), 0.046, 0.42),   # right shank
            ((27, 31), 0.015, 0.50),   # left foot
            ((28, 32), 0.015, 0.50),   # right foot
            ((11, 13), 0.027, 0.436),  # left upper arm
            ((12, 14), 0.027, 0.436),  # right upper arm
            ((13, 15), 0.023, 0.43),   # left forearm+hand
            ((14, 16), 0.023, 0.43),   # right forearm+hand
        ]

        # Visibility check (optional)
        low_vis = [
            idx for idx in {i for seg, _, _ in segments for i in seg}
            if landmarks[idx].visibility < 0.5
        ]
        if low_vis and not continuous:
            msg = f"Low-visibility landmarks: {low_vis}"
            
            self.log_message(msg)
            QMessageBox.warning(self, "Warning", msg)
            return

        # Weighted average
        tot_x = tot_y = tot_m = 0.0
        for (a, b), m, f in segments:
            pa, pb = landmarks[a], landmarks[b]
            seg_x = pa.x + f * (pb.x - pa.x)
            seg_y = pa.y + f * (pb.y - pa.y)
            tot_x += seg_x * m
            tot_y += seg_y * m
            tot_m += m

        if tot_m == 0:
            print("ERROR: total mass fraction zero")
            return

        cog_x = tot_x / tot_m
        cog_y = tot_y / tot_m
        
        self.log_message(f"CoG (norm): ({cog_x:.4f}, {cog_y:.4f})")

        self.draw_cog(cog_x, cog_y)          # always draw
        self.last_cog_position = (cog_x, cog_y)



    # ────────────────────────────────────────────────────────────────
    # 2)  DRAW ROUTINE WITH FULL PRINTS
    # ────────────────────────────────────────────────────────────────
    def draw_cog(self, norm_x: float, norm_y: float, frame=None):
        """Render CoG + projection arrow + sway + verbose prints every frame."""
        # -------------------------------------------------------------
        # Capture or use supplied frame
        # -------------------------------------------------------------
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                
                self.log_message("Failed to grab frame.")
                return

        h, w = frame.shape[:2]
        cog_px, cog_py = int(norm_x * w), int(norm_y * h)
        cog_pt = (cog_px, cog_py)
        disp = frame.copy()

        # -------------------------------------------------------------
        # History
        # -------------------------------------------------------------
        if not hasattr(self, "cog_history"):
            self.cog_history = []
        self.cog_history.append(cog_pt)

        # -------------------------------------------------------------
        # Landmark-derived points
        # -------------------------------------------------------------
        try:
            lm = self.last_landmarks
            l_ank = (int(lm[27].x * w), int(lm[27].y * h))
            r_ank = (int(lm[28].x * w), int(lm[28].y * h))
            l_hip = (int(lm[23].x * w), int(lm[23].y * h))
            r_hip = (int(lm[24].x * w), int(lm[24].y * h))

            foot_mid = ((l_ank[0] + r_ank[0]) // 2,
                        (l_ank[1] + r_ank[1]) // 2)

            # Print raw numbers
            print(f"FootMid {foot_mid} | CoGPx {cog_pt}")

            # Draw base & hip lines
            cv2.line(disp, l_ank, r_ank, (0, 255, 0), 2)   # Green ankle line
            cv2.line(disp, l_hip, r_hip, (255, 0, 0), 2)   # Red hip line

            # Projection arrow
            cv2.arrowedLine(disp, cog_pt, foot_mid, (0, 165, 255), 2)  # Orange

            # Extra dots for clarity
            cv2.circle(disp, foot_mid, 6, (255, 0, 255), -1)           # Magenta
            cv2.circle(disp, cog_pt, 6, (0, 0, 255), -1)               # Red

            # Balance %
            foot_width = np.linalg.norm(np.subtract(l_ank, r_ank))
            if foot_width:
                offset_pct = ((foot_mid[0] - cog_pt[0]) / foot_width) * 100
                side = "right" if offset_pct > 0 else "left"
                txt = f"Lateral balance: {abs(offset_pct):.1f}% {side}"
                cv2.putText(disp, txt, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        except Exception as e:
            print(f"Landmark error: {e}")
            self.log_message(f"Landmark error: {e}")

        # -------------------------------------------------------------
        # CoG crosshairs + trail
        # -------------------------------------------------------------
        cv2.line(disp, (cog_px, 0), (cog_px, h), (0, 0, 255), 2)
        cv2.line(disp, (0, cog_py), (w, cog_py), (0, 0, 255), 1)

        for i in range(1, len(self.cog_history)):
            cv2.line(disp, self.cog_history[i-1], self.cog_history[i],
                    (255, 255, 0), 2)                           # Cyan trace

        # -------------------------------------------------------------
        # Sway metrics
        # -------------------------------------------------------------
        if len(self.cog_history) > 5:
            xs = [p[0] for p in self.cog_history]
            ys = [p[1] for p in self.cog_history]
            lat = (max(xs) - min(xs)) / w * 100
            ver = (max(ys) - min(ys)) / h * 100
            cv2.putText(disp, f"Lateral sway: {lat:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(disp, f"Vertical sway: {ver:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
           

        # -------------------------------------------------------------
        # Record & display
        # -------------------------------------------------------------
        if self.recording and self.video_writer and not self.test_mode:
            self.video_writer.write(disp)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)
        )

    def check_shooting_position(self, landmarks):
        """Check shooting position based on arm positions and head orientation"""
        # Get angles for both arms
        l_angle = self.get_angle_by_landmarks(landmarks, 11, 13, 15)  # Left elbow angle
        r_angle = self.get_angle_by_landmarks(landmarks, 24, 12, 14)  # Right shoulder angle

        # Check left arm position (either in pocket or in biceps curl position)
        left_wrist_x = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x
        left_shoulder_x = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        hand_in_pocket = abs(left_wrist_x - left_shoulder_x) < 0.05

        # Left arm stages
        if l_angle > 160:
            self.left_stage = "down"
        if (l_angle < 140 and self.left_stage == "down") or hand_in_pocket:
            self.left_stage = "up"

        # Right arm stages
        if r_angle < 30:
            self.right_stage = "down"
        if r_angle > 80 and self.right_stage == "down":
            self.right_stage = "up"

        # Head orientation check
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value].x
        right_shoulder_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        left_shoulder_x = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        head_turned_right = abs(nose - right_shoulder_x) < abs(nose - left_shoulder_x) - 0.03

        # Combined position check
        was_in_position = self.is_in_shooting_position
        current_position = (
            self.left_stage == "up" and 
            self.right_stage == "up" and 
            head_turned_right
        )

        # Check for position change during after_shot
        if self.current_status == "after_shot" and was_in_position != current_position:
            self.log_message(f"Position changed during after_shot. Final duration: {self.after_shot_duration:.1f}s")
            # Save final state before transitioning (only if not in test mode)
            if not self.test_mode:
                self.save_session_data(
                    self.combo_counter,
                    self.up_time_seconds,
                    "after_shot",
                    self.preparing_duration,
                    self.aiming_duration,
                    self.after_shot_duration,
                    "No"
                )
            # Keep the current durations in the display
            self.update_stats()
            # Transition to free state
            self.current_status = "free"
            self.log_message("Transitioning to free state due to position change")

        # Update position after checking for changes
        self.is_in_shooting_position = current_position

        # Handle shot counting and recording
        if self.is_in_shooting_position:
            if self.ready_for_next and self.current_status == "free":
                with self.data_lock:
                    # Only reset durations when starting a new shot
                    self.preparing_duration = 0
                    self.aiming_duration = 0
                    self.after_shot_duration = 0
                    
                    if self.test_mode:
                        self.test_shot_counter += 1
                        self.log_message(f"🎯 Test shot #{self.test_shot_counter}")
                    else:
                        # Increment shot counter
                        self.combo_counter += 1
                        self.log_message(f"✅ Shot #{self.combo_counter}")
                    
                    self.ready_for_next = False
                    self.up_time_start = time.time()
                    self.up_time_seconds = 0

                    # Set baseline on first shot (only if not in test mode)
                    if self.combo_counter == 1 and not self.test_mode:
                        hip_left = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                        hip_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                        self.baseline_anchor = ((hip_left.x + hip_right.x) / 2, (hip_left.y + hip_right.y) / 2)
                        self.baseline_skeleton = [(lm.x, lm.y) for lm in landmarks]
                        self.baseline_angles = self.get_current_angles()
                        self.save_baseline_to_csv(self.baseline_skeleton, self.baseline_angles)
                        self.log_message("📌 Baseline skeleton and angles saved.")

                    # Start recording (only if not in test mode)
                    if not self.test_mode and not self.recording:
                        self.start_recording()
        else:
            if self.up_time_start and self.current_status != "after_shot":  # Don't reset during after_shot
                with self.data_lock:
                    self.up_time_seconds = int(time.time() - self.up_time_start)
                    self.display_up_time = self.up_time_seconds
                    
                    # Save incomplete shot data
                    if not self.test_mode:
                        self.save_session_data(
                            self.combo_counter,
                            self.up_time_seconds,
                            "free",
                            self.preparing_duration,
                            self.aiming_duration,
                            self.after_shot_duration,
                            "No"
                        )
                    
                    # Stop recording when shot ends
                    if self.recording:
                        self.stop_recording()

                self.ready_for_next = True
                self.up_time_start = None

    def update_timers(self, current_time):
        """Update all phase timers"""
        if self.is_in_shooting_position or self.current_status == "after_shot":  # Continue timing even if position lost during after_shot
            if self.current_status == "preparing":
                # Always update preparation time while in preparing state, regardless of movement
                self.preparing_duration = current_time - self.phase_start_time
            elif self.current_status == "aiming":
                self.aiming_duration = current_time - self.phase_start_time
            elif self.current_status == "after_shot":
                # Update after_shot duration
                new_duration = current_time - self.phase_start_time
                if new_duration > self.after_shot_duration:  # Only update if increasing
                    self.after_shot_duration = new_duration
        else:
            # Only reset timers if we completely leave shooting position and not in after_shot
            if self.current_status != "after_shot":
                self.current_status = "free"  # Reset to free state when leaving position
                self.phase_start_time = current_time
                self.preparing_duration = 0  # Reset preparation time when leaving position
                self.stability_progress = 0
                self.last_movement_time = None

        if self.up_time_start:
            with self.data_lock:
                self.up_time_seconds = int(current_time - self.up_time_start)

    def update_stats(self):
        """Update the statistics display"""
        # Show appropriate shot counter based on mode
        shot_number = self.test_shot_counter if self.test_mode else self.combo_counter
        self.stat_labels["shot_number"].setText(str(shot_number))
        
        # Add test mode indicator to shot number if in test mode
        if self.test_mode:
            self.stat_labels["shot_number"].setText(f"{shot_number} (Test)")
            self.stat_labels["shot_number"].setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.stat_labels["shot_number"].setStyleSheet("color: #4a90e2; font-weight: bold;")

        # Update other stats
        self.stat_labels["up_time"].setText(f"{self.up_time_seconds}s")
        self.stat_labels["shot_count"].setText(str(self.display_up_time))
        self.stat_labels["status"].setText(self.current_status.upper())
        self.stat_labels["prep_time"].setText(f"{self.preparing_duration:.1f}s")
        self.stat_labels["aim_time"].setText(f"{self.aiming_duration:.1f}s")
        self.stat_labels["after_time"].setText(f"{self.after_shot_duration:.1f}s")
        self.stat_labels["position"].setText("READY ✓" if self.is_in_shooting_position else "NOT READY ✗")
        self.stat_labels["stability"].setText(f"{int(self.stability_progress * 100)}%")

        # Update colors based on status
        status_colors = {
            'FREE': "#f8f9fa",      # Light
            'PREPARING': "#ffc107",  # Warning
            'AIMING': "#28a745",     # Success
            'AFTER_SHOT': "#17a2b8"  # Info
        }
        if self.current_status.upper() in status_colors:
            self.stat_labels["status"].setStyleSheet(f"color: {status_colors[self.current_status.upper()]}; font-weight: bold;")

        # Update position color
        position_color = "#28a745" if self.is_in_shooting_position else "#dc3545"
        self.stat_labels["position"].setStyleSheet(f"color: {position_color}; font-weight: bold;")

    def get_angle_by_landmarks(self, landmarks, idx_a, idx_b, idx_c):
        a = [landmarks[idx_a].x, landmarks[idx_a].y]
        b = [landmarks[idx_b].x, landmarks[idx_b].y]
        c = [landmarks[idx_c].x, landmarks[idx_c].y]
        return self.calculate_angle(a, b, c)

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def save_session_data(self, shot_number, up_time, status, preparing_time, aiming_time, after_shot_time, shooting_position):
        """Save shot data to the session CSV file"""
        # Skip shot number 0
        if shot_number == 0:
            return
            
        # Ensure the file exists
        if not os.path.exists(self.session_data_file):
            self.create_session_data_file()
            
        # Check if this shot has already been saved
        if shot_number <= self.last_saved_shot:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine if this is a complete shot
        complete_shot = "Yes" if preparing_time > 0 and aiming_time > 0 and after_shot_time > 0 else "No"
        
        # Prepare data row
        data = [
            self.current_player,
            f"Session {self.current_session:03d}",
            shot_number,
            timestamp,
            up_time,
            f"{preparing_time:.1f}",
            f"{aiming_time:.1f}",
            f"{after_shot_time:.1f}",
            complete_shot
        ]
        
        try:
            # Use Windows line endings
            with open(self.session_data_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
                self.last_saved_shot = shot_number
                self.log_message(f"📊 Shot {shot_number} data saved to {self.session_data_file}")
        except Exception as e:
            self.log_message(f"Error saving session data: {e}")
            QMessageBox.warning(self, "Warning", 
                f"Failed to save shot data:\n{str(e)}")

    def save_Shot_info(self, shot, time_up, filename="Shot_shots_log.csv"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, shot, time_up])
        
        self.Shot_events.append({
            "timestamp": timestamp,
            "shot": shot,
            "time_up": time_up
        })

    def closeEvent(self, event):
        """Handle application close"""
        self.is_playing_video = False  # Stop any video playback
        if self.video_cap:
            self.video_cap.release()
        if self.recording:
            self.stop_recording()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        event.accept()

    def keyPressEvent(self, event):
        # Remove the baseline keyboard shortcuts since we're using buttons now
        pass

    def draw_baseline_skeleton(self, image, width, height, baseline_relative, anchor_curr, anchor_baseline, highlight_joints=[]):
        overlay = image.copy()
        for i, (x_rel, y_rel) in enumerate(baseline_relative):
            x = anchor_curr[0] + (x_rel - anchor_baseline[0])
            y = anchor_curr[1] + (y_rel - anchor_baseline[1])
            cx, cy = int(x * width), int(y * height)
            color = (0, 0, 255) if i in highlight_joints else (0, 255, 255)
            cv2.circle(overlay, (cx, cy), 5, color, -1)

        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if (start_idx < len(baseline_relative) and end_idx < len(baseline_relative)):
                x1 = anchor_curr[0] + (baseline_relative[start_idx][0] - anchor_baseline[0])
                y1 = anchor_curr[1] + (baseline_relative[start_idx][1] - anchor_baseline[1])
                x2 = anchor_curr[0] + (baseline_relative[end_idx][0] - anchor_baseline[0])
                y2 = anchor_curr[1] + (baseline_relative[end_idx][1] - anchor_baseline[1])
                pt1 = int(x1 * width), int(y1 * height)
                pt2 = int(x2 * width), int(y2 * height)
                cv2.line(overlay, pt1, pt2, (0, 255, 255), 2)

        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    def get_current_angles(self):
        if not self.last_landmarks:
            return {}
            
        angles_to_check = {
            "left_elbow": (11, 13, 15),
            "right_elbow": (12, 14, 16),
            "left_shoulder": (23, 11, 13),
            "right_shoulder": (24, 12, 14),
        }
        
        return {
            name: self.get_angle_by_landmarks(self.last_landmarks, *ids)
            for name, ids in angles_to_check.items()
        }

    def save_baseline_to_csv(self, skeleton, angles, filename='baseline_data.csv'):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Create filename with player, session, shot number, and state
        csv_filename = os.path.join(
            self.session_folder, 
            f"{self.current_player}_session{self.current_session:03d}_shot{self.combo_counter}_{self.current_status}_{timestamp}_{filename}"
        )
        header = ['Player', 'Session', 'Landmark_X', 'Landmark_Y', 'Angle_Name', 'Angle_Value', 'Player_State']
        rows = []

        for i, (x, y) in enumerate(skeleton):
            rows.append([self.current_player, self.current_session, x, y, f"landmark_{i}", "", self.current_status])
        for name, angle in angles.items():
            rows.append([self.current_player, self.current_session, None, None, name, angle, self.current_status])

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        self.log_message(f"Baseline saved to {csv_filename} with player state: {self.current_status}")

    def load_baseline_from_csv(self, filename=None):
        if filename is None:
            # Get the most recent baseline file from the session folder
            baseline_files = [f for f in os.listdir(self.session_folder) if f.endswith('baseline_data.csv')]
            if not baseline_files:
                self.log_message("No baseline files found in session folder")
                return None, None
            filename = os.path.join(self.session_folder, sorted(baseline_files)[-1])
        
        try:
            with open(filename, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                # Process the file...
                skeleton = []
                angles = {}
                for row in reader:
                    if row[4].startswith('landmark'):
                        x, y = float(row[2]), float(row[3])
                        skeleton.append((x, y))
                    elif row[4] and row[5]:
                        angles[row[4]] = float(row[5])
                self.log_message(f"Baseline loaded from {filename}")
                return skeleton, angles
        except Exception as e:
            self.log_message(f"Error loading baseline from CSV: {e}")
            return None, None

    def refresh_video_list(self):
        """Refresh the list of recorded videos"""
        self.video_list.clear()
        # Use the correct videos folder path
        videos_folder = os.path.join(self.base_folder, self.current_player, f"session_{self.current_session:03d}", "videos")
        if os.path.exists(videos_folder):
            videos = [f for f in os.listdir(videos_folder) if f.endswith('.avi')]
            videos.sort(reverse=True)  # Most recent first
            self.video_list.addItems(videos)
            self.log_message(f"Loading videos from: {videos_folder}")
        else:
            self.log_message(f"Videos folder not found: {videos_folder}")

    def return_to_live(self):
        """Return to live camera feed"""
        self.is_playing_video = False
        self.timer_enabled = True
        self.video_controls.setVisible(False)
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.log_message("📹 Returned to live camera feed")

        # Clear shot information
        for label in self.shot_info_labels.values():
            label.setText('-')

    def play_selected_video(self, item):
        """Play the selected video"""
        if item is None:
            QMessageBox.warning(self, "Warning", "Please select a video to play")
            return

        # Use the correct videos folder path
        videos_folder = os.path.join(self.base_folder, self.current_player, f"session_{self.current_session:03d}", "videos")
        video_path = os.path.join(videos_folder, item.text())
        self.log_message(f"Playing video from: {video_path}")
        
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", f"Video file not found at: {video_path}")
            return

        try:
            # Close any existing video capture
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None

            # Disable live updates while playing video
            self.timer_enabled = False
            self.is_playing_video = True
            self.video_paused = False
            self.video_controls.setVisible(True)
            self.video_controls.play_pause_btn.setText("⏸")

            # Update shot information
            self.update_shot_info(item.text())

            # Open video
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap or not self.video_cap.isOpened():
                raise Exception("Could not open video file")

            # Get video properties
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames <= 0:
                raise Exception("Invalid video file - no frames found")

            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)

            # Playback loop
            while self.video_cap and self.video_cap.isOpened() and self.is_playing_video:
                if not self.video_paused:
                    ret, frame = self.video_cap.read()
                    if not ret:
                        break

                    self.current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.update_progress()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                        self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    
                    # Adjust delay based on playback speed
                    delay = int(30 / self.playback_speed)
                    cv2.waitKey(delay)
                else:
                    cv2.waitKey(30)
                QApplication.processEvents()

        except Exception as e:
            self.log_message(f"Error playing video: {e}")
            QMessageBox.critical(self, "Error", f"Could not play video: {str(e)}")
        finally:
            # Clean up
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            self.return_to_live()

    def delete_selected_video(self):
        """Delete the selected video file"""
        item = self.video_list.currentItem()
        if not item:
            QMessageBox.warning(self, "Warning", "Please select a video to delete")
            return
            
        # Use the correct videos folder path
        videos_folder = os.path.join(self.base_folder, self.current_player, f"session_{self.current_session:03d}", "videos")
        video_path = os.path.join(videos_folder, item.text())
        
        self.log_message(f"Attempting to delete video at: {video_path}")
        
        reply = QMessageBox.question(self, "Confirm Delete",
                                   f"Are you sure you want to delete {item.text()}?",
                                   QMessageBox.Yes | QMessageBox.No)
                                   
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    self.log_message(f"🗑 Successfully deleted video: {video_path}")
                    self.refresh_video_list()
                else:
                    self.log_message(f"Video file not found at: {video_path}")
                    QMessageBox.warning(self, "Error", f"Video file not found at:\n{video_path}")
            except Exception as e:
                self.log_message(f"Error deleting video: {e}")
                QMessageBox.critical(self, "Error", f"Could not delete video:\n{str(e)}")

    def toggle_pause(self):
        """Toggle video pause state"""
        self.video_paused = not self.video_paused
        self.video_controls.play_pause_btn.setText("▶" if self.video_paused else "⏸")

    def stop_playback(self):
        """Stop video playback and return to start"""
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.video_controls.progress_slider.setValue(0)
            self.video_paused = True
            self.video_controls.play_pause_btn.setText("▶")

    def seek_relative(self, frames):
        """Seek relative to current position"""
        if self.video_cap:
            new_frame = min(max(0, self.current_frame + frames), self.total_frames - 1)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.current_frame = new_frame
            self.update_progress()

    def seek_to_position(self, position):
        """Seek to specific position"""
        if self.video_cap:
            frame = int((position / 100.0) * self.total_frames)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.current_frame = frame

    def update_speed(self, value):
        """Update playback speed"""
        self.playback_speed = value / 100.0
        self.video_controls.speed_value.setText(f"{self.playback_speed:.1f}x")

    def update_progress(self):
        """Update progress bar and time label"""
        if self.video_cap:
            progress = (self.current_frame / self.total_frames) * 100
            self.video_controls.progress_slider.setValue(int(progress))
            
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            current_time = self.current_frame / fps
            total_time = self.total_frames / fps
            
            self.video_controls.time_label.setText(
                f"{int(current_time//60)}:{int(current_time%60):02d} / "
                f"{int(total_time//60)}:{int(total_time%60):02d}"
            )

    def update_shot_info(self, video_name):
        """Update shot information display from session data"""
        try:
            # Parse shot number and date from filename
            # Format: playername_session001_shot_1_20240424_194950.avi
            parts = video_name.replace('.avi', '').split('_')
            player_name = parts[0]
            session_num = parts[1].replace('session', '')  # Remove 'session' prefix
            shot_num = parts[3]

            # Construct the correct session folder path
            session_folder = os.path.join(self.base_folder, player_name, f"session_{int(session_num):03d}")
            session_data_file = os.path.join(session_folder, "session_data_shots.csv")

            self.log_message(f"Current video path: {os.path.join(self.shots_folder, video_name)}")
            self.log_message(f"Session folder path: {session_folder}")
            self.log_message(f"Looking for session data in: {session_data_file}")
            self.log_message(f"Does session folder exist? {os.path.exists(session_folder)}")
            if os.path.exists(session_folder):
                self.log_message(f"Contents of session folder: {os.listdir(session_folder)}")
            
            if os.path.exists(session_data_file):
                self.log_message(f"Found session data file at: {session_data_file}")
                with open(session_data_file, 'r') as f:
                    reader = csv.DictReader(f)
                    found = False
                    for row in reader:
                        if str(row['Shot_Number']) == str(shot_num):
                            found = True
                            self.log_message(f"Found shot data: {row}")
                            
                            # Update all stats to match the right panel format
                            self.shot_info_labels['shot_number'].setText(str(shot_num))
                            self.shot_info_labels['up_time'].setText(f"{row['Up_Time(s)']}s")
                            self.shot_info_labels['shot_count'].setText(str(row['Up_Time(s)']))
                            self.shot_info_labels['status'].setText(row['Status'].upper())
                            self.shot_info_labels['prep_time'].setText(f"{float(row['Preparing_Time(s)']):.1f}s")
                            self.shot_info_labels['aim_time'].setText(f"{float(row['Aiming_Time(s)']):.1f}s")
                            self.shot_info_labels['after_time'].setText(f"{float(row['After_Shot_Time(s)']):.1f}s")
                            self.shot_info_labels['position'].setText("READY ✓" if row['Shooting_Position'] == "Yes" else "NOT READY ✗")
                            
                            # Handle stability which might be in different formats
                            stability = row.get('Stability(%)', '0')
                            if not stability.endswith('%'):
                                stability = f"{stability}%"
                            self.shot_info_labels['stability'].setText(stability)
                            
                            # Style the status text
                            status_colors = {
                                'FREE': "#f8f9fa",      # Light
                                'PREPARING': "#ffc107",  # Warning
                                'AIMING': "#28a745",     # Success
                                'AFTER_SHOT': "#17a2b8"  # Info
                            }
                            status = row['Status'].upper()
                            if status in status_colors:
                                self.shot_info_labels['status'].setStyleSheet(f"color: {status_colors[status]}; font-weight: bold;")
                            
                            # Style the position text
                            position_color = "#28a745" if row['Shooting_Position'] == "Yes" else "#dc3545"
                            self.shot_info_labels['position'].setStyleSheet(f"color: {position_color}; font-weight: bold;")

                            # Style other values
                            for key in ['up_time', 'shot_count', 'prep_time', 'aim_time', 'after_time']:
                                self.shot_info_labels[key].setStyleSheet("color: #4a90e2; font-weight: bold;")
                            break
                    
                    if not found:
                        self.log_message(f"No data found for shot {shot_num} in {session_data_file}")
                        self._clear_shot_info(shot_num)
            else:
                self.log_message(f"Session file not found at path: {session_data_file}")
                self.log_message(f"Directory contents of {session_folder}:")
                if os.path.exists(session_folder):
                    self.log_message(os.listdir(session_folder))
                else:
                    self.log_message("Session folder does not exist!")
                self._clear_shot_info(shot_num)
                
        except Exception as e:
            self.log_message(f"Error updating shot info: {e}")
            import traceback
            traceback.print_exc()
            self._clear_shot_info(shot_num)

    def _clear_shot_info(self, shot_num):
        """Clear shot information display but keep the shot number"""
        self.shot_info_labels['shot_number'].setText(str(shot_num))
        for key in self.shot_info_labels:
            if key != 'shot_number':
                self.shot_info_labels[key].setText('-')
                self.shot_info_labels[key].setStyleSheet("color: #4a90e2; font-weight: bold;")

    def start_recording(self):
        """Start recording the shot"""
        if not self.recording:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Format: session_001
            session_name = f"session_{self.current_session:03d}"
            # Create the full path structure
            video_filename = os.path.join(
                self.base_folder,
                self.current_player,
                session_name,
                "videos",
                f'{self.current_player}_session{self.current_session:03d}_shot_{self.combo_counter}_{timestamp}.avi'
            )
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
            self.recording = True
            self.log_message(f"📹 Started recording shot {self.combo_counter} to {video_filename}")

    def stop_recording(self):
        """Stop recording the shot"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.log_message(f"✅ Finished recording shot {self.combo_counter}")
            self.refresh_video_list()  # Update the video list

    def update_stability_threshold(self, value):
        """Handle stability threshold input change"""
        self.stable_time_threshold = value
        self.log_message(f"Stability time threshold changed to: {value:.1f}s")

    def log_message(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        # Ensure the latest message is visible
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        # Also print to console for debugging
        print(message)

    def clear_log(self):
        """Clear the log message area"""
        self.log_text.clear()

    def toggle_test_mode(self):
        """Toggle test mode on/off"""
        self.test_mode = self.test_mode_btn.isChecked()
        
        # Update button and label
        if self.test_mode:
            self.test_mode_btn.setText("Disable Test Mode")
            self.test_mode_label.setText("Test mode is ON")
            self.test_mode_label.setStyleSheet("color: #28a745; font-weight: bold;")
            self.test_shot_counter = 0
            self.log_message("🎯 Test mode enabled - Values shown on screen")
            
            # Hide stats panel during test mode
            for label in self.stat_labels.values():
                label.setVisible(False)
        else:
            self.test_mode_btn.setText("Enable Test Mode")
            self.test_mode_label.setText("Test mode is OFF")
            self.test_mode_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            self.log_message("✅ Test mode disabled - Shot count will now be recorded")
            
            # Show stats panel again
            for label in self.stat_labels.values():
                label.setVisible(True)
            self.update_stats()

    def launch_dashboard(self):
        """Launch the Streamlit dashboard in a new process"""
        import subprocess
        import sys
        import os
        
        try:
            # Get the application path
            if getattr(sys, 'frozen', False):
                # If we're running as a bundled exe, use sys._MEIPASS
                application_path = sys._MEIPASS
            else:
                # If we're running in a normal Python environment
                application_path = os.path.dirname(os.path.abspath(__file__))
            
            dashboard_path = os.path.join(application_path, 'dashboard.py')
            
            # Create a batch file to run streamlit
            batch_content = f'''@echo off
            call {sys.executable} -m streamlit run "{dashboard_path}"
            '''
            
            batch_path = os.path.join(application_path, 'run_dashboard.bat')
            with open(batch_path, 'w') as f:
                f.write(batch_content)
            
            # Run the batch file
            subprocess.Popen(['cmd', '/c', batch_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch dashboard: {str(e)}")

    def toggle_camera(self):
        """Toggle camera state"""
        self.camera_enabled = not self.camera_enabled
        self.camera_btn.setChecked(self.camera_enabled)
        self.camera_label.setText("Camera is ON" if self.camera_enabled else "Camera is OFF")
        self.camera_label.setStyleSheet("color: #28a745; font-weight: bold;" if self.camera_enabled else "color: #dc3545; font-weight: bold;")

    def upload_baseline(self):
        """Handle baseline file upload"""
        try:
            # Get the most recent session folder
            session_folders = [f for f in os.listdir(self.player_folder) if f.startswith('session_')]
            if not session_folders:
                QMessageBox.warning(self, "Warning", "No session folders found!")
                return
                
            latest_session = sorted(session_folders)[-1]
            session_path = os.path.join(self.player_folder, latest_session)
            
            # Show file dialog to select baseline file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Baseline File",
                session_path,
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # Load the baseline data
                self.baseline_skeleton, self.baseline_angles = self.load_baseline_from_csv(file_path)
                if self.baseline_skeleton and self.last_landmarks:
                    hip_left = self.last_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                    hip_right = self.last_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                    self.baseline_anchor = ((hip_left.x + hip_right.x) / 2, (hip_left.y + hip_right.y) / 2)
                    self.log_message(f"📥 Baseline loaded from {file_path}")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to load baseline data from selected file!")
        except Exception as e:
            self.log_message(f"Error uploading baseline: {e}")
            QMessageBox.critical(self, "Error", f"Failed to upload baseline:\n{str(e)}")

    def set_baseline(self):
        """Set the current pose as baseline"""
        if self.last_landmarks:
            hip_left = self.last_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            hip_right = self.last_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            self.baseline_anchor = ((hip_left.x + hip_right.x) / 2, (hip_left.y + hip_right.y) / 2)
            self.baseline_skeleton = [(lm.x, lm.y) for lm in self.last_landmarks]
            self.baseline_angles = self.get_current_angles()
            self.log_message("📌 Baseline set from current pose")
        else:
            QMessageBox.warning(self, "Warning", "No pose detected! Please ensure you are visible to the camera.")

    def save_baseline(self):
        """Save the current baseline to CSV without applying it"""
        if self.baseline_skeleton and self.baseline_angles:
            self.save_baseline_to_csv(self.baseline_skeleton, self.baseline_angles)
        else:
            QMessageBox.warning(self, "Warning", "No baseline set! Please set a baseline first.")

    def clear_baseline(self):
        """Clear the current baseline"""
        self.baseline_skeleton = None
        self.baseline_anchor = None
        self.baseline_angles = {}
        self.log_message("❌ Baseline cleared")

    def start_skeleton_recording(self):
        """Start recording skeleton data"""
        if self.skeleton_recording:
            return
        
        self.skeleton_recording = True
        self.skeleton_frames = []
        self.skeleton_start_time = time.time()
        self.skeleton_frame_count = 0
        self.skeleton_last_frame_time = 0
        
        # Load ALL shot details from the CSV file for this session
        self.all_shot_details = []
        
        # First, ensure the session data file exists
        if not os.path.exists(self.session_data_file):
            self.log_message(f"Session data file does not exist: {self.session_data_file}")
            self.log_message("Creating session data file...")
            self.create_session_data_file()
        
        try:
            if os.path.exists(self.session_data_file):
                self.log_message(f"Loading shots from CSV file: {self.session_data_file}")
                with open(self.session_data_file, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV row to the format expected in JSON
                        shot_detail = {
                            "Player": row['Player'],
                            "Session": row['Session'],
                            "Shot_Number": row['Shot_Number'],
                            "Timestamp": row['Timestamp'],
                            "Up_Time(s)": row['Up_Time(s)'],
                            "Preparing_Time(s)": row['Preparing_Time(s)'],
                            "Aiming_Time(s)": row['Aiming_Time(s)'],
                            "After_Shot_Time(s)": row['After_Shot_Time(s)'],
                            "Complete_Shot": row['Complete_Shot']
                        }
                        self.all_shot_details.append(shot_detail)
                self.log_message(f"Loaded {len(self.all_shot_details)} previous shots from CSV")
                # Debug: Print each shot that was loaded
                for shot in self.all_shot_details:
                    self.log_message(f"  - Loaded Shot {shot['Shot_Number']}: {shot['Timestamp']}")
            else:
                self.log_message("No session data file found, starting with empty shot details")
        except Exception as e:
            self.log_message(f"Error loading previous shot details from CSV: {e}")
            self.all_shot_details = []
        
        # Also try to load from previous skeleton recordings if CSV is empty
        if not self.all_shot_details:
            skeleton_dir = os.path.join(self.session_folder, "skeletons")
            if os.path.exists(skeleton_dir):
                json_files = [f for f in os.listdir(skeleton_dir) if f.endswith('.json')]
                if json_files:
                    # Sort to get the most recent file
                    json_files.sort(reverse=True)
                    latest_file = os.path.join(skeleton_dir, json_files[0])
                    try:
                        with open(latest_file, 'r') as f:
                            data = json.load(f)
                            if "metadata" in data and "shot_details" in data["metadata"]:
                                self.all_shot_details = data["metadata"]["shot_details"]
                                self.log_message(f"Loaded {len(self.all_shot_details)} previous shot details from JSON")
                    except Exception as e:
                        self.log_message(f"Error loading previous shot details from JSON: {e}")
        
        # Update UI
        self.start_skeleton_btn.setEnabled(False)
        self.stop_skeleton_btn.setEnabled(True)
        self.skeleton_status_label.setText("Recording in progress...")
        self.skeleton_status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Create metadata
        self.skeleton_metadata = {
            "player": self.current_player,
            "session": self.current_session,
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "fps": self.skeleton_fps,
            "start_timestamp": self.skeleton_start_time
        }
        
        self.log_message("🦴 Skeleton recording started")

    def stop_skeleton_recording(self):
        """Stop recording skeleton data and save to file"""
        if not self.skeleton_recording:
            return
    
        self.skeleton_recording = False
    
        # Update metadata with end time
        self.skeleton_metadata["end_timestamp"] = time.time()
        self.skeleton_metadata["duration"] = self.skeleton_metadata["end_timestamp"] - self.skeleton_metadata["start_timestamp"]
        self.skeleton_metadata["frame_count"] = self.skeleton_frame_count
    
        # Add shot number and phases for current recording
        self.skeleton_metadata["shot_number"] = self.combo_counter
        self.skeleton_metadata["phases"] = {
            "before_shot": self.preparing_duration,
            "within_shot": self.aiming_duration,
            "after_shot": self.after_shot_duration,
            "total_time": self.preparing_duration + self.aiming_duration + self.after_shot_duration
        }
    
        # Add current shot details to the accumulated list (only if not already present)
        current_shot_detail = {
            "Player": self.current_player,
            "Session": f"Session {self.current_session:03d}",
            "Shot_Number": str(self.combo_counter),
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Up_Time(s)": str(self.up_time_seconds),
            "Preparing_Time(s)": f"{self.preparing_duration:.1f}",
            "Aiming_Time(s)": f"{self.aiming_duration:.1f}",
            "After_Shot_Time(s)": f"{self.after_shot_duration:.1f}",
            "Complete_Shot": "Yes" if self.preparing_duration and self.aiming_duration and self.after_shot_duration else "No"
        }
        
        # Check if this shot is already in the list (to avoid duplicates)
        shot_exists = any(shot["Shot_Number"] == str(self.combo_counter) for shot in self.all_shot_details)
        if not shot_exists:
            self.all_shot_details.append(current_shot_detail)
            self.log_message(f"Added shot {self.combo_counter} to accumulated shot details")
        else:
            self.log_message(f"Shot {self.combo_counter} already exists in accumulated details")
        
        # Debug: Print all shots that will be saved
        self.log_message(f"Total shots to be saved: {len(self.all_shot_details)}")
        for shot in self.all_shot_details:
            self.log_message(f"  - Shot {shot['Shot_Number']}: {shot['Timestamp']}")
    
        # Update UI
        self.start_skeleton_btn.setEnabled(True)
        self.stop_skeleton_btn.setEnabled(False)
        self.skeleton_status_label.setText(f"Recording saved: {self.skeleton_frame_count} frames")
        self.skeleton_status_label.setStyleSheet("color: green; font-style: italic;")
    
        # Save the recording
        if self.skeleton_frame_count > 0:
            self.save_skeleton_recording()
        else:
            self.log_message("No skeleton frames recorded")

    def capture_skeleton_frame(self, landmarks):
        """Add a frame to the skeleton recording if enough time has passed"""
        if not self.skeleton_recording or not landmarks:
            return
            
        current_time = time.time()
        # frame_interval = 1.0 / self.skeleton_fps # This is no longer strictly needed for capture logic
        
        # Only record at specified interval to avoid huge files
        # if current_time - self.skeleton_last_frame_time < frame_interval:
           # return
            
        # Calculate joint angles 
        angles = self.get_current_angles()
        
        # Calculate CoG if possible
        cog_position = None
        try:
            cog = self.calculate_cog_position(landmarks)
            if cog:
                cog_position = cog
        except:
            pass
        
        # Create frame data
        frame_data = {
            "timestamp": current_time,
            "relative_time": current_time - self.skeleton_start_time,
            "phase": self.current_status,
            "landmarks": self.serialize_landmarks(landmarks),
            "angles": angles,
            "shot_number": self.combo_counter  # <-- Add this line
        }
        
        if cog_position:
            frame_data["cog"] = cog_position
            
        self.skeleton_frames.append(frame_data)
        self.skeleton_frame_count += 1
        self.skeleton_last_frame_time = current_time
        
        # Update status occasionally
        if self.skeleton_frame_count % 30 == 0:
            self.skeleton_status_label.setText(f"Recording: {self.skeleton_frame_count} frames")
            
        # Log milestone frames
        if self.skeleton_frame_count % 150 == 0:
            seconds = round(current_time - self.skeleton_start_time)
            self.log_message(f"🦴 Skeleton recording: {self.skeleton_frame_count} frames ({seconds}s)")

    def calculate_cog_position(self, landmarks):
        """Calculate CoG position from landmarks"""
        try:
            # (start,end), mass fraction, CoG fraction along that segment
            segments = [
                ((0, 1),   0.081, 0.50),  # head→neck
                ((11,23),  0.432, 0.43),  # shoulders→hips
                ((23,25),  0.105, 0.36),  # left thigh
                ((24,26),  0.105, 0.36),  # right thigh
                ((25,27),  0.046, 0.42),  # left shank
                ((26,28),  0.046, 0.42),  # right shank
                ((27,31),  0.015, 0.50),  # left foot
                ((28,32),  0.015, 0.50),  # right foot
                ((11,13),  0.027, 0.436), # left upper arm
                ((12,14),  0.027, 0.436), # right upper arm
                ((13,15),  0.023, 0.43),  # left forearm+hand
                ((14,16),  0.023, 0.43),  # right forearm+hand
            ]

            # Check that all indices exist in the landmarks
            for (a, b), _, _ in segments:
                if a >= len(landmarks) or b >= len(landmarks):
                    return None

            total_x = total_y = total_mass = 0.0
            for (a, b), mass_frac, cog_frac in segments:
                p1, p2 = landmarks[a], landmarks[b]
                
                # Skip if visibility is too low
                if p1.visibility < 0.5 or p2.visibility < 0.5:
                    continue
                    
                cog_x = p1.x + cog_frac * (p2.x - p1.x)
                cog_y = p1.y + cog_frac * (p2.y - p1.y)
                total_x += cog_x * mass_frac
                total_y += cog_y * mass_frac
                total_mass += mass_frac

            if total_mass <= 0:
                return None

            return {
                "x": total_x / total_mass,
                "y": total_y / total_mass
            }
        except:
            return None

    def serialize_landmarks(self, landmarks):
        """Convert MediaPipe landmarks to serializable format"""
        serialized = []
        for i, landmark in enumerate(landmarks):
            serialized.append({
                "index": i,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z if hasattr(landmark, 'z') else 0,
                "visibility": landmark.visibility
            })
        return serialized

    def save_skeleton_recording(self):
        """Save skeleton recording data to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(
                self.session_folder, 
                f"{self.current_player}_session{self.current_session:03d}_skeleton_{timestamp}.json"
            )
            
            # Update metadata with all accumulated shot details
            self.skeleton_metadata["shot_details"] = self.all_shot_details
            
            # Debug print to verify shot details
            self.log_message(f"Saving {len(self.all_shot_details)} shot details to JSON")
            for shot in self.all_shot_details:
                self.log_message(f"  - Shot {shot['Shot_Number']}: {shot['Timestamp']}")
            
            # Verify the data structure before saving
            data = {
                "metadata": self.skeleton_metadata,
                "frames": self.skeleton_frames
            }
            
            # Debug: Print the metadata structure
            self.log_message(f"Metadata structure:")
            self.log_message(f"  - Player: {self.skeleton_metadata.get('player', 'Unknown')}")
            self.log_message(f"  - Session: {self.skeleton_metadata.get('session', 'Unknown')}")
            self.log_message(f"  - Shot details count: {len(self.skeleton_metadata.get('shot_details', []))}")
            
            skeleton_dir = os.path.join(self.session_folder, "skeletons")
            if not os.path.exists(skeleton_dir):
                os.makedirs(skeleton_dir)
                
            skeleton_file = os.path.join(skeleton_dir, os.path.basename(filename))
            with open(skeleton_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.log_message(f"🦴 Skeleton recording saved to: {skeleton_file}")
            self.log_message(f"📊 JSON file contains {len(self.all_shot_details)} shot details")
            return skeleton_file
        except Exception as e:
            self.log_message(f"Error saving skeleton data: {str(e)}")
            return None

    def view_skeleton_recordings(self):
        """Show dialog to select and view skeleton recordings"""
        try:
            # Create skeletons directory if it doesn't exist
            skeleton_dir = os.path.join(self.session_folder, "skeletons")
            if not os.path.exists(skeleton_dir):
                os.makedirs(skeleton_dir)
                
            # Check if there are any recordings
            files = [f for f in os.listdir(skeleton_dir) if f.endswith('.json')]
            if not files:
                QMessageBox.information(self, "No Recordings", 
                                       "No skeleton recordings found for this session.")
                return
                
            # Let user select a file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Skeleton Recording", 
                skeleton_dir, 
                "Skeleton Files (*.json)"
            )
            
            if file_path:
                # Open the playback dialog
                dialog = SkeletonPlaybackDialog(self, file_path)
                dialog.exec_()
        except Exception as e:
            self.log_message(f"Error loading skeleton recordings: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open skeleton recordings: {str(e)}")

    def reload_csv_data(self):
        """Manually reload CSV data for debugging"""
        try:
            if os.path.exists(self.session_data_file):
                self.log_message(f"Manually reloading CSV data from: {self.session_data_file}")
                with open(self.session_data_file, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.all_shot_details = []
                    for row in reader:
                        shot_detail = {
                            "Player": row['Player'],
                            "Session": row['Session'],
                            "Shot_Number": row['Shot_Number'],
                            "Timestamp": row['Timestamp'],
                            "Up_Time(s)": row['Up_Time(s)'],
                            "Preparing_Time(s)": row['Preparing_Time(s)'],
                            "Aiming_Time(s)": row['Aiming_Time(s)'],
                            "After_Shot_Time(s)": row['After_Shot_Time(s)'],
                            "Complete_Shot": row['Complete_Shot']
                        }
                        self.all_shot_details.append(shot_detail)
                self.log_message(f"Reloaded {len(self.all_shot_details)} shots from CSV")
                for shot in self.all_shot_details:
                    self.log_message(f"  - Shot {shot['Shot_Number']}: {shot['Timestamp']}")
            else:
                self.log_message("CSV file does not exist for reloading")
        except Exception as e:
            self.log_message(f"Error reloading CSV data: {e}")

class SessionManagementDialog(QDialog):
    def __init__(self, parent=None, base_folder=None, current_player=None):
        super().__init__(parent)
        self.base_folder = base_folder
        self.current_player = current_player
        self.setWindowTitle("Session Management")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)

        # Session List
        session_group = QGroupBox("Sessions")
        session_layout = QVBoxLayout(session_group)
        
        # Add refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_sessions)
        session_layout.addWidget(refresh_btn)
        
        # Session list widget
        self.session_list = QListWidget()
        self.session_list.itemDoubleClicked.connect(self.edit_session)
        session_layout.addWidget(self.session_list)
        
        # Buttons for session management
        btn_layout = QHBoxLayout()
        self.new_btn = QPushButton("➕ New Session")
        self.edit_btn = QPushButton("✏️ Edit Session")
        self.delete_btn = QPushButton("🗑 Delete Session")
        self.switch_btn = QPushButton("🔄 Switch to Session")
        
        self.new_btn.clicked.connect(self.new_session)
        self.edit_btn.clicked.connect(self.edit_session)
        self.delete_btn.clicked.connect(self.delete_session)
        self.switch_btn.clicked.connect(self.switch_session)
        
        btn_layout.addWidget(self.new_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.switch_btn)
        
        session_layout.addLayout(btn_layout)
        
        # Add all widgets to main layout
        layout.addWidget(session_group)
        
        # Connect signals
        self.session_list.itemSelectionChanged.connect(self.update_buttons)
        
        # Style
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
            }
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QListWidget {
                background-color: #444444;
                color: white;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        
        # Initial refresh
        self.refresh_sessions()
        self.update_buttons()
    
    def refresh_sessions(self):
        """Refresh the list of sessions"""
        self.session_list.clear()
        if not self.current_player:
            return
            
        player_folder = os.path.join(self.base_folder, self.current_player)
        if not os.path.exists(player_folder):
            return
            
        sessions = [f for f in os.listdir(player_folder) if f.startswith('session_')]
        sessions.sort()  # Sort by session number
        
        for session in sessions:
            session_path = os.path.join(player_folder, session)
            info_file = os.path.join(session_path, "session_info.txt")
            
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    lines = f.readlines()
                    session_num = session.split('_')[1]
                    date = lines[2].split(': ')[1].strip()
                    notes = lines[3].split(': ')[1].strip() if len(lines) > 3 else ""
                    item = QListWidgetItem(f"Session {session_num} - {date}")
                    if notes:
                        item.setToolTip(f"Notes: {notes}")
                    self.session_list.addItem(item)
    
    def update_buttons(self):
        """Update button states based on selection"""
        has_selection = bool(self.session_list.selectedItems())
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.switch_btn.setEnabled(has_selection)
    
    def new_session(self):
        """Create a new session"""
        dialog = PlayerDialog(self)
        dialog.setWindowTitle("New Session")
        dialog.player_combo.setCurrentText(self.current_player)
        dialog.player_combo.setEnabled(False)  # Disable player selection
        
        # Find the next available session number
        player_folder = os.path.join(self.base_folder, self.current_player)
        if os.path.exists(player_folder):
            sessions = [int(f.split('_')[1]) for f in os.listdir(player_folder) if f.startswith('session_')]
            next_session = max(sessions) + 1 if sessions else 1
            dialog.session_spin.setValue(next_session)
        
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_session_info()
            self.current_player = info['player']
            self.current_session = info['session']
            self.session_notes = info['notes']
            
            # Create directory structure
            self.parent().setup_session_directories()
            
            # Save session info
            self.parent().save_session_info()
            
            # Update window title
            self.parent().setWindowTitle(f"AEYE SHOOTING Pro - {self.current_player} - Session {self.current_session}")
            
            # Refresh list
            self.refresh_sessions()
    
    def edit_session(self):
        """Edit selected session"""
        selected = self.session_list.selectedItems()
        if not selected:
            return
            
        item = selected[0]
        session_num = int(item.text().split()[1])
        
        dialog = PlayerDialog(self)
        dialog.setWindowTitle("Edit Session")
        dialog.player_combo.setCurrentText(self.current_player)
        dialog.player_combo.setEnabled(False)  # Disable player selection
        dialog.session_spin.setValue(session_num)
        
        # Load existing notes
        session_folder = os.path.join(self.base_folder, self.current_player, f"session_{session_num:03d}")
        info_file = os.path.join(session_folder, "session_info.txt")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 3:
                    notes = lines[3].split(': ')[1].strip()
                    dialog.notes_text.setPlainText(notes)
        
        if dialog.exec_() == QDialog.Accepted:
            info = dialog.get_session_info()
            self.session_notes = info['notes']
            
            # Update session info
            self.parent().save_session_info()
            
            # Refresh list
            self.refresh_sessions()
    
    def delete_session(self):
        """Delete selected session"""
        selected = self.session_list.selectedItems()
        if not selected:
            return
            
        item = selected[0]
        session_num = int(item.text().split()[1])
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete Session {session_num}?\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            session_folder = os.path.join(self.base_folder, self.current_player, f"session_{session_num:03d}")
            try:
                import shutil
                shutil.rmtree(session_folder)
                self.refresh_sessions()
                QMessageBox.information(self, "Success", f"Session {session_num} deleted successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete session: {str(e)}")
    
    def switch_session(self):
        """Switch to selected session"""
        selected = self.session_list.selectedItems()
        if not selected:
            return
            
        item = selected[0]
        session_num = int(item.text().split()[1])
        
        self.current_session = session_num
        self.parent().current_session = session_num
        
        # Update session folder and data file paths
        self.parent().setup_session_directories()
        
        # Update window title
        self.parent().setWindowTitle(f"AEYE SHOOTING Pro - {self.current_player} - Session {self.current_session}")
        
        # Accept dialog
        self.accept()

class SkeletonPlaybackDialog(QDialog):
    def __init__(self, parent=None, skeleton_file=None):
        super().__init__(parent)
        self.parent = parent
        self.skeleton_file = skeleton_file
        self.skeleton_data = None
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        
        self.setWindowTitle("Skeleton Playback")
        self.setMinimumSize(800, 600)
        
        # Load skeleton data
        if not self.load_skeleton_data():
            return
            
        self.init_ui()
        
    def load_skeleton_data(self):
        """Load skeleton data from file"""
        try:
            with open(self.skeleton_file, 'r') as f:
                self.skeleton_data = json.load(f)
                
            if not self.skeleton_data or "frames" not in self.skeleton_data:
                QMessageBox.critical(self.parent, "Error", "Invalid skeleton data file.")
                return False
                
            return True
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to load skeleton data: {str(e)}")
            return False
            
    # Modify the init_ui method in SkeletonPlaybackDialog
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Skeleton display area - use our custom widget instead of QLabel
        self.display = DrawableDisplayWidget(self)
        self.display.setStyleSheet("background-color: black;")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.original_width = 800
        self.display.original_height = 600
        self.display.drawing_enabled = False
        layout.addWidget(self.display, 3)
        
        # Info panel
        info_layout = QHBoxLayout()
        
        # Player/session info
        metadata = self.skeleton_data.get("metadata", {})
        info_text = f"Player: {metadata.get('player', 'Unknown')} | "
        info_text += f"Session: {metadata.get('session', 'Unknown')} | "
        info_text += f"Date: {metadata.get('date', 'Unknown')} | "
        info_text += f"Frames: {len(self.skeleton_data.get("frames", []))}"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        info_layout.addWidget(info_label)
        
        # Current phase info
        self.phase_label = QLabel("Phase: N/A")
        self.phase_label.setStyleSheet("background-color: #444; color: yellow; padding: 5px; font-weight: bold;")
        info_layout.addWidget(self.phase_label)
        
        layout.addLayout(info_layout)
        
        # Drawing tools section
        drawing_group = QGroupBox("Drawing Tools")
        drawing_layout = QHBoxLayout(drawing_group)
        
        # Drawing enable checkbox
        self.drawing_checkbox = QCheckBox("Enable Drawing")
        self.drawing_checkbox.setChecked(False)
        self.drawing_checkbox.toggled.connect(self.toggle_drawing_mode)
        drawing_layout.addWidget(self.drawing_checkbox)
        
        # Shape type selection
        shape_label = QLabel("Shape:")
        drawing_layout.addWidget(shape_label)
        
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Line", "Rectangle", "Circle"])
        self.shape_combo.currentTextChanged.connect(self.change_shape_type)
        drawing_layout.addWidget(self.shape_combo)
        
        # Color selection
        color_label = QLabel("Color:")
        drawing_layout.addWidget(color_label)
        
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Green", "Red", "Blue", "Yellow", "White"])
        self.color_combo.setStyleSheet("""
            QComboBox { color: white; background-color: #444; }
            QComboBox::item:selected { background-color: #4a90e2; }
        """)
        self.color_combo.currentTextChanged.connect(self.change_shape_color)
        drawing_layout.addWidget(self.color_combo)
        
        # Clear drawings button
        self.clear_btn = QPushButton("Clear Drawings")
        self.clear_btn.clicked.connect(self.clear_drawings)
        drawing_layout.addWidget(self.clear_btn)
        
        layout.addWidget(drawing_group)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        self.reset_btn = QPushButton("⏮️ Reset")
        self.reset_btn.clicked.connect(self.reset_playback)
        controls_layout.addWidget(self.reset_btn)
        
        speed_label = QLabel("Speed:")
        controls_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(25, 200)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        controls_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        controls_layout.addWidget(self.speed_label)
        
        self.overlay_checkbox = QCheckBox("Show joint angles")
        self.overlay_checkbox.setChecked(True)
        controls_layout.addWidget(self.overlay_checkbox)
        
        layout.addLayout(controls_layout)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame: 0/0")
        slider_layout.addWidget(self.frame_label)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.skeleton_data.get("frames", [])) - 1)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        slider_layout.addWidget(self.frame_slider)
        
        self.time_label = QLabel("Time: 0.0s")
        slider_layout.addWidget(self.time_label)
        
        layout.addLayout(slider_layout)
        
        # Set style
        self.setStyleSheet("""
            QDialog {
                background-color: #222;
                color: white;
            }
            QLabel, QCheckBox {
                color: white;
            }
            QPushButton {
                background-color: #444;
                color: white;
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #333;
                margin: 2px 0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #4a90e2;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QGroupBox {
                color: white;
                border: 1px solid #444;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        
        # Draw the first frame
        self.draw_frame(0)
        
    def toggle_play(self):
        """Toggle play/pause state"""
        self.playing = not self.playing
        self.play_btn.setText("⏸️ Pause" if self.playing else "▶️ Play")
        
        if self.playing:
            self.timer.start(int(1000 / (self.skeleton_data.get("metadata", {}).get("fps", 15) * self.playback_speed)))
        else:
            self.timer.stop()
            
    def reset_playback(self):
        """Reset playback to the beginning"""
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.draw_frame(0)
        
    def update_speed(self, value):
        """Update playback speed based on slider"""
        self.playback_speed = value / 100.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
        
        if self.playing:
            self.timer.stop()
            self.timer.start(int(1000 / (self.skeleton_data.get("metadata", {}).get("fps", 15) * self.playback_speed)))
            
    def slider_changed(self, value):
        """Handle manual slider change"""
        if not self.playing:
            self.current_frame = value
            self.draw_frame(value)
            
    def update_frame(self):
        """Update to the next frame during playback"""
        frames = self.skeleton_data.get("frames", [])
        if not frames:
            self.timer.stop()
            return
            
        self.current_frame += 1
        if self.current_frame >= len(frames):
            self.current_frame = 0
            
        self.frame_slider.setValue(self.current_frame)
        self.draw_frame(self.current_frame)
        
    def draw_frame(self, frame_index):
        """Draw the specified skeleton frame"""
        frames = self.skeleton_data.get("frames", [])
        if not frames or frame_index >= len(frames):
            return
            
        frame = frames[frame_index]
        
        # Create a black image for drawing
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Extract landmarks from frame
        landmarks = frame.get("landmarks", [])
        
        # Update info labels
        self.frame_label.setText(f"Frame: {frame_index+1}/{len(frames)}")
        self.time_label.setText(f"Time: {frame.get('relative_time', 0):.1f}s")
        
        # Update phase label
        phase = frame.get("phase", "unknown")
        self.phase_label.setText(f"Phase: {phase.upper()}")
        
        # Set phase label color based on phase
        phase_colors = {
            "free": "white",
            "preparing": "yellow",
            "aiming": "lime",
            "after_shot": "orange"
        }
        self.phase_label.setStyleSheet(f"background-color: #444; color: {phase_colors.get(phase, 'white')}; padding: 5px; font-weight: bold;")
        
        # Draw the skeleton
        self.draw_skeleton(image, landmarks, phase)
        
        # Draw CoG if available
        if "cog" in frame:
            cog = frame["cog"]
            cx = int(cog["x"] * 800)
            cy = int(cog["y"] * 600)
            cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)
            cv2.line(image, (cx, 0), (cx, 600), (0, 0, 255), 1)
            cv2.line(image, (0, cy), (800, cy), (0, 0, 255), 1)
        
        # Apply user-drawn shapes to the image - THIS LINE IS CRITICAL
        image = self.display.draw_shapes(image)
        
        # Convert to QImage and display
        h, w, c = image.shape
        qimg = QImage(image.data, w, h, w*c, QImage.Format_RGB888)
        self.display.setPixmap(QPixmap.fromImage(qimg))
        
    def draw_skeleton(self, image, landmarks, phase):
        """Draw skeleton with connections"""
        if not landmarks:
            return
            
        # Define colors based on phase
        color_map = {
            "free": (200, 200, 200),
            "preparing": (0, 255, 255),
            "aiming": (0, 255, 0),
            "after_shot": (255, 165, 0)
        }
        color = color_map.get(phase, (200, 200, 200))
        
        # Define connections (similar to MediaPipe POSE_CONNECTIONS)
        connections = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11),
            # Right arm
            (12, 14), (14, 16),
            # Left arm
            (11, 13), (13, 15),
            # Right leg
            (24, 26), (26, 28), (28, 32), (32, 30), (30, 28),
            # Left leg
            (23, 25), (25, 27), (27, 31), (31, 29), (29, 27),
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Connect face to body
            (0, 1), (1, 11), (1, 12)
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
                
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            # Skip if visibility is too low
            if start.get("visibility", 0) < 0.5 or end.get("visibility", 0) < 0.5:
                continue
                
            start_point = (int(start["x"] * 800), int(start["y"] * 600))
            end_point = (int(end["x"] * 800), int(end["y"] * 600))
            
            cv2.line(image, start_point, end_point, color, 2)
        
        # Draw joints
        for i, lm in enumerate(landmarks):
            if lm.get("visibility", 0) < 0.5:
                continue
                
            x, y = int(lm["x"] * 800), int(lm["y"] * 600)
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
            
            # Draw joint index if the overlay is checked
            if self.overlay_checkbox.isChecked():
                cv2.putText(image, str(i), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def closeEvent(self, event):
        """Handle dialog close"""
        self.timer.stop()
        event.accept()
    def toggle_drawing_mode(self, enabled):
        """Toggle drawing mode on/off"""
        self.display.drawing_enabled = enabled
        print(f"Drawing mode: {enabled}")  # Debug print
        if enabled:
            self.playing = False
            self.timer.stop()
            self.play_btn.setText("▶️ Play")

    def change_shape_type(self, shape_type):
        """Change the current shape type for drawing"""
        self.display.shape_type = shape_type.lower()

    def change_shape_color(self, color_name):
        """Change the current drawing color"""
        color_map = {
            "Green": (0, 255, 0),
            "Red": (0, 0, 255),  # OpenCV uses BGR
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255)
        }
        self.display.shape_color = color_map.get(color_name, (0, 255, 0))

    def clear_drawings(self):
        """Clear all drawn shapes"""
        self.display.clear_shapes()

    def redraw_current_frame(self):
        """Redraw the current frame with updated shapes"""
        self.draw_frame(self.current_frame)

  

class DrawableDisplayWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shapes = []
        self.current_shape = None
        self.drawing = False
        self.shape_type = "line"  # Default shape type
        self.shape_color = (0, 255, 0)  # Default color (green)
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        if not hasattr(self, 'drawing_enabled') or not self.drawing_enabled:
            return super().mousePressEvent(event)
            
        self.drawing = True
        pos = event.pos()
        x, y = pos.x(), pos.y()
        
        # Scale coordinates to image dimensions
        if self.pixmap():
            x = x * self.original_width / self.pixmap().width()
            y = y * self.original_height / self.pixmap().height()
        
        if self.shape_type == "line":
            self.current_shape = {"type": "line", "start": (x, y), "end": (x, y), "color": self.shape_color}
        elif self.shape_type == "rectangle":
            self.current_shape = {"type": "rectangle", "start": (x, y), "end": (x, y), "color": self.shape_color}
        elif self.shape_type == "circle":
            self.current_shape = {"type": "circle", "center": (x, y), "radius": 0, "color": self.shape_color}
        
    def mouseMoveEvent(self, event):
        if not self.drawing or not self.current_shape:
            return super().mouseMoveEvent(event)
            
        pos = event.pos()
        x, y = pos.x(), pos.y()
        
        # Scale coordinates to image dimensions
        if self.pixmap():
            x = x * self.original_width / self.pixmap().width()
            y = y * self.original_height / self.pixmap().height()
        
        if self.shape_type == "line" or self.shape_type == "rectangle":
            self.current_shape["end"] = (x, y)
        elif self.shape_type == "circle":
            dx = x - self.current_shape["center"][0]
            dy = y - self.current_shape["center"][1]
            self.current_shape["radius"] = int(np.sqrt(dx*dx + dy*dy))
        
        # Redraw with the updated shape
        self.parent().redraw_current_frame()
        
    def mouseReleaseEvent(self, event):
        if not self.drawing or not self.current_shape:
            return super().mouseReleaseEvent(event)
            
        self.drawing = False
        if self.current_shape:
            self.shapes.append(self.current_shape.copy())
            self.current_shape = None
            
        # Redraw after finalizing the shape
        self.parent().redraw_current_frame()
            
    def clear_shapes(self):
        self.shapes = []
        self.current_shape = None
        self.parent().redraw_current_frame()
        
    def draw_shapes(self, image):
        # Draw all saved shapes
        for shape in self.shapes:
            self._draw_shape(image, shape)
            
        # Draw the shape currently being drawn
        if self.drawing and self.current_shape:
            self._draw_shape(image, self.current_shape)
            
        return image
        
    def _draw_shape(self, image, shape):
        if shape["type"] == "line":
            start_point = (int(shape["start"][0]), int(shape["start"][1]))
            end_point = (int(shape["end"][0]), int(shape["end"][1]))
            cv2.line(image, start_point, end_point, shape["color"], 2)
            
        elif shape["type"] == "rectangle":
            start_point = (int(shape["start"][0]), int(shape["start"][1]))
            end_point = (int(shape["end"][0]), int(shape["end"][1]))
            cv2.rectangle(image, start_point, end_point, shape["color"], 2)
            
        elif shape["type"] == "circle":
            center = (int(shape["center"][0]), int(shape["center"][1]))
            radius = int(shape["radius"])
            cv2.circle(image, center, radius, shape["color"], 2)

    

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    window = ArcheryAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())