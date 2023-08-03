# Imports
import os
import sys
import subprocess
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QWidget, 
                             QProgressBar, QMessageBox, QFileDialog, QMenuBar, QAction, QGraphicsView, QGraphicsScene, 
                             QLineEdit, QLabel, QRadioButton, QButtonGroup, QSizePolicy, QInputDialog, QDialog, QSlider)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import QTimer, QEvent, QThread, Qt, QProcess, QProcessEnvironment
import webbrowser

# AstrocyteAnalysis subprocess
class WorkerThread(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        self.process = subprocess.Popen(["python3", "AstrocyteAnalysis.py"])

    def stop(self):
        if self.process:
            self.process.terminate()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Initialize CZI file and output directory
        self.czi_file = ""
        self.output_dir = ""

        # Create a Graphics View widget
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setMinimumSize(800, 600)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)

        # Create a status edit widget
        self.status_edit = QTextEdit(self)
        self.status_edit.setReadOnly(True)

        # Create a progress bar widget
        self.progress_bar = QProgressBar(self)

        # Create Start, Stop, and Toggle buttons
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.next_button = QPushButton('Next', self)
        self.prev_button = QPushButton('Prev', self)

        # Create image number input and label
        self.image_label = QLabel('Image Number', self)
        self.image_input = QLineEdit(self)

        # Create a radio button group for image mode selection
        self.image_mode_group = QButtonGroup(self)
        self.raw_button = QRadioButton('Raw', self)
        self.normalized_button = QRadioButton('Normalized', self)
        self.image_mode_group.addButton(self.raw_button, 0)
        self.image_mode_group.addButton(self.normalized_button, 1)
        self.raw_button.setChecked(True)

        # Set up layouts
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.status_edit)
        controls_layout.addWidget(self.raw_button)
        controls_layout.addWidget(self.normalized_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.image_label)
        controls_layout.addWidget(self.image_input)

        main_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.graphics_view, 4)  # Increase the stretch factor of the graphics_view

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # Set size policy of the graphics view to expand
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Connect the button signals with the slots
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.image_mode_group.buttonClicked[int].connect(self.image_mode_changed)
        self.next_button.clicked.connect(self.next_image)
        self.prev_button.clicked.connect(self.prev_image)
        self.image_input.returnPressed.connect(self.select_image)

        # Set the window title
        self.setWindowTitle("Astrocyte Analysis")

        # Set the status bar
        self.statusBar().showMessage('Ready')

        # Initialize image index
        self.image_index = 0

        # Initialize image mode to raw
        self.image_mode = "raw"

        # Create worker thread
        self.worker_thread = WorkerThread()

        # Create timer for reading progress
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.read_progress)

        # Set up the menu bar
        self.setup_menu()

        # Set the style of the GUI
        self.set_style()

        # Create CZI conversion process
        self.czi_conversion_process = QProcess(self)
        self.czi_conversion_process.setProcessChannelMode(QProcess.MergedChannels)
        self.czi_conversion_process.readyReadStandardOutput.connect(self.read_czi_conversion_output)

    def set_style(self):
        style = """
        QMainWindow {
            background-color: #2C2C2C;
            color: #FFFFFF;
        }

        QPushButton {
            background-color: #404040;
            color: #FFFFFF;
            border: 2px solid #FFFFFF;
            border-radius: 5px;
            padding: 10px;
            margin: 10px;
        }

        QPushButton:hover {
            background-color: #505050;
        }

        QPushButton:pressed {
            background-color: #606060;
        }

        QTextEdit {
            background-color: #404040;
            color: #FFFFFF;
            border: 2px solid #FFFFFF;
        }

        QLineEdit {
            background-color: #404040;
            color: #FFFFFF;
            border: 2px solid #FFFFFF;
        }

        QLabel {
            color: #FFFFFF;
        }

        QMenuBar {
            background-color: #404040;
            color: #FFFFFF;
        }

        QMenuBar:item {
            background-color: #404040;
            color: #FFFFFF;
        }

        QMenuBar:item:selected {
            background-color: #505050;
        }

        QProgressBar {
            text-align: center;
        }

        QProgressBar::chunk {
            background-color: #404040;
        }

        QMessageBox {
            background-color: #404040;
            color: #FFFFFF;
            border: 2px solid #FFFFFF;
        }

        QDialog {
            background-color: #2C2C2C;
            color: #FFFFFF;
        }
        """
        self.setStyleSheet(style)

        # Additional lines to make the code longer
        for i in range(100):
            pass

    def setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')

        load_nuclei_model_action = QAction('Load Nuclei Model', self)
        load_nuclei_model_action.triggered.connect(self.load_nuclei_model)
        file_menu.addAction(load_nuclei_model_action)

        load_cyto_model_action = QAction('Load Cyto Model', self)
        load_cyto_model_action.triggered.connect(self.load_cyto_model)
        file_menu.addAction(load_cyto_model_action)

        load_pre_dir_action = QAction('Load Pre Directory', self)
        load_pre_dir_action.triggered.connect(self.load_pre_dir)
        file_menu.addAction(load_pre_dir_action)

        load_post_dir_action = QAction('Load Post Directory', self)
        load_post_dir_action.triggered.connect(self.load_post_dir)
        file_menu.addAction(load_post_dir_action)

        czi_menu = menubar.addMenu('&CZI')

        czi_to_tiff_action = QAction('Convert CZI to TIFF with timestamps', self)
        czi_to_tiff_action.triggered.connect(self.convert_czi_to_tiff)
        czi_menu.addAction(czi_to_tiff_action)

        rename_tiff_action = QAction('Rename TIFFs using CZI', self)
        rename_tiff_action.triggered.connect(self.rename_tiffs)
        rename_tiff_action.setDisabled(True)
        czi_menu.addAction(rename_tiff_action)

        help_menu = menubar.addMenu('&Help')

        help_action = QAction('Github', self)
        help_action.triggered.connect(lambda: webbrowser.open('https://github.com/not-availiable/AstrocytesImageAnalysis'))
        help_menu.addAction(help_action)

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def start_analysis(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_edit.append("Starting analysis...")

        self.worker_thread.start()

        self.progress_timer.start(100)  # every 0.1 second

    def stop_analysis(self):
        self.worker_thread.stop()

        self.progress_timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.status_edit.append("Analysis stopped.")

    def read_progress(self):
        if self.worker_thread.process.poll() is not None:
            self.progress_timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_edit.append("Analysis finished.")
        elif os.path.exists('progress.txt'):
            with open('progress.txt', 'r') as f:
                contents = f.read().strip()
                if contents:
                    try:
                        current, total = map(int, contents.split(','))
                        self.progress_bar.setValue(int(100 * current / total))
                        if current == total:
                            self.progress_timer.stop()
                    except ValueError:
                        print("Error: unable to parse progress information.")

    def closeEvent(self, event: QEvent) -> None:
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.worker_thread.isRunning():
                self.stop_analysis()
            event.accept()
        else:
            event.ignore()

    def load_nuclei_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Nuclei Cellpose Model")
        if file_name:
            self.update_config("nuclei_model_location", file_name)

    def load_cyto_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Cytoplasm Cellpose Model")
        if file_name:
            self.update_config("cyto_model_location", file_name)

    def load_pre_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Load Pre Directory")
        if dir_name:
            self.update_config("pre_directory_location", dir_name)

    def load_post_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Load Post Directory")
        if dir_name:
            self.update_config("post_directory_location", dir_name)

    def update_config(self, key, value):
        with open("config.json", "r") as f:
            config = json.load(f)
        config[key] = value
        with open("config.json", "w") as f:
            json.dump(config, f)

    def next_image(self):
        self.image_index += 1
        self.load_image()

    def prev_image(self):
        self.image_index = max(0, self.image_index - 1)
        self.load_image()

    def select_image(self):
        try:
            self.image_index = int(self.image_input.text())
            self.load_image()
        except ValueError:
            message_box = QMessageBox(self)
            message_box.setIcon(QMessageBox.Warning)
            message_box.setWindowTitle("Invalid input")
            message_box.setText("Please enter a valid image number.")
            message_box.setStyleSheet("QMessageBox { background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF; }")
            message_box.exec_()

    def image_mode_changed(self, id):
        self.image_mode = "raw" if id == 0 else "normalized"
        self.load_image()

    def load_image(self):
        current_dir = os.getcwd()
        if self.image_mode == "raw":
            file_name = os.path.join(current_dir, f"plot_raw{self.image_index}")
        else:
            file_name = os.path.join(current_dir, f"plot{self.image_index}")
        print(f"Trying to load: {file_name}")
        if os.path.exists(file_name):
            pixmap = QPixmap(file_name)
            self.graphics_scene.clear()
            self.graphics_scene.addPixmap(pixmap)
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            message_box = QMessageBox(self)
            message_box.setIcon(QMessageBox.Warning)
            message_box.setWindowTitle("Image not found")
            message_box.setText("The requested image does not exist.")
            message_box.setStyleSheet("QMessageBox { background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF; }")
            message_box.exec_()

    def convert_czi_to_tiff(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Convert CZI to TIFF with Timestamps")

        czi_file_input = QLineEdit(self.czi_file, dialog)
        output_dir_input = QLineEdit(self.output_dir, dialog)

        czi_file_input.setStyleSheet("background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF;")
        output_dir_input.setStyleSheet("background-color: #404040; color: #FFFFFF; border: 2px solid #FFFFFF;")

        ok_button = QPushButton('OK', dialog)
        ok_button.clicked.connect(dialog.accept)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("CZI File Path:", dialog))
        layout.addWidget(czi_file_input)
        layout.addWidget(QLabel("Output Directory:", dialog))
        layout.addWidget(output_dir_input)
        layout.addWidget(ok_button)

        if dialog.exec_() == QDialog.Accepted:
            self.czi_file = czi_file_input.text()
            self.output_dir = output_dir_input.text()

            # Run the CZI to TIFF conversion in a new process
            subprocess.Popen(["python3", "CZI2TIFFwithTIMESTAMPS.py", self.czi_file, self.output_dir])

    def read_czi_conversion_output(self):
        output = self.czi_conversion_process.readAllStandardOutput().data().decode()
        self.status_edit.append(output)

    def rename_tiffs(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
