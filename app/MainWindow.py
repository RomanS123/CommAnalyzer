import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLineEdit,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
    QSizePolicy,
    QFormLayout,
    QMessageBox,
    QDialog,
    QHBoxLayout,
    QProgressBar,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontDatabase, QIcon
import os
import json
from app.youtube_api import YouTubeAPI
from googletrans import Translator
from langdetect import detect, detect_langs
from app.SentimentClassfier import SentimentClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np

# MainWindow: GUI Component


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CommAnalyzer")
        self.setMinimumSize(1000, 800)
        self.setStyleSheet("background-color: #F1E8B8;")

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_view = self.create_main_view()
        self.result_view = self.create_result_view()

        self.stacked_widget.addWidget(self.main_view)
        self.stacked_widget.addWidget(self.result_view)

        self.stacked_widget.setCurrentWidget(self.main_view)

        self.center_window()

    def create_main_view(self):
        """
        Create the main view with a search bar, search button, and settings button.
        """
        main_layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        history_icon_path = os.path.join(
            current_dir, "../resources/icons/history_icon.png")
        settings_icon_path = os.path.join(
            current_dir, "../resources/icons/settings_icon.png")

        self.history_button = QPushButton()
        self.history_button.setIcon(QIcon(history_icon_path))
        self.history_button.setIconSize(
            self.history_button.iconSize().scaled(40, 40, Qt.KeepAspectRatio))
        self.history_button.setStyleSheet("border: none;")

        self.settings_button = QPushButton()
        self.settings_button.setIcon(QIcon(settings_icon_path))
        self.settings_button.setIconSize(
            self.settings_button.iconSize().scaled(40, 40, Qt.KeepAspectRatio))
        self.settings_button.setStyleSheet("border: none;")
        self.settings_button.clicked.connect(self.open_settings)

        top_layout.addWidget(self.history_button, alignment=Qt.AlignLeft)
        top_layout.addStretch()
        top_layout.addWidget(self.settings_button, alignment=Qt.AlignRight)

        main_layout.addLayout(top_layout)

        main_layout.addStretch(10)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(
            current_dir, "../resources/fonts/OpenSans-VariableFont_wdth,wght.ttf")

        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print("Failed to load font.")
        else:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            custom_font = QFont(font_family, 25)

        label = QLabel("CommAnalyzer")
        label.setFont(custom_font if 'custom_font' in locals()
                      else QFont("Arial", 25))
        label.setAlignment(Qt.AlignHCenter)
        main_layout.addWidget(label)

        main_layout.addStretch(1)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter YouTube video link")
        self.search_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.search_bar.setStyleSheet("""
            QLineEdit {
                padding: 20px;  
                border-radius: 30px;       
                border: 2px solid #757780;                 
            }
        """)
        main_layout.addWidget(self.search_bar)

        main_layout.addStretch(1)

        self.search_button = QPushButton("Search")
        self.search_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.search_button.clicked.connect(self.show_search_results)
        self.search_button.setStyleSheet("""
            QPushButton {
                background-color: #E8B8F1;
                border-radius: 10px;
                padding: 20px 40px;
                font-size: 30px;
            }
            QPushButton:hover {
                background-color: #f0d5f5;
            }
        """)
        main_layout.addWidget(self.search_button, alignment=Qt.AlignCenter)

        main_layout.addStretch(1)

        main_layout.addStretch(10)

        main_container = QWidget()
        main_container.setLayout(main_layout)

        return main_container

    def create_result_view(self):
        """
        Create the result view with a label, progress bar, and back button.
        """
        result_layout = QVBoxLayout()

        result_layout.addStretch(1)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)
        result_layout.addWidget(self.result_label)

        self.progress_status_label = QLabel("Progress Status")
        self.progress_status_label.setAlignment(Qt.AlignCenter)
        self.progress_status_label.setStyleSheet("font-size: 18px;")
        result_layout.addWidget(self.progress_status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #8f8f91;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
        """)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        result_layout.addWidget(self.progress_bar)

        # Embedded Pie Chart Canvas
        self.pie_chart_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        result_layout.addWidget(
            QLabel("Sentiment Distribution"), alignment=Qt.AlignCenter)
        result_layout.addWidget(self.pie_chart_canvas)

        self.pca_plot_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        result_layout.addWidget(
            QLabel("Comments Visualization"), alignment=Qt.AlignCenter)
        result_layout.addWidget(self.pca_plot_canvas)

        result_layout.addSpacing(10)

        self.back_button = QPushButton("Back")
        self.back_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.back_button.clicked.connect(self.show_main_view)
        result_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        result_layout.addStretch(2)

        result_container = QWidget()
        result_container.setLayout(result_layout)

        return result_container

    def show_search_results(self):
        """
        Processes comments and outputs statistics
        """
        search_query = self.search_bar.text()
        api_key = ""
        try:
            with open("../data/api_key.json", "r") as file:
                data = json.load(file)
                api_key = data["api_key"]
        except:
            print("Error while reading API key")

        if search_query.strip():
            yt = YouTubeAPI(api_key)
            self.progress_status_label.setText("Сбор комментариев...")
            self.progress_bar.show()

            comments = yt.get_comments(search_query.strip())
            translator = Translator()

            total_comments = len(comments)
            if total_comments == 0:
                QMessageBox.information(
                    self, "No Comments", "No comments found for this video.")
                return

            self.progress_bar.setMaximum(total_comments)
            self.progress_bar.setValue(0)
            self.stacked_widget.setCurrentWidget(self.result_view)

            # Translate comments
            translated_comments = []
            for i, comment in enumerate(comments, start=1):
                translated_comments.append(
                    translator.translate(comment, dest="en").text)
                self.progress_bar.setValue(i)
                QApplication.processEvents()

            analyzer = SentimentClassifier()

            # Sentiment Analysis Phase
            self.progress_status_label.setText("Анализ комментариев...")
            self.progress_bar.setMaximum(len(translated_comments))
            self.progress_bar.setValue(0)

            sentiments_scores = []
            hidden_states = []
            print(len(translated_comments))
            for i, comment in enumerate(translated_comments, start=1):
                input_tensor = analyzer.preprocess(
                    comment).unsqueeze(0).to(analyzer.device)
                if input_tensor.numel() == 0:
                    sentiments_scores.append(None)
                    continue

                with torch.no_grad():
                    output, hidden = analyzer.model(input_tensor)
                    sentiments_scores.append(output.item())
                    hidden_states.append(hidden[0][-1].cpu().numpy())

                # Update progress bar for analysis
                self.progress_bar.setValue(i)
                QApplication.processEvents()

            # Filter None scores and hidden states
            sentiments_scores = [x for x in sentiments_scores if x is not None]

            print(len(hidden_states))
            hidden_states = np.array(hidden_states)
            print(hidden_states.shape)

            # Sentiment Classification
            sentiments = [1 if score >=
                          0.5 else 0 for score in sentiments_scores]
            positive_count = sum(sentiments)
            negative_count = len(sentiments) - positive_count

            # --- Update Pie Chart ---
            self.pie_chart_canvas.figure.clear()
            ax_pie = self.pie_chart_canvas.figure.add_subplot(111)
            labels = ['Positive', 'Negative']
            sizes = [positive_count, negative_count]
            colors = ['#4CAF50', '#F44336']
            ax_pie.pie(sizes, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=140)
            ax_pie.set_title(
                "Sentiment Analysis: Positive vs Negative Comments")
            self.pie_chart_canvas.draw()

            # --- PCA Visualization ---
            self.pca_plot_canvas.figure.clear()
            ax_pca = self.pca_plot_canvas.figure.add_subplot(111)
            pca = PCA(n_components=2)
            hidden_states_2d = pca.fit_transform(np.squeeze(hidden_states))

            scatter = ax_pca.scatter(
                hidden_states_2d[:, 0], hidden_states_2d[:, 1], c=sentiments_scores, cmap='bwr', alpha=0.7)
            ax_pca.set_title("LSTM Hidden States Visualization (PCA)")
            ax_pca.set_xlabel("PCA Component 1")
            ax_pca.set_ylabel("PCA Component 2")
            colorbar = self.pca_plot_canvas.figure.colorbar(scatter, ax=ax_pca)
            colorbar.set_label("Sentiment (1 = Positive, 0 = Negative)")
            self.pca_plot_canvas.draw()

            # Hide the progress bar after completion
            self.progress_bar.hide()
            self.progress_status_label.setText("Анализ завершен!")
            QApplication.processEvents()

    def show_main_view(self):
        self.stacked_widget.setCurrentWidget(self.main_view)

    def open_settings(self):
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec_()

    def center_window(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

# SettingsDialog: Dialog for setting the YouTube API key


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        api_key = ""
        try:
            with open("../data/api_key.json", "r") as file:
                data = json.load(file)
                api_key = data["api_key"]
        except:
            print("Error while reading API key")

        self.api_key_input = QLineEdit(self)
        self.api_key_input.setText(api_key)
        self.api_key_input.setPlaceholderText("Enter YouTube API Key")

        save_button = QPushButton("Save", self)
        save_button.clicked.connect(self.save_api_key)

        layout = QFormLayout()
        layout.addRow("YouTube API Key:", self.api_key_input)
        layout.addRow(save_button)
        self.setLayout(layout)

    def save_api_key(self):
        api_key = self.api_key_input.text()
        if not api_key:
            QMessageBox.warning(self, "Invalid Input",
                                "API Key cannot be empty.")
            return

        if not os.path.exists("../data"):
            os.makedirs("../data")

        with open("../data/api_key.json", "w") as f:
            json.dump({"api_key": api_key}, f)

        QMessageBox.information(self, "Success", "API Key saved successfully.")
        self.accept()