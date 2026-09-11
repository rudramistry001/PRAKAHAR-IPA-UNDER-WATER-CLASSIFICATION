"""
AquaVision AI — Main Window Implementation
===========================================
PySide6 GUI Desktop Application built on the Stitch "Abyssal Precision" Design System.
Integrates with PyTorch Fish4Knowledge classification models.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QStackedWidget, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QComboBox, QLineEdit, QSplitter,
    QScrollArea, QListWidget, QListWidgetItem, QMessageBox
)

from gui.theme import STYLE_SHEET
from gui.species_data import SPECIES_MAPPING, get_species_info
from gui.workers import ModelLoaderWorker, SingleInferenceWorker, BatchInferenceWorker
from src.models import FishClassifier

NUM_CLASSES = 23
DEFAULT_DATA_ROOT = r"d:\prakhar ipa\my dataset\fish4konwledge"


class DropLabel(QLabel):
    """Interactive drag and drop image label."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drag & Drop Image Here\nor click 'Browse Image'")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #152C41;
                border-radius: 8px;
                background-color: #041D32;
                color: #8CA2BC;
                font-size: 13px;
            }
            QLabel:hover {
                border-color: #00E5FF;
                color: #00E5FF;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.parent().load_image(file_path)
                break


class MainWindow(QMainWindow):
    def __init__(self, data_root=DEFAULT_DATA_ROOT):
        super().__init__()
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.active_model_name = "ConvNeXt-Base"
        self.current_image_path = None
        self.latest_results = []

        self.setWindowTitle("AquaVision AI — Marine Species Identification Workstation")
        self.resize(1340, 860)
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(STYLE_SHEET)

        self._build_ui()
        self._init_default_model()

    # ─────────────────────────────────────────────────────────────
    #  UI Construction
    # ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Header ───────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("headerBar")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        logo_layout = QVBoxLayout()
        title = QLabel("AQUAVISION AI")
        title.setObjectName("headerTitle")
        subtitle = QLabel("Fish4Knowledge Deep Learning Classification Engine")
        subtitle.setObjectName("headerSubtitle")
        logo_layout.addWidget(title)
        logo_layout.addWidget(subtitle)
        header_layout.addLayout(logo_layout)

        header_layout.addStretch()

        # Badges
        device_text = f"CUDA: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "DEVICE: CPU"
        self.device_badge = QLabel(device_text)
        self.device_badge.setProperty("class", "BadgeCUDA" if torch.cuda.is_available() else "BadgeCPU")

        self.model_badge = QLabel(f"BACKBONE: {self.active_model_name}")
        self.model_badge.setProperty("class", "BadgeModel")

        self.latency_badge = QLabel("LATENCY: -- ms")
        self.latency_badge.setProperty("class", "BadgeCUDA")

        header_layout.addWidget(self.device_badge)
        header_layout.addWidget(self.model_badge)
        header_layout.addWidget(self.latency_badge)

        main_layout.addWidget(header)

        # ── Body Splitter (Sidebar + Stacked Pages) ──────────────
        body_frame = QFrame()
        body_layout = QHBoxLayout(body_frame)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Sidebar Navigation
        sidebar = QFrame()
        sidebar.setObjectName("sidebarFrame")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 12, 0, 12)
        sidebar_layout.setSpacing(4)

        self.nav_btn_analyzer = QPushButton("🔍  Image Analyzer")
        self.nav_btn_analyzer.setProperty("class", "nav-btn")
        self.nav_btn_analyzer.setCheckable(True)
        self.nav_btn_analyzer.setChecked(True)
        self.nav_btn_analyzer.clicked.connect(lambda: self.switch_page(0))

        self.nav_btn_batch = QPushButton("📁  Batch Processing")
        self.nav_btn_batch.setProperty("class", "nav-btn")
        self.nav_btn_batch.setCheckable(True)
        self.nav_btn_batch.clicked.connect(lambda: self.switch_page(1))

        self.nav_btn_dataset = QPushButton("📊  Species Explorer")
        self.nav_btn_dataset.setProperty("class", "nav-btn")
        self.nav_btn_dataset.setCheckable(True)
        self.nav_btn_dataset.clicked.connect(lambda: self.switch_page(2))

        self.nav_btn_config = QPushButton("⚙️  Model Config")
        self.nav_btn_config.setProperty("class", "nav-btn")
        self.nav_btn_config.setCheckable(True)
        self.nav_btn_config.clicked.connect(lambda: self.switch_page(3))

        sidebar_layout.addWidget(self.nav_btn_analyzer)
        sidebar_layout.addWidget(self.nav_btn_batch)
        sidebar_layout.addWidget(self.nav_btn_dataset)
        sidebar_layout.addWidget(self.nav_btn_config)
        sidebar_layout.addStretch()

        body_layout.addWidget(sidebar)

        # Stacked Pages
        self.pages = QStackedWidget()
        self.pages.addWidget(self._create_analyzer_page())
        self.pages.addWidget(self._create_batch_page())
        self.pages.addWidget(self._create_dataset_page())
        self.pages.addWidget(self._create_config_page())

        body_layout.addWidget(self.pages)
        main_layout.addWidget(body_frame)

    def switch_page(self, index):
        self.nav_btn_analyzer.setChecked(index == 0)
        self.nav_btn_batch.setChecked(index == 1)
        self.nav_btn_dataset.setChecked(index == 2)
        self.nav_btn_config.setChecked(index == 3)
        self.pages.setCurrentIndex(index)

    # ─────────────────────────────────────────────────────────────
    #  Page 1: Single Image Analyzer
    # ─────────────────────────────────────────────────────────────
    def _create_analyzer_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Left Column: Image Viewport
        left_card = QFrame()
        left_card.setProperty("class", "panel-card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)

        lbl_head = QLabel("SPECIMEN IMAGE VIEWPORT")
        lbl_head.setProperty("class", "section-header")
        left_layout.addWidget(lbl_head)

        self.image_drop_area = DropLabel(left_card)
        self.image_drop_area.setMinimumSize(420, 360)
        left_layout.addWidget(self.image_drop_area, stretch=1)

        # Controls
        ctrl_layout = QHBoxLayout()
        btn_browse = QPushButton("Browse Image...")
        btn_browse.setProperty("class", "btn-secondary")
        btn_browse.clicked.connect(self.browse_single_image)

        self.chk_tta = QCheckBox("Enable TTA (Test-Time Augmentation)")

        ctrl_layout.addWidget(btn_browse)
        ctrl_layout.addWidget(self.chk_tta)
        left_layout.addLayout(ctrl_layout)

        btn_analyze = QPushButton("ANALYZE SPECIMEN")
        btn_analyze.setProperty("class", "btn-primary")
        btn_analyze.clicked.connect(self.run_single_inference)
        left_layout.addWidget(btn_analyze)

        layout.addWidget(left_card, stretch=5)

        # Right Column: Prediction Results & Top-5 Distribution
        right_card = QFrame()
        right_card.setProperty("class", "panel-card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        right_head = QLabel("CLASSIFICATION METRICS")
        right_head.setProperty("class", "section-header")
        right_layout.addWidget(right_head)

        # Result Details Card
        res_card = QFrame()
        res_card.setProperty("class", "panel-card-elevated")
        res_layout = QVBoxLayout(res_card)

        self.lbl_common_name = QLabel("No Specimen Analyzed")
        self.lbl_common_name.setStyleSheet("font-size: 18px; font-weight: 700; color: #00E5FF;")

        self.lbl_scientific_name = QLabel("Select an image to run classification")
        self.lbl_scientific_name.setStyleSheet("font-size: 13px; font-style: italic; color: #8CA2BC;")

        self.lbl_family = QLabel("Family: --")
        self.lbl_family.setStyleSheet("font-size: 12px; color: #8CA2BC;")

        # Confidence Gauge
        conf_layout = QHBoxLayout()
        conf_lbl = QLabel("Model Confidence:")
        conf_lbl.setProperty("class", "muted-label")
        self.lbl_conf_value = QLabel("0.0%")
        self.lbl_conf_value.setStyleSheet("font-weight: 700; color: #00E676; font-family: monospace;")
        conf_layout.addWidget(conf_lbl)
        conf_layout.addStretch()
        conf_layout.addWidget(self.lbl_conf_value)

        self.progress_conf = QProgressBar()
        self.progress_conf.setRange(0, 100)
        self.progress_conf.setValue(0)
        self.progress_conf.setProperty("class", "progress-high")

        res_layout.addWidget(self.lbl_common_name)
        res_layout.addWidget(self.lbl_scientific_name)
        res_layout.addWidget(self.lbl_family)
        res_layout.addSpacing(6)
        res_layout.addLayout(conf_layout)
        res_layout.addWidget(self.progress_conf)

        right_layout.addWidget(res_card)

        # Description Card
        self.lbl_description = QLabel("Description: Select an image and click Analyze.")
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet("color: #CFE4FF; font-size: 12px; line-height: 1.4;")
        right_layout.addWidget(self.lbl_description)

        # Top 5 Distribution
        lbl_top5_head = QLabel("TOP 5 SPECIES PROBABILITIES")
        lbl_top5_head.setProperty("class", "section-header")
        right_layout.addWidget(lbl_top5_head)

        self.top5_container = QVBoxLayout()
        self.top5_bars = []
        for i in range(5):
            row_layout = QHBoxLayout()
            lbl_sp = QLabel(f"#{i+1} --")
            lbl_sp.setStyleSheet("font-size: 11px; color: #CFE4FF;")
            lbl_sp.setMinimumWidth(180)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(10)
            bar.setTextVisible(False)

            val_lbl = QLabel("0.0%")
            val_lbl.setStyleSheet("font-size: 10px; font-family: monospace; color: #8CA2BC;")
            val_lbl.setFixedWidth(45)

            row_layout.addWidget(lbl_sp)
            row_layout.addWidget(bar)
            row_layout.addWidget(val_lbl)

            self.top5_container.addLayout(row_layout)
            self.top5_bars.append((lbl_sp, bar, val_lbl))

        right_layout.addLayout(self.top5_container)
        right_layout.addStretch()

        layout.addWidget(right_card, stretch=6)
        return page

    def browse_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Fish Specimen Image", self.data_root,
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(
            self.image_drop_area.size(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_drop_area.setPixmap(scaled_pixmap)

    def run_single_inference(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select or drop an image first.")
            return

        if self.model is None:
            QMessageBox.warning(self, "Model Not Ready", "Classification model is loading. Please wait.")
            return

        use_tta = self.chk_tta.isChecked()
        self.worker_single = SingleInferenceWorker(
            self.current_image_path, self.model, self.device,
            image_size=288, use_tta=use_tta
        )
        self.worker_single.finished.connect(self._on_single_inference_finished)
        self.worker_single.error.connect(lambda err: QMessageBox.critical(self, "Error", err))
        self.worker_single.start()

    def _on_single_inference_finished(self, res):
        self.lbl_common_name.setText(res.get("common_name", "Unknown Species"))
        class_name = res.get("class_name", f"fish_{str(res.get('top_class_id', 0) + 1).zfill(2)}")
        self.lbl_scientific_name.setText(f"{res.get('scientific_name', 'Specimen')} ({class_name})")
        self.lbl_family.setText(f"Family: {res.get('family', 'Pomacentridae')}")
        self.lbl_description.setText(res.get("description", ""))

        conf_pct = res.get("confidence", 0.0) * 100.0
        self.lbl_conf_value.setText(f"{conf_pct:.2f}%")
        self.progress_conf.setValue(int(conf_pct))

        # Color coding confidence
        if conf_pct >= 90.0:
            self.progress_conf.setProperty("class", "progress-high")
            self.lbl_conf_value.setStyleSheet("font-weight: 700; color: #00E676; font-family: monospace;")
        elif conf_pct >= 60.0:
            self.progress_conf.setProperty("class", "progress-medium")
            self.lbl_conf_value.setStyleSheet("font-weight: 700; color: #FFB300; font-family: monospace;")
        else:
            self.progress_conf.setProperty("class", "progress-low")
            self.lbl_conf_value.setStyleSheet("font-weight: 700; color: #FF7043; font-family: monospace;")
        self.progress_conf.style().unpolish(self.progress_conf)
        self.progress_conf.style().polish(self.progress_conf)

        self.latency_badge.setText(f"LATENCY: {res.get('latency_ms', 0.0):.1f} ms")

        # Top 5
        for i, item in enumerate(res["top5"]):
            lbl_sp, bar, val_lbl = self.top5_bars[i]
            pct = item["probability"] * 100.0
            lbl_sp.setText(f"#{i+1} {item['common_name']}")
            bar.setValue(int(pct))
            val_lbl.setText(f"{pct:.1f}%")

    # ─────────────────────────────────────────────────────────────
    #  Page 2: Batch Processing
    # ─────────────────────────────────────────────────────────────
    def _create_batch_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        head_card = QFrame()
        head_card.setProperty("class", "panel-card")
        head_layout = QHBoxLayout(head_card)

        lbl_path = QLabel("Target Directory:")
        lbl_path.setProperty("class", "muted-label")
        self.txt_batch_dir = QLineEdit()
        default_img_dir = os.path.join(self.data_root, 'fish_image')
        self.txt_batch_dir.setText(default_img_dir if os.path.exists(default_img_dir) else self.data_root)

        btn_browse_batch = QPushButton("Browse...")
        btn_browse_batch.setProperty("class", "btn-secondary")
        btn_browse_batch.clicked.connect(self.browse_batch_dir)

        btn_run_batch = QPushButton("START BATCH SCAN")
        btn_run_batch.setProperty("class", "btn-primary")
        btn_run_batch.clicked.connect(self.run_batch_inference)

        head_layout.addWidget(lbl_path)
        head_layout.addWidget(self.txt_batch_dir, stretch=1)
        head_layout.addWidget(btn_browse_batch)
        head_layout.addWidget(btn_run_batch)
        layout.addWidget(head_card)

        # Progress bar
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        layout.addWidget(self.batch_progress)

        # Table
        self.table_batch = QTableWidget(0, 5)
        self.table_batch.setHorizontalHeaderLabels([
            "Filename", "Class ID", "Scientific Name", "Common Name", "Confidence"
        ])
        self.table_batch.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_batch)

        # Export Footer
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        btn_export = QPushButton("Export Results CSV...")
        btn_export.setProperty("class", "btn-secondary")
        btn_export.clicked.connect(self.export_batch_csv)
        footer_layout.addWidget(btn_export)
        layout.addLayout(footer_layout)

        return page

    def browse_batch_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory", self.data_root)
        if folder:
            self.txt_batch_dir.setText(folder)

    def run_batch_inference(self):
        folder = self.txt_batch_dir.text().strip()
        if not os.path.exists(folder):
            QMessageBox.warning(self, "Invalid Path", "Target directory does not exist.")
            return

        if self.model is None:
            QMessageBox.warning(self, "Model Not Ready", "Model is loading. Please wait.")
            return

        self.table_batch.setRowCount(0)
        self.latest_results = []
        self.batch_progress.setValue(0)

        self.worker_batch = BatchInferenceWorker(folder, self.model, self.device)
        self.worker_batch.progress.connect(lambda cur, tot: self.batch_progress.setValue(int(cur / tot * 100)))
        self.worker_batch.image_result.connect(self._add_batch_row)
        self.worker_batch.batch_finished.connect(lambda res: QMessageBox.information(self, "Complete", f"Batch inference completed for {len(res)} images."))
        self.worker_batch.error.connect(lambda err: QMessageBox.critical(self, "Error", err))
        self.worker_batch.start()

    def _add_batch_row(self, res):
        self.latest_results.append(res)
        row = self.table_batch.rowCount()
        self.table_batch.insertRow(row)
        self.table_batch.setItem(row, 0, QTableWidgetItem(res["filename"]))
        self.table_batch.setItem(row, 1, QTableWidgetItem(f"fish_{str(res['class_id']+1).zfill(2)}"))
        self.table_batch.setItem(row, 2, QTableWidgetItem(res["scientific_name"]))
        self.table_batch.setItem(row, 3, QTableWidgetItem(res["common_name"]))
        conf_item = QTableWidgetItem(f"{res['confidence']*100.0:.2f}%")
        self.table_batch.setItem(row, 4, conf_item)

    def export_batch_csv(self):
        if not self.latest_results:
            QMessageBox.information(self, "No Data", "No batch inference results to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Export CSV", self.data_root, "CSV Files (*.csv)")
        if file_path:
            pd.DataFrame(self.latest_results).to_csv(file_path, index=False)
            QMessageBox.information(self, "Saved", f"Exported results to {file_path}")

    # ─────────────────────────────────────────────────────────────
    #  Page 3: Species Explorer
    # ─────────────────────────────────────────────────────────────
    def _create_dataset_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Species List
        list_card = QFrame()
        list_card.setProperty("class", "panel-card")
        list_layout = QVBoxLayout(list_card)

        lbl_lst_head = QLabel("FISH4KNOWLEDGE SPECIES (23)")
        lbl_lst_head.setProperty("class", "section-header")
        list_layout.addWidget(lbl_lst_head)

        self.species_list = QListWidget()
        for idx in range(NUM_CLASSES):
            info = get_species_info(idx)
            item = QListWidgetItem(f"[{info['class_name']}] {info['common_name']}")
            item.setData(Qt.UserRole, idx)
            self.species_list.addItem(item)

        self.species_list.currentRowChanged.connect(self._on_species_selected)
        list_layout.addWidget(self.species_list)

        layout.addWidget(list_card, stretch=4)

        # Species Detail View
        detail_card = QFrame()
        detail_card.setProperty("class", "panel-card")
        detail_layout = QVBoxLayout(detail_card)

        self.exp_common_name = QLabel("Select a Species")
        self.exp_common_name.setStyleSheet("font-size: 20px; font-weight: 700; color: #00E5FF;")

        self.exp_scientific_name = QLabel("--")
        self.exp_scientific_name.setStyleSheet("font-size: 14px; font-style: italic; color: #8CA2BC;")

        self.exp_family = QLabel("Family: --")
        self.exp_family.setStyleSheet("font-size: 12px; color: #8CA2BC;")

        self.exp_desc = QLabel("Detailed description will be displayed here.")
        self.exp_desc.setWordWrap(True)

        detail_layout.addWidget(self.exp_common_name)
        detail_layout.addWidget(self.exp_scientific_name)
        detail_layout.addWidget(self.exp_family)
        detail_layout.addSpacing(10)
        detail_layout.addWidget(self.exp_desc)

        # Sample Image Preview
        detail_layout.addSpacing(12)
        lbl_sample_head = QLabel("DATASET SAMPLE PREVIEW")
        lbl_sample_head.setProperty("class", "section-header")
        detail_layout.addWidget(lbl_sample_head)

        self.sample_img_label = QLabel("No sample loaded")
        self.sample_img_label.setAlignment(Qt.AlignCenter)
        self.sample_img_label.setMinimumSize(320, 240)
        self.sample_img_label.setStyleSheet("border: 1px solid #152C41; background-color: #041D32;")
        detail_layout.addWidget(self.sample_img_label, stretch=1)

        btn_load_sample = QPushButton("LOAD SAMPLE IN ANALYZER")
        btn_load_sample.setProperty("class", "btn-primary")
        btn_load_sample.clicked.connect(self._load_selected_sample_in_analyzer)
        detail_layout.addWidget(btn_load_sample)

        layout.addWidget(detail_card, stretch=6)
        self.species_list.setCurrentRow(0)
        return page

    def _on_species_selected(self, row):
        if row < 0: return
        info = get_species_info(row)
        self.exp_common_name.setText(info["common_name"])
        self.exp_scientific_name.setText(f"{info['scientific_name']} ({info['class_name']})")
        self.exp_family.setText(f"Family: {info['family']}")
        self.exp_desc.setText(info["description"])

        # Try finding sample image in dataset directory
        sample_folder = os.path.join(self.data_root, 'fish_image', info["class_name"])
        images = glob.glob(os.path.join(sample_folder, '*.png'))
        if images:
            self.selected_sample_path = images[0]
            pix = QPixmap(self.selected_sample_path).scaled(
                self.sample_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.sample_img_label.setPixmap(pix)
        else:
            self.selected_sample_path = None
            self.sample_img_label.setText("No dataset image found on disk")

    def _load_selected_sample_in_analyzer(self):
        if hasattr(self, 'selected_sample_path') and self.selected_sample_path:
            self.load_image(self.selected_sample_path)
            self.switch_page(0)
            self.run_single_inference()

    # ─────────────────────────────────────────────────────────────
    #  Page 4: Model Configuration
    # ─────────────────────────────────────────────────────────────
    def _create_config_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        card = QFrame()
        card.setProperty("class", "panel-card")
        card_layout = QVBoxLayout(card)

        head = QLabel("MODEL BACKBONE & CHECKPOINT MANAGER")
        head.setProperty("class", "section-header")
        card_layout.addWidget(head)

        grid = QVBoxLayout()

        # Backbone Combo
        row1 = QHBoxLayout()
        lbl_arch = QLabel("Select Architecture:")
        lbl_arch.setFixedWidth(160)
        self.combo_arch = QComboBox()
        self.combo_arch.addItems(["convnext_base", "efficientnet_v2m"])
        row1.addWidget(lbl_arch)
        row1.addWidget(self.combo_arch, stretch=1)
        grid.addLayout(row1)

        # Checkpoint Path
        row2 = QHBoxLayout()
        lbl_ckpt = QLabel("Checkpoint File (.pt):")
        lbl_ckpt.setFixedWidth(160)
        self.txt_ckpt = QLineEdit()
        self.txt_ckpt.setText(os.path.join(self.data_root, "runs", "convnext_base", "best_macro_f1.pt"))
        btn_browse_ckpt = QPushButton("Browse...")
        btn_browse_ckpt.setProperty("class", "btn-secondary")
        btn_browse_ckpt.clicked.connect(self.browse_checkpoint)

        row2.addWidget(lbl_ckpt)
        row2.addWidget(self.txt_ckpt, stretch=1)
        row2.addWidget(btn_browse_ckpt)
        grid.addLayout(row2)

        card_layout.addLayout(grid)

        btn_reload = QPushButton("RELOAD MODEL WEIGHTS")
        btn_reload.setProperty("class", "btn-primary")
        btn_reload.clicked.connect(self.reload_model)
        card_layout.addWidget(btn_reload)

        card_layout.addStretch()
        layout.addWidget(card)
        return page

    def browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", self.data_root, "PyTorch Model (*.pt *.pth)"
        )
        if path:
            self.txt_ckpt.setText(path)

    def reload_model(self):
        ckpt_path = self.txt_ckpt.text().strip()
        arch = self.combo_arch.currentText()
        self.active_model_name = arch
        self.model_badge.setText(f"BACKBONE: {arch.upper()}")

        self.worker_loader = ModelLoaderWorker(ckpt_path, self.device)
        self.worker_loader.loaded.connect(self._on_model_loaded)
        self.worker_loader.error.connect(lambda err: QMessageBox.critical(self, "Error", err))
        self.worker_loader.start()

    def _init_default_model(self):
        default_ckpt = os.path.join(self.data_root, "runs", "convnext_base", "best_macro_f1.pt")
        self.worker_loader = ModelLoaderWorker(default_ckpt, self.device)
        self.worker_loader.loaded.connect(self._on_model_loaded)
        self.worker_loader.error.connect(lambda err: None)
        self.worker_loader.start()

    def _on_model_loaded(self, model, arch, msg):
        self.model = model
        self.active_model_name = arch
        self.model_badge.setText(f"BACKBONE: {arch.upper()}")
