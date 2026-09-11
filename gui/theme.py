"""
AquaVision AI — Abyssal Precision Theme Sheet
==============================================
Dark mode marine computer vision workstation stylesheet.
Colors:
  - Base Background: #001427
  - Primary Container: #082136
  - Secondary Container: #152C41
  - Card/Overlay Surface: #20364C
  - Bioluminescent Cyan (Accent): #00E5FF
  - High Confidence Emerald: #00E676
  - Warning Gold: #FFB300
  - Alert Coral: #FF7043
  - Monospace Font: JetBrains Mono / Consolas
  - Body Font: Inter / Segoe UI
  - Heading Font: Space Grotesk / Segoe UI Semibold
"""

STYLE_SHEET = """
/* ─────────────────────────────────────────────────────────────
   GLOBAL & ROOT
   ───────────────────────────────────────────────────────────── */
QWidget {
    background-color: #001427;
    color: #CFE4FF;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
    selection-background-color: #00E5FF;
    selection-color: #001427;
}

QMainWindow {
    background-color: #001427;
}

/* ─────────────────────────────────────────────────────────────
   TOP HEADER BAR
   ───────────────────────────────────────────────────────────── */
#headerBar {
    background-color: #041D32;
    border-bottom: 1px solid #152C41;
    min-height: 52px;
    max-height: 52px;
}

#headerTitle {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

#headerSubtitle {
    color: #8CA2BC;
    font-size: 11px;
}

.BadgeCUDA {
    background-color: rgba(0, 230, 118, 0.15);
    color: #00E676;
    border: 1px solid rgba(0, 230, 118, 0.4);
    border-radius: 4px;
    padding: 3px 8px;
    font-family: 'Consolas', 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
}

.BadgeCPU {
    background-color: rgba(255, 179, 0, 0.15);
    color: #FFB300;
    border: 1px solid rgba(255, 179, 0, 0.4);
    border-radius: 4px;
    padding: 3px 8px;
    font-family: 'Consolas', 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
}

.BadgeModel {
    background-color: rgba(0, 229, 255, 0.15);
    color: #00E5FF;
    border: 1px solid rgba(0, 229, 255, 0.4);
    border-radius: 4px;
    padding: 3px 8px;
    font-family: 'Consolas', 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
}

/* ─────────────────────────────────────────────────────────────
   SIDEBAR NAVIGATION DOCK
   ───────────────────────────────────────────────────────────── */
#sidebarFrame {
    background-color: #082136;
    border-right: 1px solid #152C41;
    min-width: 220px;
    max-width: 220px;
}

QPushButton.nav-btn {
    background-color: transparent;
    color: #8CA2BC;
    border: none;
    border-left: 3px solid transparent;
    padding: 12px 16px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
    border-radius: 0px;
}

QPushButton.nav-btn:hover {
    background-color: #152C41;
    color: #FFFFFF;
}

QPushButton.nav-btn:checked {
    background-color: #152C41;
    color: #00E5FF;
    border-left: 3px solid #00E5FF;
}

/* ─────────────────────────────────────────────────────────────
   PANELS & CARDS
   ───────────────────────────────────────────────────────────── */
QFrame.panel-card {
    background-color: #082136;
    border: 1px solid #152C41;
    border-radius: 6px;
}

QFrame.panel-card-elevated {
    background-color: #152C41;
    border: 1px solid #20364C;
    border-radius: 6px;
}

QLabel.section-header {
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

QLabel.muted-label {
    color: #8CA2BC;
    font-size: 12px;
}

QLabel.code-label {
    font-family: 'Consolas', 'JetBrains Mono', monospace;
    color: #00E5FF;
    font-size: 12px;
}

/* ─────────────────────────────────────────────────────────────
   BUTTONS
   ───────────────────────────────────────────────────────────── */
QPushButton.btn-primary {
    background-color: #00E5FF;
    color: #001427;
    font-weight: 700;
    font-size: 13px;
    border: none;
    border-radius: 4px;
    padding: 8px 18px;
}

QPushButton.btn-primary:hover {
    background-color: #40EDFF;
}

QPushButton.btn-primary:pressed {
    background-color: #00BFA5;
}

QPushButton.btn-secondary {
    background-color: #152C41;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 13px;
    border: 1px solid #20364C;
    border-radius: 4px;
    padding: 8px 16px;
}

QPushButton.btn-secondary:hover {
    background-color: #20364C;
    border-color: #00E5FF;
    color: #00E5FF;
}

QPushButton.btn-secondary:pressed {
    background-color: #082136;
}

/* ─────────────────────────────────────────────────────────────
   PROGRESS BARS & GAUGES
   ───────────────────────────────────────────────────────────── */
QProgressBar {
    background-color: #041D32;
    border: 1px solid #152C41;
    border-radius: 4px;
    height: 14px;
    text-align: center;
    color: #FFFFFF;
    font-size: 10px;
    font-family: 'Consolas', monospace;
}

QProgressBar::chunk {
    background-color: #00E5FF;
    border-radius: 3px;
}

QProgressBar.progress-high::chunk {
    background-color: #00E676;
}

QProgressBar.progress-medium::chunk {
    background-color: #FFB300;
}

QProgressBar.progress-low::chunk {
    background-color: #FF7043;
}

/* ─────────────────────────────────────────────────────────────
   TABLES & LISTS
   ───────────────────────────────────────────────────────────── */
QTableWidget, QListWidget {
    background-color: #041D32;
    border: 1px solid #152C41;
    gridline-color: #152C41;
    border-radius: 4px;
    color: #CFE4FF;
}

QTableWidget::item, QListWidget::item {
    padding: 6px;
}

QTableWidget::item:selected, QListWidget::item:selected {
    background-color: #152C41;
    color: #00E5FF;
}

QHeaderView::section {
    background-color: #082136;
    color: #8CA2BC;
    font-weight: 600;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #152C41;
    border-right: 1px solid #152C41;
    padding: 6px;
}

/* ─────────────────────────────────────────────────────────────
   COMBOBOX & INPUTS
   ───────────────────────────────────────────────────────────── */
QComboBox, QLineEdit {
    background-color: #041D32;
    border: 1px solid #152C41;
    border-radius: 4px;
    padding: 6px 10px;
    color: #FFFFFF;
}

QComboBox:focus, QLineEdit:focus {
    border: 1px solid #00E5FF;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #082136;
    border: 1px solid #152C41;
    selection-background-color: #152C41;
    selection-color: #00E5FF;
}

QCheckBox {
    color: #CFE4FF;
    font-size: 12px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #152C41;
    background-color: #041D32;
    border-radius: 3px;
}

QCheckBox::indicator:checked {
    background-color: #00E5FF;
    border-color: #00E5FF;
}

/* ─────────────────────────────────────────────────────────────
   SCROLL BARS
   ───────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background-color: #041D32;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #152C41;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #00E5FF;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
