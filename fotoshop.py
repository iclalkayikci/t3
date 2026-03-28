"""
FotoShop — PyQt5 Interaktif Goruntu Isleme Araci
==================================================
Ilk 6 dersin tekniklerini birlestiren, Gaussian ve Median
filtreleri ile gercek zamanli fotograf duzenleme uygulamasi.

Kullanilanlar:
  01 - Goruntu yukleme, RGB / Gri ton donusumu, histogram
  02 - Ornekleme (Sampling) - cozunurluk azaltma
  03 - Kuantalama (Quantization) - bit derinligi
  04 - Renk uzaylari - HSV parlaklik / doygunluk / ton
  05 - CLAHE kontrast iyilestirme
  06 - Gaussian Blur & Median Blur ile restorasyon

Kullanim:
  py fotoshop.py
"""

import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider,
    QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QGridLayout, QSplitter, QStatusBar, QMessageBox,
    QComboBox, QCheckBox, QScrollArea, QFrame, QColorDialog
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import (QImage, QPixmap, QFont, QPalette, QColor,
                          QIcon, QPainter, QPen, QCursor)


# ── Stil (koyu tema) ────────────────────────────────────────────────────────
KOYU_TEMA = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #3a3a5e;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 18px;
    font-weight: bold;
    font-size: 13px;
    color: #4fc3f7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QSlider::groove:horizontal {
    border: 1px solid #3a3a5e;
    height: 6px;
    background: #0d1117;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4fc3f7;
    border: 2px solid #2196f3;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #81d4fa;
    border: 2px solid #4fc3f7;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1565c0, stop:1 #4fc3f7);
    border-radius: 3px;
}
QPushButton {
    background-color: #2a2a4e;
    border: 1px solid #4fc3f7;
    border-radius: 6px;
    padding: 8px 18px;
    color: #4fc3f7;
    font-weight: bold;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #3a3a6e;
    border-color: #81d4fa;
    color: #81d4fa;
}
QPushButton:pressed {
    background-color: #4fc3f7;
    color: #1a1a2e;
}
QPushButton#btn_kaydet {
    border-color: #66bb6a;
    color: #66bb6a;
}
QPushButton#btn_kaydet:hover {
    background-color: #2e7d32;
    color: white;
}
QPushButton#btn_sifirla {
    border-color: #ff8a65;
    color: #ff8a65;
}
QPushButton#btn_sifirla:hover {
    background-color: #e64a19;
    color: white;
}
QLabel#lbl_baslik {
    font-size: 11px;
    color: #b0bec5;
    font-weight: normal;
}
QLabel#lbl_deger {
    font-size: 11px;
    color: #ffcc80;
    font-weight: bold;
    min-width: 45px;
}
QLabel#lbl_goruntu {
    background-color: #0d1117;
    border: 1px solid #3a3a5e;
    border-radius: 6px;
}
QStatusBar {
    background-color: #0d1117;
    color: #b0bec5;
    font-size: 12px;
    border-top: 1px solid #3a3a5e;
}
QComboBox {
    background-color: #2a2a4e;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #2a2a4e;
    color: #e0e0e0;
    selection-background-color: #4fc3f7;
}
QCheckBox {
    color: #b0bec5;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4fc3f7;
    border-radius: 3px;
    background-color: #0d1117;
}
QCheckBox::indicator:checked {
    background-color: #4fc3f7;
}
QScrollArea {
    border: none;
}
"""


def cv2_to_qpixmap(img, max_w=None, max_h=None):
    """OpenCV BGR/Gray goruntusunu QPixmap'e donustur."""
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if max_w and max_h:
        pix = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


def psnr(a, b):
    """PSNR hesapla."""
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


class KaydiriciBolum(QWidget):
    """Etiket + slider + deger gostergesinden olusan tekrar kullanilabilir widget."""

    def __init__(self, baslik, minimum, maximum, varsayilan, ipucu="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.lbl_baslik = QLabel(baslik)
        self.lbl_baslik.setObjectName("lbl_baslik")
        self.lbl_baslik.setFixedWidth(120)
        if ipucu:
            self.lbl_baslik.setToolTip(ipucu)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(varsayilan)
        self.slider.setMinimumWidth(180)

        self.lbl_deger = QLabel(str(varsayilan))
        self.lbl_deger.setObjectName("lbl_deger")
        self.lbl_deger.setAlignment(Qt.AlignCenter)

        self.slider.valueChanged.connect(lambda v: self.lbl_deger.setText(str(v)))

        layout.addWidget(self.lbl_baslik)
        layout.addWidget(self.slider)
        layout.addWidget(self.lbl_deger)

    def value(self):
        return self.slider.value()

    def setValue(self, v):
        self.slider.setValue(v)

    def connectChanged(self, fn):
        self.slider.valueChanged.connect(fn)


class GoruntuLabel(QLabel):
    """Mouse olaylarini yakalayan ozel goruntu etiketi."""
    mousePressed = pyqtSignal(int, int)   # img x, y
    mouseMoved = pyqtSignal(int, int)
    mouseReleased = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._img_rect = QRect()  # pixmap'in label icindeki konumu

    def setPixmap(self, pix):
        super().setPixmap(pix)
        if pix and not pix.isNull():
            lw, lh = self.width(), self.height()
            pw, ph = pix.width(), pix.height()
            x = (lw - pw) // 2
            y = (lh - ph) // 2
            self._img_rect = QRect(x, y, pw, ph)

    def _to_img_coords(self, pos):
        if self._img_rect.isNull() or not self.pixmap():
            return None, None
        x = pos.x() - self._img_rect.x()
        y = pos.y() - self._img_rect.y()
        if x < 0 or y < 0 or x >= self._img_rect.width() or y >= self._img_rect.height():
            return None, None
        return x, y

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            x, y = self._to_img_coords(e.pos())
            if x is not None:
                self.mousePressed.emit(x, y)

    def mouseMoveEvent(self, e):
        x, y = self._to_img_coords(e.pos())
        if x is not None:
            self.mouseMoved.emit(x, y)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            x, y = self._to_img_coords(e.pos())
            if x is not None:
                self.mouseReleased.emit(x, y)


class FotoShopPenceresi(QMainWindow):
    """Ana pencere."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FotoShop - Goruntu Isleme Araci")
        self.setMinimumSize(1100, 700)

        self.orijinal = None
        self.sonuc = None
        self._undo_stack = []  # geri alma yigini
        self._aktif_arac = "yok"  # yok, kirp, sil, ciz
        self._firca_renk = (255, 255, 255)  # beyaz
        self._firca_boyut = 15
        self._cizim_aktif = False
        self._kirp_baslangic = None
        self._kirp_bitis = None
        self._son_cizim_noktasi = None
        self._silme_maskesi = None
        self._islem_oncesi_orijinal = None
        self.mutlak_orijinal = None
        self._kement_noktalari = []

        # Webcam degiskenleri
        self.kamera = None
        self.kamera_aktif = False
        self.kamera_timer = QTimer()
        self.kamera_timer.timeout.connect(self._kamera_kare_al)

        self._arayuz_olustur()
        self._varsayilan_yukle()

    # ── Arayuz ──────────────────────────────────────────────────────────────
    def _arayuz_olustur(self):

        merkez = QWidget()
        self.setCentralWidget(merkez)
        ana_layout = QHBoxLayout(merkez)
        ana_layout.setSpacing(10)
        ana_layout.setContentsMargins(10, 10, 10, 10)

        # ── Sol: Kontrol Paneli ─────────────────────────────────────────────
        sol_scroll = QScrollArea()
        sol_scroll.setWidgetResizable(True)
        sol_scroll.setFixedWidth(380)
        sol_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sol_widget = QWidget()
        sol_layout = QVBoxLayout(sol_widget)
        sol_layout.setSpacing(6)

        # Dosya islemleri
        grp_dosya = QGroupBox("Dosya")
        g_dosya = QGridLayout(grp_dosya)
        self.btn_ac = QPushButton("Goruntu Ac")
        self.btn_ac.clicked.connect(self._dosya_ac)
        self.btn_kaydet = QPushButton("Kaydet")
        self.btn_kaydet.setObjectName("btn_kaydet")
        self.btn_kaydet.clicked.connect(self._kaydet)
        self.btn_sifirla = QPushButton("Sifirla")
        self.btn_sifirla.setObjectName("btn_sifirla")
        self.btn_sifirla.clicked.connect(self._sifirla)
        g_dosya.addWidget(self.btn_ac, 0, 0)
        g_dosya.addWidget(self.btn_kaydet, 0, 1)
        g_dosya.addWidget(self.btn_sifirla, 0, 2)
        sol_layout.addWidget(grp_dosya)

        # Webcam islemleri
        grp_kamera = QGroupBox("Canli Kamera (Webcam)")
        grp_kamera.setStyleSheet(
            "QGroupBox { color: #ff8a65; border-color: #ff8a65; }")
        g_kamera = QGridLayout(grp_kamera)
        self.btn_kamera_ac = QPushButton("Kamerayi Ac")
        self.btn_kamera_ac.setStyleSheet(
            "QPushButton { border-color: #ff8a65; color: #ff8a65; }"
            "QPushButton:hover { background-color: #e64a19; color: white; }")
        self.btn_kamera_ac.clicked.connect(self._kamera_ac)
        self.btn_kamera_kapat = QPushButton("Kamerayi Kapat")
        self.btn_kamera_kapat.setStyleSheet(
            "QPushButton { border-color: #ef5350; color: #ef5350; }"
            "QPushButton:hover { background-color: #c62828; color: white; }")
        self.btn_kamera_kapat.clicked.connect(self._kamera_kapat)
        self.btn_kamera_kapat.setEnabled(False)
        self.btn_fotograf = QPushButton("Fotograf Cek")
        self.btn_fotograf.setStyleSheet(
            "QPushButton { border-color: #66bb6a; color: #66bb6a; }"
            "QPushButton:hover { background-color: #2e7d32; color: white; }")
        self.btn_fotograf.clicked.connect(self._fotograf_cek)
        self.btn_fotograf.setEnabled(False)
        self.cmb_kamera = QComboBox()
        self.cmb_kamera.addItems(["Kamera 0", "Kamera 1", "Kamera 2"])
        g_kamera.addWidget(self.cmb_kamera, 0, 0, 1, 3)
        g_kamera.addWidget(self.btn_kamera_ac, 1, 0)
        g_kamera.addWidget(self.btn_kamera_kapat, 1, 1)
        g_kamera.addWidget(self.btn_fotograf, 1, 2)
        sol_layout.addWidget(grp_kamera)

        # 06 - Restorasyon (Gaussian & Median)
        grp_filtre = QGroupBox("06 - Gaussian & Median Filtre")
        g_filtre = QVBoxLayout(grp_filtre)
        self.sld_gauss = KaydiriciBolum("Gaussian Blur", 0, 30, 0,
            "Gaussian bulaklastirma. 0=kapali. Deger = cekirdek yarisi (k=2v+1)")
        self.sld_median = KaydiriciBolum("Median Blur", 0, 30, 0,
            "Median filtre. Tuz-biber gurultusune etkili. 0=kapali.")
        g_filtre.addWidget(self.sld_gauss)
        g_filtre.addWidget(self.sld_median)
        sol_layout.addWidget(grp_filtre)

        # Gurultu ekleme
        grp_gurultu = QGroupBox("Gurultu Ekle (Test icin)")
        g_gurultu = QVBoxLayout(grp_gurultu)
        self.sld_gurultu = KaydiriciBolum("Gurultu Siddeti", 0, 100, 0,
            "Yapay Gaussian gurultusu ekler (0 = kapali)")
        self.chk_tuz_biber = QCheckBox("Tuz-Biber Gurultusu")
        self.sld_tuz_biber = KaydiriciBolum("Tuz-Biber %", 0, 30, 0,
            "Tuz-biber gurultu orani (%)")
        g_gurultu.addWidget(self.sld_gurultu)
        g_gurultu.addWidget(self.chk_tuz_biber)
        g_gurultu.addWidget(self.sld_tuz_biber)
        sol_layout.addWidget(grp_gurultu)

        # 02 - Ornekleme
        grp_ornek = QGroupBox("02 - Ornekleme (Sampling)")
        g_ornek = QVBoxLayout(grp_ornek)
        self.sld_ornek = KaydiriciBolum("Kucultme (1/n)", 1, 16, 1,
            "Cozunurlugu n'e boler, sonra geri buyutur. Piksellesme etkisi.")
        g_ornek.addWidget(self.sld_ornek)
        sol_layout.addWidget(grp_ornek)

        # 03 - Kuantalama
        grp_kuant = QGroupBox("03 - Kuantalama (Quantization)")
        g_kuant = QVBoxLayout(grp_kuant)
        self.sld_bit = KaydiriciBolum("Bit Derinligi", 1, 8, 8,
            "Renk seviyesi: 2^bit. 8=orijinal, 1=siyah-beyaz")
        g_kuant.addWidget(self.sld_bit)
        sol_layout.addWidget(grp_kuant)

        # 04 - Renk Ayarlari (HSV)
        grp_renk = QGroupBox("04 - Renk Uzayi (HSV)")
        g_renk = QVBoxLayout(grp_renk)
        self.sld_parlak = KaydiriciBolum("Parlaklik", -100, 100, 0,
            "HSV V kanalini artir/azalt")
        self.sld_kontrast = KaydiriciBolum("Kontrast", 0, 200, 100,
            "100 = normal. 200 = 2x kontrast. 0 = gri.")
        self.sld_doygun = KaydiriciBolum("Doygunluk", 0, 200, 100,
            "100 = normal. 0 = gri ton. 200 = super doygun.")
        self.sld_ton = KaydiriciBolum("Renk Tonu", -90, 90, 0,
            "HSV H kanalini kaydirir. 0 = degisiklik yok.")
        g_renk.addWidget(self.sld_parlak)
        g_renk.addWidget(self.sld_kontrast)
        g_renk.addWidget(self.sld_doygun)
        g_renk.addWidget(self.sld_ton)
        sol_layout.addWidget(grp_renk)

        # 05 - CLAHE
        grp_clahe = QGroupBox("05 - CLAHE Kontrast Iyilestirme")
        g_clahe = QVBoxLayout(grp_clahe)
        self.sld_clahe = KaydiriciBolum("CLAHE Clip", 0, 80, 0,
            "0 = kapali. Deger/10 = clipLimit. Adaptif histogram esitleme.")
        g_clahe.addWidget(self.sld_clahe)
        sol_layout.addWidget(grp_clahe)

        # Ekstra: Keskinlestirme
        grp_keskin = QGroupBox("Keskinlestirme (Unsharp Mask)")
        g_keskin = QVBoxLayout(grp_keskin)
        self.sld_keskin = KaydiriciBolum("Keskinlik", 0, 20, 0,
            "Unsharp mask ile keskinlestirme. 0 = kapali.")
        g_keskin.addWidget(self.sld_keskin)
        sol_layout.addWidget(grp_keskin)

        # Goruntuleme modu
        grp_mod = QGroupBox("Goruntuleme")
        g_mod = QVBoxLayout(grp_mod)
        self.cmb_mod = QComboBox()
        self.cmb_mod.addItems([
            "Renkli (BGR)",
            "Gri Ton",
            "Sadece R Kanali",
            "Sadece G Kanali",
            "Sadece B Kanali",
            "HSV - H (Ton)",
            "HSV - S (Doygunluk)",
            "HSV - V (Parlaklik)",
            "Fark Goruntusu (Hata Haritasi)"
        ])
        self.cmb_mod.currentIndexChanged.connect(self._guncelle)
        g_mod.addWidget(self.cmb_mod)
        sol_layout.addWidget(grp_mod)

        # Araclar (Kirpma, Silme, Cizim)
        grp_arac = QGroupBox("Araclar (Kirpma / Silme / Cizim)")
        grp_arac.setStyleSheet(
            "QGroupBox { color: #ce93d8; border-color: #ce93d8; }")
        g_arac = QVBoxLayout(grp_arac)

        arac_buton = QHBoxLayout()
        self.btn_arac_yok = QPushButton("Normal")
        self.btn_arac_yok.clicked.connect(lambda: self._arac_sec("yok"))
        self.btn_arac_kirp = QPushButton("Kirp")
        self.btn_arac_kirp.clicked.connect(lambda: self._arac_sec("kirp"))
        self.btn_arac_sil = QPushButton("Firca Sil")
        self.btn_arac_sil.clicked.connect(lambda: self._arac_sec("sil"))
        self.btn_arac_kement = QPushButton("Kement Sil")
        self.btn_arac_kement.clicked.connect(lambda: self._arac_sec("kement_sil"))
        self.btn_arac_ciz = QPushButton("Ciz")
        self.btn_arac_ciz.clicked.connect(lambda: self._arac_sec("ciz"))
        for b in [self.btn_arac_yok, self.btn_arac_kirp, self.btn_arac_sil, self.btn_arac_kement, self.btn_arac_ciz]:
            b.setStyleSheet(
                "QPushButton { border-color:#ce93d8; color:#ce93d8; padding:6px; }"
                "QPushButton:hover { background-color:#7b1fa2; color:white; }")
        arac_buton.addWidget(self.btn_arac_yok)
        arac_buton.addWidget(self.btn_arac_kirp)
        arac_buton.addWidget(self.btn_arac_sil)
        arac_buton.addWidget(self.btn_arac_kement)
        arac_buton.addWidget(self.btn_arac_ciz)
        g_arac.addLayout(arac_buton)

        self.sld_firca = KaydiriciBolum("Firca Boyutu", 3, 80, 15, "Silme/cizim firca boyutu")
        self.sld_firca.connectChanged(lambda _: self._firca_guncelle())
        g_arac.addWidget(self.sld_firca)

        renk_undo = QHBoxLayout()
        self.btn_renk = QPushButton("Renk Sec")
        self.btn_renk.setStyleSheet(
            "QPushButton { border-color:#66bb6a; color:#66bb6a; }"
            "QPushButton:hover { background-color:#2e7d32; color:white; }")
        self.btn_renk.clicked.connect(self._renk_sec)
        self.btn_geri_al = QPushButton("Geri Al (Ctrl+Z)")
        self.btn_geri_al.setStyleSheet(
            "QPushButton { border-color:#ffcc80; color:#ffcc80; }"
            "QPushButton:hover { background-color:#e65100; color:white; }")
        self.btn_geri_al.clicked.connect(self._geri_al)
        renk_undo.addWidget(self.btn_renk)
        renk_undo.addWidget(self.btn_geri_al)
        g_arac.addLayout(renk_undo)

        self.lbl_arac_durum = QLabel("Aktif arac: Normal")
        self.lbl_arac_durum.setStyleSheet("color:#ce93d8; font-size:11px;")
        self.lbl_arac_durum.setAlignment(Qt.AlignCenter)
        g_arac.addWidget(self.lbl_arac_durum)
        sol_layout.addWidget(grp_arac)

        sol_layout.addStretch()
        sol_scroll.setWidget(sol_widget)
        ana_layout.addWidget(sol_scroll)

        # ── Sag: Goruntu Alani ──────────────────────────────────────────────
        sag_widget = QWidget()
        sag_layout = QVBoxLayout(sag_widget)
        sag_layout.setSpacing(4)

        # Ust etiketler
        ust_etiket = QHBoxLayout()
        self.lbl_orijinal_baslik = QLabel("ORIJINAL")
        self.lbl_orijinal_baslik.setAlignment(Qt.AlignCenter)
        self.lbl_orijinal_baslik.setStyleSheet("font-size:15px; font-weight:bold; color:#4fc3f7;")
        lbl_sonuc_baslik = QLabel("ISLEM SONUCU")
        lbl_sonuc_baslik.setAlignment(Qt.AlignCenter)
        lbl_sonuc_baslik.setStyleSheet("font-size:15px; font-weight:bold; color:#a5d6a7;")
        ust_etiket.addWidget(self.lbl_orijinal_baslik)
        ust_etiket.addWidget(lbl_sonuc_baslik)
        sag_layout.addLayout(ust_etiket)

        # Goruntu etiketleri
        goruntu_layout = QHBoxLayout()
        self.lbl_orijinal = QLabel("Goruntu yuklemek icin\n'Goruntu Ac' butonuna basin")
        self.lbl_orijinal.setObjectName("lbl_goruntu")
        self.lbl_orijinal.setAlignment(Qt.AlignCenter)
        self.lbl_orijinal.setMinimumSize(300, 250)
        self.lbl_orijinal.setStyleSheet(
            "QLabel { background-color: #0d1117; border: 1px solid #3a3a5e; "
            "border-radius: 6px; color: #546e7a; font-size: 14px; }")

        self.lbl_sonuc = GoruntuLabel("Islem sonucu burada gorunecek")
        self.lbl_sonuc.setObjectName("lbl_goruntu")
        self.lbl_sonuc.setAlignment(Qt.AlignCenter)
        self.lbl_sonuc.setMinimumSize(300, 250)
        self.lbl_sonuc.setStyleSheet(
            "QLabel { background-color: #0d1117; border: 1px solid #3a3a5e; "
            "border-radius: 6px; color: #546e7a; font-size: 14px; }")
        self.lbl_sonuc.mousePressed.connect(self._mouse_press)
        self.lbl_sonuc.mouseMoved.connect(self._mouse_move)
        self.lbl_sonuc.mouseReleased.connect(self._mouse_release)

        goruntu_layout.addWidget(self.lbl_orijinal)
        goruntu_layout.addWidget(self.lbl_sonuc)
        sag_layout.addLayout(goruntu_layout, stretch=1)

        # Bilgi satiri
        self.lbl_bilgi = QLabel("")
        self.lbl_bilgi.setAlignment(Qt.AlignCenter)
        self.lbl_bilgi.setStyleSheet(
            "font-size:11px; color:#b0bec5; background-color:#0d1117; "
            "border:1px solid #3a3a5e; border-radius:4px; padding:6px;")
        sag_layout.addWidget(self.lbl_bilgi)

        ana_layout.addWidget(sag_widget, stretch=1)

        # ── Status Bar ──────────────────────────────────────────────────────
        self.statusBar().showMessage("Hazir  |  Bir goruntu yukleyin")

        # ── Tum slider sinyallerini bagla ────────────────────────────────────
        for sld in [self.sld_gauss, self.sld_median, self.sld_ornek,
                    self.sld_bit, self.sld_parlak, self.sld_kontrast,
                    self.sld_doygun, self.sld_ton, self.sld_clahe,
                    self.sld_keskin, self.sld_gurultu, self.sld_tuz_biber]:
            sld.connectChanged(self._guncelle)
        self.chk_tuz_biber.stateChanged.connect(self._guncelle)

    # ── Varsayilan goruntu yukle ────────────────────────────────────────────
    def _varsayilan_yukle(self):
        for dosya in ["ornek.jpg", "araba.jpg"]:
            yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), dosya)
            if os.path.isfile(yol):
                self._goruntu_yukle(yol)
                return

    # ── Kamera islemleri ────────────────────────────────────────────────────
    def _kamera_ac(self):
        """Webcam'i ac ve canli yayina basla."""
        kamera_no = self.cmb_kamera.currentIndex()
        self.kamera = cv2.VideoCapture(kamera_no)

        if not self.kamera.isOpened():
            QMessageBox.warning(self, "Hata",
                f"Kamera {kamera_no} acilamadi!\n"
                "Baska bir kamera numarasi deneyin.")
            self.kamera = None
            return

        # Kamera cozunurlugunu ayarla
        self.kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.kamera_aktif = True
        self.btn_kamera_ac.setEnabled(False)
        self.btn_kamera_kapat.setEnabled(True)
        self.btn_fotograf.setEnabled(True)
        self.btn_ac.setEnabled(False)
        self.lbl_orijinal_baslik.setText("CANLI KAMERA")
        self.lbl_orijinal_baslik.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#ff8a65;")

        # 30 FPS hedefi (~33ms)
        self.kamera_timer.start(33)
        self.statusBar().showMessage(
            f"Kamera {kamera_no} aktif  |  Filtreleri kaydiriclarla ayarlayin")

    def _kamera_kapat(self):
        """Webcam'i kapat."""
        self.kamera_timer.stop()
        self.kamera_aktif = False

        if self.kamera is not None:
            self.kamera.release()
            self.kamera = None

        self.btn_kamera_ac.setEnabled(True)
        self.btn_kamera_kapat.setEnabled(False)
        self.btn_fotograf.setEnabled(False)
        self.btn_ac.setEnabled(True)
        self.lbl_orijinal_baslik.setText("ORIJINAL")
        self.lbl_orijinal_baslik.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#4fc3f7;")
        self.statusBar().showMessage("Kamera kapatildi")

    def _kamera_kare_al(self):
        """Timer ile her karede cagirilir: kameradan kare al ve isle."""
        if self.kamera is None or not self.kamera_aktif:
            return

        ret, kare = self.kamera.read()
        if not ret:
            return

        self.orijinal = kare

        # Orijinal kareyi goster
        ow = self.lbl_orijinal.width() - 10
        oh = self.lbl_orijinal.height() - 10
        self.lbl_orijinal.setPixmap(cv2_to_qpixmap(kare, ow, oh))

        # Islem uygula ve sonucu goster
        self.sonuc = self._islem_uygula(kare)
        gosterilecek = self._mod_uygula(self.sonuc)
        self.lbl_sonuc.setPixmap(cv2_to_qpixmap(gosterilecek, ow, oh))

        # Bilgi guncelle
        self._bilgi_guncelle()

    def _fotograf_cek(self):
        """Mevcut islenmis kareyi kaydet."""
        if self.sonuc is None:
            return

        import time
        dosya_adi = f"fotograf_{int(time.time())}.jpg"
        yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), dosya_adi)
        cv2.imwrite(yol, self.sonuc)
        self.statusBar().showMessage(f"Fotograf kaydedildi: {dosya_adi}")

    def closeEvent(self, event):
        """Pencere kapanirken kamerayi serbest birak."""
        self._kamera_kapat()
        event.accept()

    # ── Dosya ac ────────────────────────────────────────────────────────────
    def _dosya_ac(self):
        yol, _ = QFileDialog.getOpenFileName(
            self, "Goruntu Sec",
            os.path.dirname(os.path.abspath(__file__)),
            "Goruntu Dosyalari (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;Tum Dosyalar (*)")
        if yol:
            self._goruntu_yukle(yol)

    def _goruntu_yukle(self, yol):
        # Kamera aciksa once kapat
        if self.kamera_aktif:
            self._kamera_kapat()

        img = cv2.imread(yol)
        if img is None:
            QMessageBox.warning(self, "Hata", f"Goruntu okunamadi:\n{yol}")
            return

        # Buyuk resimleri kucult
        h, w = img.shape[:2]
        MAX = 900
        if max(h, w) > MAX:
            olcek = MAX / max(h, w)
            img = cv2.resize(img, (int(w * olcek), int(h * olcek)))

        self.orijinal = img
        self.mutlak_orijinal = img.copy()
        self.statusBar().showMessage(
            f"Yuklendi: {os.path.basename(yol)}  |  "
            f"{img.shape[1]}x{img.shape[0]} piksel  |  {img.dtype}")
        self._guncelle()

    # ── Islem pipeline ──────────────────────────────────────────────────────
    def _islem_uygula(self, img):
        sonuc = img.copy()

        # ── Gurultu ekle (test icin) ────────────────────────────────────────
        gurultu_siddeti = self.sld_gurultu.value()
        if gurultu_siddeti > 0:
            noise = np.random.default_rng(42).normal(0, gurultu_siddeti, sonuc.shape)
            sonuc = np.clip(sonuc.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if self.chk_tuz_biber.isChecked() and self.sld_tuz_biber.value() > 0:
            oran = self.sld_tuz_biber.value() / 100.0
            rng = np.random.default_rng(42)
            # Tuz (beyaz)
            tuz = rng.random(sonuc.shape[:2]) < (oran / 2)
            sonuc[tuz] = 255
            # Biber (siyah)
            biber = rng.random(sonuc.shape[:2]) < (oran / 2)
            sonuc[biber] = 0

        # ── 02 - Ornekleme (Sampling) ───────────────────────────────────────
        ornek_val = self.sld_ornek.value()
        if ornek_val > 1:
            sh, sw = sonuc.shape[:2]
            kucuk = cv2.resize(sonuc,
                               (max(1, sw // ornek_val), max(1, sh // ornek_val)),
                               interpolation=cv2.INTER_NEAREST)
            sonuc = cv2.resize(kucuk, (sw, sh), interpolation=cv2.INTER_NEAREST)

        # ── 03 - Kuantalama (Quantization) ──────────────────────────────────
        bit_val = self.sld_bit.value()
        if bit_val < 8:
            levels = 2 ** bit_val
            step = max(1, 256 // levels)
            sonuc = (sonuc // step) * step

        # ── 04 - Renk (HSV) ────────────────────────────────────────────────
        parlak_fark  = self.sld_parlak.value()          # -100..100
        kontrast_kat = self.sld_kontrast.value() / 100  # 0..2
        doygun_kat   = self.sld_doygun.value() / 100    # 0..2
        ton_fark     = self.sld_ton.value()             # -90..90

        hsv = cv2.cvtColor(sonuc, cv2.COLOR_BGR2HSV).astype(np.float32)

        if ton_fark != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + ton_fark) % 180
        if doygun_kat != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * doygun_kat, 0, 255)
        if parlak_fark != 0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + parlak_fark * 1.5, 0, 255)

        sonuc = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if kontrast_kat != 1.0:
            sonuc = np.clip(
                128 + kontrast_kat * (sonuc.astype(np.float32) - 128),
                0, 255).astype(np.uint8)

        # ── 05 - CLAHE ─────────────────────────────────────────────────────
        clahe_val = self.sld_clahe.value()
        if clahe_val > 0:
            clip = clahe_val / 10.0
            lab = cv2.cvtColor(sonuc, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l = clahe.apply(l)
            sonuc = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

        # ── 06 - Gaussian Blur ─────────────────────────────────────────────
        gauss_val = self.sld_gauss.value()
        if gauss_val > 0:
            k = 2 * gauss_val + 1
            sonuc = cv2.GaussianBlur(sonuc, (k, k), 0)

        # ── 06 - Median Blur ──────────────────────────────────────────────
        median_val = self.sld_median.value()
        if median_val > 0:
            k = 2 * median_val + 1
            sonuc = cv2.medianBlur(sonuc, k)

        # ── Keskinlestirme ─────────────────────────────────────────────────
        keskin_val = self.sld_keskin.value()
        if keskin_val > 0:
            bulanik = cv2.GaussianBlur(sonuc, (0, 0), 3)
            miktar = 0.3 + keskin_val * 0.25
            sonuc = cv2.addWeighted(sonuc, 1 + miktar, bulanik, -miktar, 0)

        return sonuc

    # ── Goruntuleme modu ────────────────────────────────────────────────────
    def _mod_uygula(self, img):
        """Secilen goruntuleme moduna gore goruntuyu donustur."""
        mod = self.cmb_mod.currentIndex()
        if mod == 0:    # Renkli
            return img
        elif mod == 1:  # Gri
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif mod == 2:  # R
            out = np.zeros_like(img)
            out[:, :, 2] = img[:, :, 2]
            return out
        elif mod == 3:  # G
            out = np.zeros_like(img)
            out[:, :, 1] = img[:, :, 1]
            return out
        elif mod == 4:  # B
            out = np.zeros_like(img)
            out[:, :, 0] = img[:, :, 0]
            return out
        elif mod == 5:  # HSV-H
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_ch = hsv[:, :, 0]
            return cv2.applyColorMap(h_ch, cv2.COLORMAP_HSV)
        elif mod == 6:  # HSV-S
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR)
        elif mod == 7:  # HSV-V
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR)
        elif mod == 8:  # Fark
            if self.orijinal is not None:
                diff = np.abs(self.orijinal.astype(np.float32) - img.astype(np.float32))
                diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
                return cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)
        return img

    # ── Bilgi guncelle ──────────────────────────────────────────────────────
    def _bilgi_guncelle(self):
        """Alt bilgi satirini guncelle."""
        if self.orijinal is None or self.sonuc is None:
            return

        aktif = []
        if self.sld_gauss.value() > 0:
            k = 2 * self.sld_gauss.value() + 1
            aktif.append(f"Gaussian {k}x{k}")
        if self.sld_median.value() > 0:
            k = 2 * self.sld_median.value() + 1
            aktif.append(f"Median {k}x{k}")
        if self.sld_ornek.value() > 1:
            aktif.append(f"Ornekleme 1/{self.sld_ornek.value()}")
        if self.sld_bit.value() < 8:
            aktif.append(f"{self.sld_bit.value()}-bit")
        if self.sld_clahe.value() > 0:
            aktif.append(f"CLAHE {self.sld_clahe.value()/10:.1f}")
        if self.sld_keskin.value() > 0:
            aktif.append("Keskin")
        if self.sld_parlak.value() != 0:
            aktif.append(f"Parlak {self.sld_parlak.value():+d}")
        if self.sld_kontrast.value() != 100:
            aktif.append(f"Kontrast {self.sld_kontrast.value()}%")
        if self.sld_doygun.value() != 100:
            aktif.append(f"Doygun {self.sld_doygun.value()}%")
        if self.sld_ton.value() != 0:
            aktif.append(f"Ton {self.sld_ton.value():+d}")
        if self.sld_gurultu.value() > 0:
            aktif.append(f"Gurultu {self.sld_gurultu.value()}")

        g_orig = cv2.cvtColor(self.orijinal, cv2.COLOR_BGR2GRAY)
        g_son  = cv2.cvtColor(self.sonuc, cv2.COLOR_BGR2GRAY)
        p = psnr(g_orig, g_son)
        psnr_str = f"PSNR: {p:.1f} dB" if p < 999 else "PSNR: Degisiklik yok"

        h, w = self.orijinal.shape[:2]
        mod = "KAMERA" if self.kamera_aktif else "DOSYA"
        aktif_str = " | ".join(aktif) if aktif else "Aktif filtre yok"
        self.lbl_bilgi.setText(
            f"[{mod}]  {psnr_str}   |   {w}x{h}   |   {aktif_str}")

    # ── Guncelle (dosya modu icin) ─────────────────────────────────────────
    def _guncelle(self, _=None):
        if self.orijinal is None:
            return
        # Kamera modundaysa slider degisikliklerinde sadece bilgi guncelle
        # (kare yakalama timer'da yapiliyor)
        if self.kamera_aktif:
            return

        # Orijinali goster (Daima hic degistirilmemis olan mutlak orijinali gosterir)
        ow = self.lbl_orijinal.width() - 10
        oh = self.lbl_orijinal.height() - 10
        hedef_gorsel = self.mutlak_orijinal if self.mutlak_orijinal is not None else self.orijinal
        self.lbl_orijinal.setPixmap(cv2_to_qpixmap(hedef_gorsel, ow, oh))

        # Islem uygula
        self.sonuc = self._islem_uygula(self.orijinal)
        gosterilecek = self._mod_uygula(self.sonuc)
        self.lbl_sonuc.setPixmap(cv2_to_qpixmap(gosterilecek, ow, oh))

        self._bilgi_guncelle()

    # ── Arac secimi ─────────────────────────────────────────────────────────
    def _arac_sec(self, arac):
        self._aktif_arac = arac
        isimler = {"yok": "Normal", "kirp": "Kirpma", "sil": "Silme (Firca)", "kement_sil": "Akilli Silme (Kement)", "ciz": "Cizim"}
        self.lbl_arac_durum.setText(f"Aktif arac: {isimler[arac]}")
        self.statusBar().showMessage(f"Arac: {isimler[arac]}")
        if arac == "kirp":
            self.statusBar().showMessage(
                "KIRPMA: Sonuc goruntusu uzerinde dikdortgen cizin")
        elif arac in ("sil", "ciz"):
            self.statusBar().showMessage(
                "Sonuc goruntusu uzerinde cizin. Renk secebilirsiniz.")
        elif arac == "kement_sil":
            self.statusBar().showMessage(
                "KEMENT SIL: Istenmeyen nesnenin etrafinda daire/sekil cizin")

    def _firca_guncelle(self):
        self._firca_boyut = self.sld_firca.value()

    def _renk_sec(self):
        renk = QColorDialog.getColor(QColor(*self._firca_renk[::-1]), self, "Firca Rengi Sec")
        if renk.isValid():
            self._firca_renk = (renk.blue(), renk.green(), renk.red())  # BGR
            self.btn_renk.setStyleSheet(
                f"QPushButton {{ border-color:{renk.name()}; color:{renk.name()}; }}"
                f"QPushButton:hover {{ background-color:{renk.name()}; color:white; }}")

    def _undo_kaydet(self):
        """Mevcut orijinali undo yiginina ekle."""
        if self.orijinal is not None:
            if len(self._undo_stack) > 20:
                self._undo_stack.pop(0)
            self._undo_stack.append(self.orijinal.copy())

    def _geri_al(self):
        if self._undo_stack:
            self.orijinal = self._undo_stack.pop()
            self._guncelle()
            self.statusBar().showMessage("Geri alindi")
        else:
            self.statusBar().showMessage("Geri alinacak islem yok")

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self._geri_al()
        super().keyPressEvent(event)

    # ── Widget koordinatini goruntu koordinatina cevir ──────────────────────
    def _widget_to_img(self, wx, wy):
        """GoruntuLabel uzerindeki piksel konumunu orijinal goruntu koordinatina cevir."""
        if self.sonuc is None:
            return None, None
        pix = self.lbl_sonuc.pixmap()
        if pix is None or pix.isNull():
            return None, None
        ih, iw = self.sonuc.shape[:2]
        pw, ph = pix.width(), pix.height()
        ix = int(wx * iw / pw)
        iy = int(wy * ih / ph)
        ix = max(0, min(ix, iw - 1))
        iy = max(0, min(iy, ih - 1))
        return ix, iy

    # ── Mouse olaylari ─────────────────────────────────────────────────────
    def _mouse_press(self, wx, wy):
        if self.orijinal is None:
            return
        ix, iy = self._widget_to_img(wx, wy)
        if ix is None:
            return

        if self._aktif_arac == "kirp":
            self._kirp_baslangic = (ix, iy)
            self._kirp_bitis = None
        elif self._aktif_arac in ("sil", "ciz", "kement_sil"):
            self._undo_kaydet()
            self._cizim_aktif = True
            self._son_cizim_noktasi = (ix, iy)
            if self._aktif_arac == "sil":
                self._islem_oncesi_orijinal = self.orijinal.copy()
                self._silme_maskesi = np.zeros(self.orijinal.shape[:2], dtype=np.uint8)
            elif self._aktif_arac == "kement_sil":
                self._islem_oncesi_orijinal = self.orijinal.copy()
                self._silme_maskesi = np.zeros(self.orijinal.shape[:2], dtype=np.uint8)
                self._kement_noktalari = [(ix, iy)]
                
            if self._aktif_arac in ("sil", "ciz"):
                self._ciz_nokta(ix, iy)

    def _mouse_move(self, wx, wy):
        if self.orijinal is None:
            return
        ix, iy = self._widget_to_img(wx, wy)
        if ix is None:
            return

        if self._aktif_arac == "kirp" and self._kirp_baslangic:
            self._kirp_bitis = (ix, iy)
            self._kirp_onizleme()
        elif self._aktif_arac in ("sil", "ciz") and self._cizim_aktif:
            self._ciz_nokta(ix, iy)
        elif self._aktif_arac == "kement_sil" and self._cizim_aktif:
            self._kement_noktalari.append((ix, iy))
            cv2.line(self.orijinal, self._son_cizim_noktasi, (ix, iy), (0, 0, 255), 2)
            self._son_cizim_noktasi = (ix, iy)
            self.sonuc = self._islem_uygula(self.orijinal)
            ow, oh = self.lbl_sonuc.width() - 10, self.lbl_sonuc.height() - 10
            if ow > 0 and oh > 0:
                self.lbl_sonuc.setPixmap(cv2_to_qpixmap(self.sonuc, ow, oh))

    def _mouse_release(self, wx, wy):
        if self.orijinal is None:
            return
        ix, iy = self._widget_to_img(wx, wy)

        if self._aktif_arac == "kirp" and self._kirp_baslangic:
            if ix is not None:
                self._kirp_bitis = (ix, iy)
            if self._kirp_bitis:
                self._kirp_uygula()
        elif self._aktif_arac in ("sil", "ciz", "kement_sil"):
            if self._cizim_aktif and ix is not None:
                if self._aktif_arac in ("sil", "ciz"):
                    self._ciz_nokta(ix, iy)
                elif self._aktif_arac == "kement_sil":
                    self._kement_noktalari.append((ix, iy))
            self._cizim_aktif = False
            self._son_cizim_noktasi = None

            if self._aktif_arac == "sil" and self._silme_maskesi is not None:
                self.statusBar().showMessage("Siliniyor, lutfen bekleyin... Islem yapiliyor.")
                QApplication.processEvents() # UI'nin guncellenmesini sagla
                # Inpaint: Arka plana uygun sekilde boslugu doldurur
                r = self._firca_boyut // 2
                inpaint_radius = max(3, r)
                self.orijinal = cv2.inpaint(self._islem_oncesi_orijinal, self._silme_maskesi, inpaint_radius, cv2.INPAINT_TELEA)
                self._silme_maskesi = None
                self._islem_oncesi_orijinal = None
                self.statusBar().showMessage("Silme islemi tamamlandi")
            
            elif self._aktif_arac == "kement_sil" and self._silme_maskesi is not None:
                if len(self._kement_noktalari) > 2:
                    self.statusBar().showMessage("Akilli silme islemi (Yapay Zeka Destekli) yapiliyor...")
                    QApplication.processEvents()
                    pts = np.array(self._kement_noktalari, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(self._silme_maskesi, [pts], 255) # Iceriyi tamamen doldurur
                    # Cok daha buyuk bir capta inpaint eder
                    self.orijinal = cv2.inpaint(self._islem_oncesi_orijinal, self._silme_maskesi, 15, cv2.INPAINT_TELEA)
                    self.statusBar().showMessage("Akilli silme tamamlandi")
                else:
                    self.orijinal = self._islem_oncesi_orijinal
                    self.statusBar().showMessage("Kement alani cok kucuk")
                self._silme_maskesi = None
                self._islem_oncesi_orijinal = None
                self._kement_noktalari = []

            self._guncelle()

    def _ciz_nokta(self, ix, iy):
        """Orijinal goruntu uzerine firca ile ciz veya silme maskesi olustur."""
        r = self._firca_boyut // 2
        onceki_nokta = self._son_cizim_noktasi if self._son_cizim_noktasi else (ix, iy)
        kalinlik = max(1, r * 2)

        if self._aktif_arac == "sil":
            # Maskeye isaretle
            cv2.line(self._silme_maskesi, onceki_nokta, (ix, iy), 255, kalinlik)
            # Kullanici ne sildigini gorsun diye gecici kirmizi firca goster
            cv2.line(self.orijinal, onceki_nokta, (ix, iy), (0, 0, 255), kalinlik)
        else:
            # Secilen renk ile surekli cizgi ciz
            cv2.line(self.orijinal, onceki_nokta, (ix, iy), self._firca_renk, kalinlik)
            
        self._son_cizim_noktasi = (ix, iy)

        # Anlik guncelle
        self.sonuc = self._islem_uygula(self.orijinal)
        ow = self.lbl_sonuc.width() - 10
        oh = self.lbl_sonuc.height() - 10
        if ow > 0 and oh > 0:
            self.lbl_sonuc.setPixmap(cv2_to_qpixmap(self.sonuc, ow, oh))

    def _kirp_onizleme(self):
        """Kirpma alanini sonuc goruntusunde dikdortgen olarak goster."""
        if self.sonuc is None or not self._kirp_baslangic or not self._kirp_bitis:
            return
        onizleme = self.sonuc.copy()
        x1, y1 = self._kirp_baslangic
        x2, y2 = self._kirp_bitis
        cv2.rectangle(onizleme, (x1, y1), (x2, y2), (0, 255, 0), 2)
        ow = self.lbl_sonuc.width() - 10
        oh = self.lbl_sonuc.height() - 10
        self.lbl_sonuc.setPixmap(cv2_to_qpixmap(onizleme, ow, oh))

    def _kirp_uygula(self):
        """Secilen alani kirp ve yeni orijinal yap."""
        if not self._kirp_baslangic or not self._kirp_bitis:
            return
        x1, y1 = self._kirp_baslangic
        x2, y2 = self._kirp_bitis
        xa, xb = min(x1, x2), max(x1, x2)
        ya, yb = min(y1, y2), max(y1, y2)
        if xb - xa < 10 or yb - ya < 10:
            self.statusBar().showMessage("Kirpma alani cok kucuk")
            self._kirp_baslangic = None
            self._kirp_bitis = None
            return
        self._undo_kaydet()
        self.orijinal = self.orijinal[ya:yb, xa:xb].copy()
        self._kirp_baslangic = None
        self._kirp_bitis = None
        self._guncelle()
        h, w = self.orijinal.shape[:2]
        self.statusBar().showMessage(f"Kirpildi: {w}x{h} piksel")

    # ── Sifirla ─────────────────────────────────────────────────────────────
    def _sifirla(self):
        self.sld_gauss.setValue(0)
        self.sld_median.setValue(0)
        self.sld_ornek.setValue(1)
        self.sld_bit.setValue(8)
        self.sld_parlak.setValue(0)
        self.sld_kontrast.setValue(100)
        self.sld_doygun.setValue(100)
        self.sld_ton.setValue(0)
        self.sld_clahe.setValue(0)
        self.sld_keskin.setValue(0)
        self.sld_gurultu.setValue(0)
        self.sld_tuz_biber.setValue(0)
        self.chk_tuz_biber.setChecked(False)
        self.cmb_mod.setCurrentIndex(0)
        self._arac_sec("yok")

        if self.mutlak_orijinal is not None:
            self.orijinal = self.mutlak_orijinal.copy()
            self._undo_stack.clear()
            self._guncelle()
        
        self.statusBar().showMessage("Tum ayarlar sifirlandi ve orijinal goruntu geri yuklendi")

    # ── Kaydet ──────────────────────────────────────────────────────────────
    def _kaydet(self):
        if self.sonuc is None:
            QMessageBox.information(self, "Bilgi", "Kaydedilecek goruntu yok.")
            return
        yol, _ = QFileDialog.getSaveFileName(
            self, "Sonucu Kaydet",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuc.jpg"),
            "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp);;Tum Dosyalar (*)")
        if yol:
            cv2.imwrite(yol, self.sonuc)
            self.statusBar().showMessage(f"Kaydedildi: {yol}")

    # ── Pencere boyutu degisince goruntuyu yeniden olcekle ──────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self._guncelle)


# ── Calistir ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(KOYU_TEMA)

    # Uygulama fontu
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    pencere = FotoShopPenceresi()
    pencere.show()
    sys.exit(app.exec_())
