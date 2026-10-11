"""Audiobook Studio Pro — Application entry point.

Features: splash screen, pre-flight FFmpeg check, single instance guard,
Catppuccin Mocha/Latte theme support, dependency checking.
"""

import logging
import os
import sys
import tempfile

from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QColor, QPainter, QFont

from config import AppConfig
from styles import get_stylesheet
from main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audiobook")

APP_VERSION = "5.0.0"


# ══════════════════════════════════════════════════════════════════
#  Splash Screen
# ══════════════════════════════════════════════════════════════════

def _create_splash() -> QSplashScreen:
    """Generate a branded splash screen."""
    pm = QPixmap(480, 280)
    pm.fill(QColor("#1e1e2e"))

    painter = QPainter(pm)

    # Title
    painter.setPen(QColor("#89b4fa"))
    painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
    painter.drawText(pm.rect(), Qt.AlignCenter, "Audiobook Studio Pro")

    # Version
    painter.setPen(QColor("#6c7086"))
    painter.setFont(QFont("Segoe UI", 11))
    sub_rect = pm.rect()
    sub_rect.moveTop(sub_rect.top() + 40)
    painter.drawText(sub_rect, Qt.AlignCenter, f"Version {APP_VERSION}")

    # Loading text
    painter.setPen(QColor("#45475a"))
    painter.setFont(QFont("Segoe UI", 10))
    load_rect = pm.rect()
    load_rect.moveTop(load_rect.top() + 80)
    painter.drawText(load_rect, Qt.AlignCenter, "Loading...")

    painter.end()

    splash = QSplashScreen(pm)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
    return splash


# ══════════════════════════════════════════════════════════════════
#  Pre-flight Checks
# ══════════════════════════════════════════════════════════════════

def _check_ffmpeg() -> tuple[bool, str]:
    """Check if FFmpeg is available."""
    from audio_postprocessor import AudioPostProcessor
    return AudioPostProcessor.check_ffmpeg()


def _check_dependencies() -> list[str]:
    """Check for optional dependencies and return list of missing ones."""
    missing = []
    try:
        import pypdf
    except ImportError:
        missing.append("pypdf (PDF support)")
    try:
        import docx
    except ImportError:
        missing.append("python-docx (DOCX support)")
    try:
        import bs4
    except ImportError:
        missing.append("beautifulsoup4 (HTML/EPUB support)")
    try:
        import ebooklib
    except ImportError:
        missing.append("ebooklib (EPUB support)")
    try:
        import spacy
    except ImportError:
        missing.append("spacy (NLP speaker detection)")
    try:
        import edge_tts
    except ImportError:
        missing.append("edge-tts (Edge TTS backend)")
    try:
        import aiohttp
    except ImportError:
        missing.append("aiohttp (DramaBox API backend)")
    return missing


# ══════════════════════════════════════════════════════════════════
#  Main Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    # ── Single instance guard ───────────────────────────────────
    lock_path = os.path.join(tempfile.gettempdir(), "audiobook_studio.lock")
    lock_file = None
    try:
        import fcntl
        lock_file = open(lock_path, "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError):
        pass

    config = AppConfig.load()

    # ── Apply log level ─────────────────────────────────────────
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger("audiobook").setLevel(log_level)

    # ── Create Application ─────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyleSheet(get_stylesheet(config.theme))
    app.setStyle("Fusion")
    app.setApplicationName("Audiobook Studio Pro")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Audiobook Studio")

    # ── Splash Screen ──────────────────────────────────────────
    splash = _create_splash()
    splash.show()
    app.processEvents()

    # ── Pre-flight Checks ──────────────────────────────────────
    splash.showMessage("Checking FFmpeg...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#6c7086"))
    app.processEvents()

    ffmpeg_ok, ffmpeg_msg = _check_ffmpeg()
    if not ffmpeg_ok and config.verify_ffmpeg_on_start:
        splash.close()
        QMessageBox.critical(
            None, "FFmpeg Required",
            f"Audiobook Studio Pro requires FFmpeg for audio processing.\n\n"
            f"{ffmpeg_msg}\n\n"
            f"Please install FFmpeg and ensure it is on your system PATH.\n"
            f"Download: https://ffmpeg.org/download.html"
        )
        sys.exit(1)

    splash.showMessage("Checking dependencies...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#6c7086"))
    app.processEvents()

    missing = _check_dependencies()
    if missing:
        logger.warning("Optional dependencies missing: %s", ", ".join(missing))

    # ── Create Main Window ─────────────────────────────────────
    splash.showMessage("Loading interface...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#6c7086"))
    app.processEvents()

    window = MainWindow(config)

    if missing and not hasattr(app, '_deps_warned'):
        app._deps_warned = True
        QTimer.singleShot(500, lambda: QMessageBox.warning(
            window, "Optional Dependencies",
            "Some optional dependencies are missing:\n\n- " +
            "\n- ".join(missing) +
            "\n\nInstall them with pip for full functionality."
        ))

    splash.finish(window)
    window.show()

    ret = app.exec()

    window.save_state()
    config.save()

    if lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
        except Exception:
            pass

    return ret


if __name__ == "__main__":
    sys.exit(main())
