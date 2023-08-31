""" Package initialization. """
from pathlib import Path

__version__ = "0.3.0"
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent.parent.resolve()

__all__ = ["__version__", "APP_DIR", "PROJECT_DIR"]
