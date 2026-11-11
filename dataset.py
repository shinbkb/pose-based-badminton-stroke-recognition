"""Compatibility import; new code should use badminton_stroke.dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from badminton_stroke.dataset import *  # noqa: F401,F403,E402
