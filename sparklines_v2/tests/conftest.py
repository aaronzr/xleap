"""Test session setup: ensure sparklines_v2 is importable in headless Qt mode."""

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

os.environ["QT_API"] = "PyQt5"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import PyQt5  # noqa: F401,E402
import matplotlib as mpl  # noqa: E402

# Matplotlib's Qt backend checks for an X/Wayland display before honoring Qt's
# offscreen platform. The CI/test environment here is intentionally headless.
try:
    mpl._c_internal_utils.display_is_valid = lambda: True
except AttributeError:
    pass
