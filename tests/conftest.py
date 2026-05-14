import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPARKLINES_ROOT = ROOT / "sparklines"

os.environ["QT_API"] = "PyQt5"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if str(SPARKLINES_ROOT) not in sys.path:
    sys.path.insert(0, str(SPARKLINES_ROOT))

import PyQt5  # noqa: F401
import matplotlib as mpl

# Matplotlib's Qt backend checks for an X/Wayland display before honoring Qt's
# offscreen platform. The CI/test environment here is intentionally headless.
try:
    mpl._c_internal_utils.display_is_valid = lambda: True
except AttributeError:
    pass


collect_ignore = [
    "archive/test_http_archive_server.py",
    "archive/test_pva_archive_server.py",
    "names/test_ds_server.py",
    "model/test_model_server.py",
]
