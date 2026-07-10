"""Shared, package-wide constants and default paths.

The single source of truth for archive endpoints, default timeouts, and the
YAML config files used by the archive and UI layers. Mirrors the pattern of
``xleap_parser/beamlines.py``: one immutable knob per constant, referenced by
every subpackage instead of scattered module-level defaults.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo


CONFIG_DIR: Path = Path(__file__).parent / "config"
DEFAULT_PV_GROUPS_YAML: Path = CONFIG_DIR / "pv_groups.yaml"
DEFAULT_MONITOR_PVS_YAML: Path = CONFIG_DIR / "monitor_pvs.yaml"

ARCHIVER_URL: str = "http://lcls-archapp.slac.stanford.edu/retrieval/data/getData.json"
ARCHIVE_TIMEOUT_SECONDS: float = 20.0
ARCHIVE_MAX_WORKERS: int = 8
LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")
