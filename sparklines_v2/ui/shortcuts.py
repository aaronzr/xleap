"""Shortcut inventories displayed by the Qt help dialog."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


def toolbar_shortcuts() -> tuple[Path, list[tuple[str, str]]]:
    """Return toolbar shortcuts from the active matplotlibrc file."""
    rc_path = Path(mpl.matplotlib_fname())
    shortcut_map = {
        "home": "Reset original view",
        "back": "Back to previous view",
        "forward": "Forward to next view",
        "pan": "Pan axes",
        "zoom": "Zoom to rectangle",
    }
    resolved = {key: "Unavailable" for key in shortcut_map}
    try:
        for raw_line in rc_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                line = line[1:].strip()
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith("keymap.") or ":" not in line:
                continue
            name, value = line.split(":", 1)
            key = name.removeprefix("keymap.").strip()
            if key in resolved:
                resolved[key] = value.strip() or "Unavailable"
    except OSError:
        pass
    return rc_path, [(label, resolved[key]) for key, label in shortcut_map.items()]


def viewer_shortcuts() -> list[tuple[str, str]]:
    """Return the static viewer shortcuts shown by the Qt help dialog."""
    return [
        ("Back one level", "backspace, left"),
        ("Go home", "h"),
        ("Toggle raw samples", "Show data points checkbox"),
        ("Filter Beam_Path", "SXR/HXR and Cu/SC checkboxes"),
    ]


def canvas_help_items() -> list[tuple[str, str]]:
    """Return canvas guidance moved out of the plot header."""
    return [
        (
            "All PV groups",
            "Click a plot or blue y-label to descend. Use the Beam_Path "
            "checkboxes to filter the current view.",
        ),
        (
            "PV group",
            "Click a PV plot, trace, legend label, or monitor plot to isolate "
            "a single PV. Use Show data points and Beam_Path filters to "
            "refine the view.",
        ),
        (
            "Single PV",
            "Use Show data points, Beam_Path filters, or the breadcrumb path "
            "to navigate.",
        ),
        (
            "No matching data",
            "Use the Beam_Path checkboxes to expand or narrow the current view.",
        ),
    ]
