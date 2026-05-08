"""PyQt5 desktop app for the MEME sparkline hierarchy viewer."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
import traceback
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

try:
    from .hotkeys import SparklineHotkeyController
    from .sparklines_hierarchy import build_default_composite_hierarchy
    from .sparklines_viewer import HierarchySparklineViewer
except ImportError:  # pragma: no cover - script fallback
    from hotkeys import SparklineHotkeyController
    from sparklines_hierarchy import build_default_composite_hierarchy
    from sparklines_viewer import HierarchySparklineViewer


Signal = getattr(QtCore, "Signal", QtCore.pyqtSignal)
Slot = getattr(QtCore, "Slot", QtCore.pyqtSlot)
DEFAULT_DRAW_REPORT_PATH = Path(__file__).with_name("sparklines_draw_report.txt")
DEFAULT_WINDOW_HOURS = 8.0
DEFAULT_ELOGBOOK = "lcls2"
DEFAULT_ELOG_TITLE = "MEME Sparklines"


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
    """Return the static viewer shortcuts shown by the Matplotlib help overlay."""
    return [
        ("Back one level", "backspace, left"),
        ("Go home", "h"),
        ("Toggle raw samples", "Show data points checkbox"),
        ("Filter Beam_Path", "SXR/HXR and Cu/SC checkboxes"),
        ("Toggle help", "?, shift+/"),
    ]


def parse_datetime_text(value: str) -> dt.datetime:
    """Parse an ISO-like datetime string into a naive local datetime."""
    text = value.strip()
    if not text:
        raise ValueError("Datetime cannot be empty.")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed.replace(microsecond=0)


def format_datetime_text(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat(sep=" ")


def resolve_time_range(
    *,
    start_text: str | None,
    end_text: str | None,
    hours: float = DEFAULT_WINDOW_HOURS,
    now: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    """Resolve CLI or UI inputs into an absolute time range."""
    window = dt.timedelta(hours=float(hours))
    current = (now or dt.datetime.now()).replace(microsecond=0)

    start = parse_datetime_text(start_text) if start_text else None
    end = parse_datetime_text(end_text) if end_text else None

    if start is None and end is None:
        end = current
        start = end - window
    elif start is None:
        start = end - window
    elif end is None:
        end = start + window

    if start >= end:
        raise ValueError("Start time must be earlier than end time.")
    return start, end


class HierarchyLoadWorker(QtCore.QObject):
    """Load the hierarchy in a background Qt thread."""

    loaded = Signal(object, object, object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, *, start_time, end_time, pv_groups_path=None, builder=None):
        super().__init__()
        self._start_time = start_time
        self._end_time = end_time
        self._pv_groups_path = pv_groups_path
        self._builder = builder or build_default_composite_hierarchy

    @Slot()
    def run(self) -> None:
        try:
            hierarchy = self._builder(
                self._start_time,
                self._end_time,
                pv_groups_path=self._pv_groups_path,
            )
        except Exception:  # pragma: no cover - runtime GUI error path
            self.error.emit(traceback.format_exc())
        else:
            self.loaded.emit(hierarchy, self._start_time, self._end_time)
        finally:
            self.finished.emit()


class SparklineHelpDialog(QtWidgets.QDialog):
    """Qt help dialog mirroring the Matplotlib keyboard help overlay."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MEME Sparklines Help")
        self.resize(720, 420)

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Keyboard Shortcuts")
        title.setStyleSheet("QLabel { font-size: 18px; font-weight: 600; }")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Built-in toolbar shortcuts are read from the active matplotlibrc file."
        )
        subtitle.setWordWrap(True)
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(subtitle)

        toolbar_group = QtWidgets.QGroupBox("Toolbar")
        toolbar_layout = QtWidgets.QVBoxLayout(toolbar_group)
        self.toolbar_table = self._build_shortcut_table()
        toolbar_layout.addWidget(self.toolbar_table)
        layout.addWidget(toolbar_group, 1)

        viewer_group = QtWidgets.QGroupBox("Viewer")
        viewer_layout = QtWidgets.QVBoxLayout(viewer_group)
        self.viewer_table = self._build_shortcut_table()
        viewer_layout.addWidget(self.viewer_table)
        layout.addWidget(viewer_group, 1)

        self.source_label = QtWidgets.QLabel("")
        self.source_label.setStyleSheet("QLabel { color: #666; }")
        self.source_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.source_label)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)

        self.refresh_contents()

    def _build_shortcut_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, 2, self)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setFocusPolicy(QtCore.Qt.NoFocus)
        table.setAlternatingRowColors(False)
        table.setShowGrid(False)
        table.setWordWrap(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        table.setStyleSheet(
            "QTableWidget { background: transparent; border: none; }"
            "QHeaderView::section { background: transparent; border: none; "
            "font-weight: 600; padding: 2px 6px; }"
            "QTableWidget::item { padding: 4px 6px; }"
        )
        return table

    def _populate_shortcut_table(
        self, table: QtWidgets.QTableWidget, shortcuts: list[tuple[str, str]]
    ) -> None:
        table.setRowCount(len(shortcuts))
        for row, (action, shortcut) in enumerate(shortcuts):
            action_item = QtWidgets.QTableWidgetItem(action)
            shortcut_item = QtWidgets.QTableWidgetItem(shortcut)
            action_item.setFlags(QtCore.Qt.ItemIsEnabled)
            shortcut_item.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(row, 0, action_item)
            table.setItem(row, 1, shortcut_item)
        table.resizeRowsToContents()

    def refresh_contents(self) -> None:
        rc_path, toolbar_items = toolbar_shortcuts()
        self._populate_shortcut_table(self.toolbar_table, toolbar_items)
        self._populate_shortcut_table(self.viewer_table, viewer_shortcuts())
        self.source_label.setText(f"Source: {rc_path}")


class SparklineMainWindow(QtWidgets.QMainWindow):
    """Qt shell around the Matplotlib-based hierarchy viewer."""

    def __init__(
        self,
        *,
        start_time: dt.datetime,
        end_time: dt.datetime,
        pv_groups_path: str | Path | None = None,
        autoload: bool = True,
        hierarchy_builder=None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("MEME Sparklines")
        self.resize(1600, 1100)

        self._pv_groups_path = None if pv_groups_path is None else Path(pv_groups_path)
        self._hierarchy_builder = hierarchy_builder or build_default_composite_hierarchy
        self._loader_thread = None
        self._loader_worker = None
        self._loaded_hierarchy = None
        self._start_time = start_time
        self._end_time = end_time
        self.viewer = None
        self._help_dialog = None
        self._hotkeys = None

        self._build_ui()
        self._hotkeys = SparklineHotkeyController(self)
        self.start_edit.setText(format_datetime_text(start_time))
        self.end_edit.setText(format_datetime_text(end_time))

        if autoload:
            QtCore.QTimer.singleShot(0, self.reload_requested_range)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)
        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        top_bar.addWidget(self.help_button)
        top_bar.addStretch(1)
        self.elog_button = QtWidgets.QPushButton("Upload to LCLS-II Elog")
        self.elog_button.setStyleSheet(
            "QPushButton { background-color: rgb(85, 255, 255); font-weight: 600; }"
        )
        self.elog_button.clicked.connect(self.upload_canvas_to_elog)
        top_bar.addWidget(self.elog_button)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        self.start_edit = QtWidgets.QLineEdit()
        self.start_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.end_edit = QtWidgets.QLineEdit()
        self.end_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.reload_button = QtWidgets.QPushButton("Reload")
        self.status_label = QtWidgets.QLabel("Idle")
        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)

        controls.addWidget(QtWidgets.QLabel("Start"))
        controls.addWidget(self.start_edit, 1)
        controls.addWidget(QtWidgets.QLabel("End"))
        controls.addWidget(self.end_edit, 1)
        controls.addWidget(self.reload_button)

        layout.addLayout(top_bar)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addWidget(self.summary_label)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(12, 8), dpi=100))
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.toolbar = self.toolbar
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolbar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.toolbar.locLabel.setVisible(False)
        self.toolbar.locLabel.setMinimumWidth(0)
        self.toolbar.locLabel.setMaximumWidth(0)
        self.toolbar.setFixedWidth(self.toolbar.sizeHint().width())

        canvas_row = QtWidgets.QHBoxLayout()
        canvas_row.setContentsMargins(0, 0, 0, 0)
        canvas_row.setSpacing(6)
        canvas_row.addWidget(self.toolbar)
        canvas_row.addWidget(self.canvas, 1)

        layout.addLayout(canvas_row, 1)
        self.setCentralWidget(root)

        self.reload_button.clicked.connect(self.reload_requested_range)
        self.start_edit.returnPressed.connect(self.reload_requested_range)
        self.end_edit.returnPressed.connect(self.reload_requested_range)
        self._refresh_elog_button_state()

    def _set_busy(self, busy: bool) -> None:
        self.reload_button.setEnabled(not busy)
        self.start_edit.setEnabled(not busy)
        self.end_edit.setEnabled(not busy)
        self._refresh_elog_button_state(busy=busy)

    def _refresh_elog_button_state(self, *, busy: bool | None = None) -> None:
        if busy is None:
            busy = bool(self._loader_thread is not None and self._loader_thread.isRunning())
        has_figure = self.viewer is not None and self._loaded_hierarchy is not None
        self.elog_button.setEnabled(has_figure and not busy)

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_label.setText(text)
        if error:
            self.status_label.setStyleSheet(
                "QLabel { color: #b00020; font-weight: 600; }"
            )
        else:
            self.status_label.setStyleSheet("")

    @Slot()
    def reload_requested_range(self) -> None:
        try:
            start_time, end_time = resolve_time_range(
                start_text=self.start_edit.text(),
                end_text=self.end_edit.text(),
            )
        except ValueError as exc:
            self._set_status(f"Invalid time range: {exc}", error=True)
            return

        self._start_time = start_time
        self._end_time = end_time
        self._start_load(start_time, end_time)

    def _start_load(self, start_time: dt.datetime, end_time: dt.datetime) -> None:
        if self._loader_thread is not None and self._loader_thread.isRunning():
            self._set_status("A load is already in progress.", error=True)
            return

        self._set_busy(True)
        self._set_status(
            f"Loading archive data for {format_datetime_text(start_time)} to "
            f"{format_datetime_text(end_time)}..."
        )
        self.summary_label.setText("")

        self._loader_thread = QtCore.QThread(self)
        self._loader_worker = HierarchyLoadWorker(
            start_time=start_time,
            end_time=end_time,
            pv_groups_path=self._pv_groups_path,
            builder=self._hierarchy_builder,
        )
        self._loader_worker.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.loaded.connect(self._on_hierarchy_loaded)
        self._loader_worker.error.connect(self._on_load_error)
        self._loader_worker.finished.connect(self._loader_thread.quit)
        self._loader_worker.finished.connect(self._loader_worker.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.finished.connect(self._on_loader_finished)
        self._loader_thread.start()

    @Slot()
    def _on_loader_finished(self) -> None:
        self._loader_thread = None
        self._loader_worker = None
        self._set_busy(False)

    @Slot(object, object, object)
    def _on_hierarchy_loaded(self, hierarchy, start_time, end_time) -> None:
        self._apply_hierarchy(hierarchy, start_time, end_time)

    @Slot(str)
    def _on_load_error(self, error_text: str) -> None:
        self._set_status("Failed to load hierarchy.", error=True)
        self.summary_label.setText(error_text)

    def _apply_hierarchy(
        self,
        hierarchy: dict,
        start_time: dt.datetime,
        end_time: dt.datetime,
    ) -> None:
        self._loaded_hierarchy = hierarchy
        self._start_time = start_time
        self._end_time = end_time

        self.viewer = HierarchySparklineViewer(
            hierarchy,
            start_time,
            end_time,
            draw_report_path=DEFAULT_DRAW_REPORT_PATH,
            figure=self.canvas.figure,
        )
        self.viewer.draw()
        self.canvas.draw()

        timing = hierarchy.get("timing", {})
        skipped = len(hierarchy.get("skipped_pvs", {}))
        pv_cache = len(hierarchy.get("pv_cache", {}))
        groups = len(hierarchy.get("groups", {}))
        monitors = len(hierarchy.get("monitor_pvs", {}))
        summary = (
            f"Loaded {pv_cache} PVs across {groups} groups and {monitors} monitors. "
            f"Skipped {skipped} PVs. "
            f"Fetch: {timing.get('fetch_wall_seconds', 0.0):.2f}s. "
            f"Build: {timing.get('build_wall_seconds', 0.0):.2f}s."
        )
        self._set_status("Hierarchy loaded.")
        self.summary_label.setText(summary)
        self._refresh_elog_button_state()
        QtCore.QTimer.singleShot(0, self._redraw_viewer_to_window_size)

    @Slot()
    def _redraw_viewer_to_window_size(self) -> None:
        if self.viewer is None:
            return
        self.viewer.draw()
        self.canvas.draw_idle()

    @Slot()
    def upload_canvas_to_elog(self) -> None:
        if self.viewer is None or self._loaded_hierarchy is None:
            self._set_status("Load hierarchy data before uploading to the elog.", error=True)
            return

        try:
            import physicselog  # type: ignore
        except ImportError as exc:
            self._set_status(
                f"physicselog is unavailable in this environment: {exc}",
                error=True,
            )
            return

        self.elog_button.setEnabled(False)
        self._set_status("Uploading current canvas to the LCLS-II elog...")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="meme_sparklines_",
                suffix=".png",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            self.canvas.figure.savefig(temp_path)
            physicselog.submit_entry(
                DEFAULT_ELOGBOOK,
                "ops",
                DEFAULT_ELOG_TITLE,
                attachment=str(temp_path),
            )
        except Exception as exc:
            self._set_status(f"Failed to upload canvas to the elog: {exc}", error=True)
        else:
            self._set_status("Uploaded current canvas to the LCLS-II elog.")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            self._refresh_elog_button_state()

    @Slot()
    def show_help_dialog(self) -> None:
        if self._help_dialog is None:
            self._help_dialog = SparklineHelpDialog(self)
        self._help_dialog.refresh_contents()
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI lifecycle
        if self._loader_thread is not None:
            self._loader_thread.quit()
            self._loader_thread.wait(250)
        super().closeEvent(event)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the MEME sparklines desktop app.")
    parser.add_argument("--start", help="Start time in ISO format, e.g. 2026-04-10 00:00:00")
    parser.add_argument("--end", help="End time in ISO format, e.g. 2026-04-10 08:00:00")
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_WINDOW_HOURS,
        help="Default window size when only one or neither endpoint is provided.",
    )
    parser.add_argument(
        "--pv-groups",
        dest="pv_groups_path",
        help="Optional override path for pv_groups.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        start_time, end_time = resolve_time_range(
            start_text=args.start,
            end_text=args.end,
            hours=args.hours,
        )
    except ValueError as exc:
        parser.error(str(exc))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("MEME Sparklines")
    win = SparklineMainWindow(
        start_time=start_time,
        end_time=end_time,
        pv_groups_path=args.pv_groups_path,
    )
    win.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return exec_fn()


if __name__ == "__main__":
    raise SystemExit(main())
