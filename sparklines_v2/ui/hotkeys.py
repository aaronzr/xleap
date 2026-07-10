"""Qt hotkey glue for the MEME sparklines desktop app."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib as mpl
from PyQt5 import QtCore, QtGui, QtWidgets


_TOOLBAR_KEYMAPS: dict[str, tuple[str, ...]] = {
    "reset_original_view": ("keymap.home",),
    "toolbar_back": ("keymap.back",),
    "toolbar_forward": ("keymap.forward",),
    "toolbar_pan": ("keymap.pan",),
    "toolbar_zoom": ("keymap.zoom",),
}

_VIEWER_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "viewer_back": ("Backspace", "Left"),
    "viewer_home": ("H",),
    "reset_original_view": ("Home", "R"),
}

_SHADOWED_TOOLBAR_KEYS: dict[str, set[str]] = {
    "reset_original_view": {"h"},
    "toolbar_back": {"backspace", "left"},
}

_SPECIAL_KEYS: dict[str, str] = {
    "backspace": "Backspace",
    "down": "Down",
    "enter": "Enter",
    "esc": "Esc",
    "escape": "Esc",
    "home": "Home",
    "left": "Left",
    "pagedown": "PgDown",
    "pageup": "PgUp",
    "return": "Return",
    "right": "Right",
    "space": "Space",
    "tab": "Tab",
    "up": "Up",
}

_MODIFIER_KEYS: dict[str, str] = {
    "alt": "Alt",
    "cmd": "Meta",
    "command": "Meta",
    "control": "Ctrl",
    "ctrl": "Ctrl",
    "meta": "Meta",
    "shift": "Shift",
    "super": "Meta",
}


class SparklineHotkeyController(QtCore.QObject):
    """Capture app-level key presses and dispatch viewer or toolbar actions."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._shortcut_actions: dict[str, Callable[[], None]] = {}
        self._register_shortcuts()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):  # pragma: no cover - Qt runtime path
        if event.type() != QtCore.QEvent.KeyPress:
            return super().eventFilter(obj, event)
        if not self._window.isVisible() or not self._window.isActiveWindow():
            return super().eventFilter(obj, event)
        if self._focus_widget_accepts_text():
            return super().eventFilter(obj, event)

        sequence = QtGui.QKeySequence(int(event.modifiers()) | event.key())
        normalized = sequence.toString(QtGui.QKeySequence.PortableText)
        action = self._shortcut_actions.get(normalized)
        if action is None:
            return super().eventFilter(obj, event)

        action()
        event.accept()
        return True

    def _register_shortcuts(self) -> None:
        self._bind_sequences("viewer_back", _VIEWER_SHORTCUTS["viewer_back"])
        self._bind_sequences("viewer_home", _VIEWER_SHORTCUTS["viewer_home"])
        self._bind_sequences(
            "reset_original_view", _VIEWER_SHORTCUTS["reset_original_view"]
        )

        for action_name, rc_params in _TOOLBAR_KEYMAPS.items():
            tokens: list[str] = []
            for rc_param in rc_params:
                tokens.extend(mpl.rcParams.get(rc_param, ()))
            self._bind_sequences(
                action_name,
                self._normalize_matplotlib_tokens(
                    tokens,
                    shadowed=_SHADOWED_TOOLBAR_KEYS.get(action_name, set()),
                ),
            )

    def _bind_sequences(self, action_name: str, sequences) -> None:
        action = getattr(self, f"_handle_{action_name}", None)
        if action is None:
            return
        for sequence in sequences:
            normalized = QtGui.QKeySequence(sequence).toString(
                QtGui.QKeySequence.PortableText
            )
            if normalized:
                self._shortcut_actions[normalized] = action

    def _normalize_matplotlib_tokens(
        self,
        tokens,
        *,
        shadowed: set[str],
    ) -> list[str]:
        normalized: list[str] = []
        for token in tokens:
            sequence = self._matplotlib_token_to_sequence(token)
            if sequence is None:
                continue
            if token.strip().lower() in shadowed:
                continue
            normalized.append(sequence)
        return normalized

    def _matplotlib_token_to_sequence(self, token: str) -> str | None:
        raw = token.strip()
        if not raw:
            return None
        if raw.lower().startswith("mousebutton."):
            return None

        parts = [part.strip() for part in raw.split("+") if part.strip()]
        if not parts:
            return None

        modifiers = [_MODIFIER_KEYS.get(part.lower()) for part in parts[:-1]]
        if any(modifier is None for modifier in modifiers):
            return None

        key = self._normalize_key_name(parts[-1])
        if key is None:
            return None
        return "+".join([*modifiers, key]) if modifiers else key

    def _normalize_key_name(self, token: str) -> str | None:
        lowered = token.lower()
        if lowered in _SPECIAL_KEYS:
            return _SPECIAL_KEYS[lowered]
        if len(token) == 1:
            return token.upper() if token.isalpha() else token
        if lowered.startswith("f") and lowered[1:].isdigit():
            return lowered.upper()
        return None

    def _focus_widget_accepts_text(self) -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        if widget is None:
            return False
        if isinstance(
            widget,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
            ),
        ):
            return True
        if isinstance(widget, QtWidgets.QComboBox) and widget.isEditable():
            return True
        return bool(widget.inputMethodHints() & QtCore.Qt.ImhMultiLine)

    def _toolbar(self):
        return getattr(self._window, "toolbar", None)

    def _viewer(self):
        return getattr(self._window, "viewer", None)

    def _handle_viewer_back(self) -> None:
        viewer = self._viewer()
        if viewer is not None:
            viewer.back()

    def _handle_viewer_home(self) -> None:
        viewer = self._viewer()
        if viewer is not None:
            viewer.home()

    def _handle_reset_original_view(self) -> None:
        viewer = self._viewer()
        if viewer is not None:
            viewer._trigger_toolbar_home()
            return
        toolbar = self._toolbar()
        if toolbar is not None:
            toolbar.home()

    def _handle_toolbar_back(self) -> None:
        toolbar = self._toolbar()
        if toolbar is not None:
            toolbar.back()

    def _handle_toolbar_forward(self) -> None:
        toolbar = self._toolbar()
        if toolbar is not None:
            toolbar.forward()

    def _handle_toolbar_pan(self) -> None:
        toolbar = self._toolbar()
        if toolbar is not None:
            toolbar.pan()

    def _handle_toolbar_zoom(self) -> None:
        toolbar = self._toolbar()
        if toolbar is not None:
            toolbar.zoom()
