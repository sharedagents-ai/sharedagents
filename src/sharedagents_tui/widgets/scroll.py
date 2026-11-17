"""TranscriptScroll — scroll-position-reporting VerticalScroll (from app.py)."""

from __future__ import annotations

from textual.containers import VerticalScroll


class TranscriptScroll(VerticalScroll):
    """VerticalScroll subclass that reports scroll-position changes to the App.

    Textual 8.2.7 stops ``MouseScrollDown`` / ``MouseScrollUp`` at the
    scrolled widget so those events never bubble to the App.  Instead we
    observe ``scroll_y`` (a reactive on ``Widget``) — this fires for
    wheel, touchpad, scrollbar-drag, PageUp/PageDown, and programmatic
    scrolls in a single code path.
    """

    def watch_scroll_y(self, old: float, new: float) -> None:
        """Notify the App whenever the transcript scroll position changes."""
        super().watch_scroll_y(old, new)
        # Unwrap through the ComposeResult shim:
        #   ComposeResult → _ScrollContainer → this TranscriptScroll
        # The App is the nearest ancestor of type ChatApp.
        app = self.app
        if app is not None and hasattr(app, "_update_scroll_follow"):
            app._update_scroll_follow(self)
