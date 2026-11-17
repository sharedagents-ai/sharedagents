"""Harden Textual's terminal input against non-UTF-8 bytes.

Textual's Linux input thread (``textual/drivers/linux_driver.py``,
``run_input_thread``) decodes raw stdin with a *strict* incremental
UTF-8 decoder::

    utf8_decoder = getincrementaldecoder("utf-8")().decode   # strict

A single byte that isn't valid UTF-8 raises ``UnicodeDecodeError``
*inside that thread*. Common sources over a real terminal / SSH link:

- a legacy **X10 mouse report** — its button/coordinate bytes are
  ``value + 32`` and routinely exceed ``0x7F`` (e.g. ``0x86``);
- an **Alt/Meta key** on a terminal configured for "meta sets the
  high bit" rather than ESC-prefix;
- a **non-UTF-8 paste**, or any stray byte the link delivers in the
  middle of an escape sequence.

Textual's only handling is to ``App.panic()`` (the Rich traceback box
the operator sees), after which the input thread is dead: keystrokes
stop being read and the TUI freezes / feels laggy until it tears down.
So the crash and the "lag" are one bug — a dead input thread.

We can't reach the per-thread decoder Textual constructs, but we can
swap the factory it imports. Replacing
``textual.drivers.linux_driver.getincrementaldecoder`` with a wrapper
that forces ``errors="surrogateescape"`` maps each stray byte to a
private surrogate (U+DC80..U+DCFF) instead of crashing the thread.
Unlike ``errors="replace"`` (which substitutes U+FFFD and *widens* a
single byte to three UTF-8 code units), surrogateescape preserves the
original byte value inside the decoded string so escape-sequence parsers
(X10/SGR mouse reports, CSI key chords) keep matching. ``replace`` was
observed leaking U+FFFD into the chat input whenever the operator
scrolled with the mouse over a resumed session — the corrupted sequence
fell through Textual's parser as literal keypresses.

Incremental buffering of legitimately split multibyte sequences is
preserved — surrogateescape only substitutes for genuinely invalid
sequences, never for a valid-but-incomplete trailing sequence (which
the incremental decoder still buffers across reads), so real UTF-8
input — including emoji split across two ``os.read`` calls — keeps
decoding correctly.
"""

from __future__ import annotations

import codecs

# Marker attribute stamped on the patched module so the install is
# idempotent and easy to assert against in tests.
_SENTINEL = "_sharedagents_tolerant_decoder"


def install_tolerant_input_decoder() -> bool:
    """Patch Textual's Linux input decoder to tolerate non-UTF-8 bytes.

    Idempotent and defensive: returns ``True`` when the patch is in
    place (or already was), ``False`` when Textual's internals don't
    match the shape we patch (non-Linux, Textual missing, or upstream
    renamed the symbol). On ``False`` we leave Textual untouched — a
    hardening shim must never break TUI startup.
    """
    try:
        from textual.drivers import linux_driver as _ld
    except Exception:  # noqa: BLE001 — non-Linux, Textual missing, or moved
        return False
    if getattr(_ld, _SENTINEL, False):
        return True
    try:

        def _tolerant_getincrementaldecoder(encoding: str):
            base_cls = codecs.getincrementaldecoder(encoding)
            # Scope the hardening to UTF-8 — the only encoding Textual's
            # input thread requests. Any other encoding (and its
            # caller-supplied error policy) is returned untouched, so
            # this shim can never alter unrelated decoding behaviour.
            if codecs.lookup(encoding).name != "utf-8":
                return base_cls

            class _TolerantIncrementalDecoder(base_cls):  # type: ignore[misc,valid-type]
                # Textual constructs this with no arguments
                # (``getincrementaldecoder("utf-8")()``), so the default
                # here is what takes effect: decode with "surrogateescape"
                # so a stray non-UTF-8 byte becomes a private surrogate
                # (preserving the byte for escape-sequence parsers) instead
                # of killing the input thread or widening it to U+FFFD.
                def __init__(self, errors: str = "surrogateescape") -> None:
                    super().__init__(errors="surrogateescape")

            return _TolerantIncrementalDecoder

        _ld.getincrementaldecoder = _tolerant_getincrementaldecoder
        setattr(_ld, _SENTINEL, True)
        return True
    except Exception:  # noqa: BLE001 — never let hardening break launch
        return False
