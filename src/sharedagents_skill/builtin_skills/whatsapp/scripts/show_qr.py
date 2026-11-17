#!/usr/bin/env python3
"""Copy the live WhatsApp pairing QR into a chat-deliverable artifact.

On-demand skill script — agents load the ``whatsapp`` skill, then run this
(or ``python -m sharedagents_whatsapp.qr``). Not a builtin always-on tool.
"""

from sharedagents_whatsapp.qr import main

if __name__ == "__main__":
    raise SystemExit(main())
