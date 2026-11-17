"""WhatsApp channel for sharedagents — a linked-device, self-chat frontend.

The operator writes in WhatsApp's "Message yourself" chat. A standalone service
(``python -m sharedagents_whatsapp`` / ``sharedagents whatsapp run``) forwards each message
to an sharedagents agent through the gateway's canonical submit route, and delivers
every turn of the bound session — including turns started on web/TUI and
background wake-ups — back into the same chat.

The WhatsApp protocol client is ``neonize`` (a binding of the Go ``whatsmeow``
library), an optional dependency: ``pip install sharedagents[whatsapp]``, then
``sharedagents whatsapp link``.
"""
