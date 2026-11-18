# WhatsApp (self-chat channel)

Talk to your agent from WhatsApp: sharedagents links to your account as a companion device
(like WhatsApp Web, shown as "sharedagents" under *Linked devices*). You write in your own
"Message yourself" chat and the agent answers in the same chat, including turns that
finish later in the background.

```bash
pip install 'sharedagents[whatsapp]'           # plus: apt install libmagic1
sharedagents whatsapp link                     # scan the QR (also saved as ~/.sharedagents/whatsapp/link-qr.png)
sharedagents whatsapp link --phone +5511...    # or get an 8-character pairing code instead
sharedagents whatsapp install-service          # systemd user unit sharedagents-whatsapp.service
sharedagents whatsapp status
```

In the chat, `/new` starts a fresh session, `/stop` stops a turn, `/agent <name>` switches
agents and `/status` shows the link. Every other `/command` runs in the agent runtime.
Voice notes are transcribed through the `stt` service, and agent questions or permission
prompts are answered by replying with a number.

**Files.** Send an image (as a photo or as a file) and the agent sees it; send a PDF,
Markdown/text or Office document and the agent reads its text (converted with `markitdown`
from the `[full]` extra; up to `max_media_mb`, default 16 MB). When the agent links a file
it made (`[Report](</abs/path/report.pdf>)`, the same contract the web UI uses), the file
follows its answer: PNG/JPEG/WebP as a photo, anything else as a document. Agents can also
queue files with the `whatsapp` skill: `send --file </abs/path/report.pdf> [text]`.
Telegram does the same: images and documents in, the agent's linked files out.

- **Privacy.** The service only reads and writes your self-chat. Other chats are ignored
  and never logged.
- **Credentials.** `~/.sharedagents/whatsapp/neonize.sqlite3` is a live WhatsApp login, stored
  with permissions 0600. `sharedagents whatsapp logout` revokes it.
- **Terms of service.** This is an unofficial client, the same approach as WhatsApp Web
  automation tools. Keep it to your own self-chat. Settings live in
  `~/.sharedagents/whatsapp/config.json`.
