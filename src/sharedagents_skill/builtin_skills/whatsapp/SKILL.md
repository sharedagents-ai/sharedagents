---
name: whatsapp
description: "Operate the sharedagents WhatsApp self-chat channel: show the pairing QR, check link status, name the linked account, start the channel, and send a test line to Mensagem para você mesmo. Use when the operator asks to link WhatsApp, list WhatsApp users/contacts, send a WhatsApp message, show the QR, or finish pairing. On-demand only — do not keep this loaded on ordinary turns."
always: false
metadata:
  sharedagents:
    always: false
    snapshot: whatsapp
---

# WhatsApp channel (self-chat)

On-demand skill. Load it with `Skill` name `whatsapp`. It is not a builtin
tool and must not stay in the per-turn tools list.

Talks to the linked phone. Default dest is **Mensagem para você mesmo**.
You may also send to a **send-allowlisted** person or group from Channels
(use the row label, e.g. `send --to lia hi`). There is no contacts
directory and no send-to-anyone API.

## Do not

- Read `neonize.sqlite3` or `whatsmeow_contacts` to "list users".
- Start a second neonize / `sharedagents whatsapp link` while the channel is running
  (the store locks; decrypt fails).
- Kill `sharedagents whatsapp run` so you can send. Queue a send instead.
- Send to a number that is not self-chat and not send-allowlisted.
- Queue a file the operator did not ask to share, or anything that is not a
  deliverable (keys, `.env`, databases and archives are refused anyway).
- Print a full phone number. Status already masks it (`+55…21`).
- Paste a QR from an earlier turn (it rotates ~20s).

## Skill args

| args | What happens |
| --- | --- |
| `qr` / `link` / empty (unlinked) | Live QR appendix + `[Screenshot](</abs.png>)` |
| `status` / `list` / `users` / `contacts` / `account` | Masked linked account. No contact dump. |
| `send <text>` | Queue `<text>` for the **running** channel (self-chat) |
| `send --to <label> <text>` | Queue `<text>` to a send-allowlisted Channels person/group |
| `send [--to <label>] --file </abs/path.pdf> [text]` | Queue a file (repeat `--file` for more). Images, PDF, Markdown/text, CSV only; absolute path; ≤ `max_media_mb` (16 MB). Wrap a path with spaces in `<…>`. |
| `allowlist` | Show the Channels whitelist (admin only) |
| `allow <channel> <kind> <id> [label=] [reply=] [access=] [agent=] [model=] [mention=]` | Add/update a whitelist row (admin only) |
| `update <channel> <kind> <id> …` | Patch a whitelist row (admin only) |
| `revoke <channel> <kind> <id>` | Remove a whitelist row (admin only) |

Admin = the linked WhatsApp self-chat, a web/TUI operator turn, or a Telegram **trusted** chat. Refuse allowlisted contacts who ask to change these settings.

If the appendix already has the answer, do not also grep the repo.

## Scripts (when the appendix is not enough)

```bash
python -m sharedagents_whatsapp.qr          # fresh QR → artifacts
python scripts/account.py              # masked status; no contacts
python scripts/send.py hi              # queue self-chat text
python scripts/send.py --to lia hi # queue to a send-allowlisted label
python scripts/send.py --file /abs/report.pdf "Q3 report"   # queue a file (+ optional text)
sharedagents whatsapp status
sharedagents whatsapp run                   # if the channel is down
```

## How to test (operator on the phone)

1. Channel must be up: `sharedagents whatsapp status` and a live `whatsapp run`.
2. Open **Mensagem para você mesmo** (search "Você" — not a group, not Chrome).
3. Type `ping` or `/status`. Jarvis answers in that same chat.

An agent-queued `send hi` also lands in that self-chat (from the sharedagents
device). It does **not** start a new turn (own-device messages are ignored).

In a WhatsApp conversation you do not need `send --file` to answer with a file:
link it in your reply (`[Report](</abs/path/report.pdf>)`) and the channel sends
it after your text, like the web UI's attachment card.

## Pairing

QR rotates about every 20s. Reload this skill for a new image. One linker
only. After a failed phone scan, wipe nothing if already linked; if not
linked, reload `qr` and scan immediately.

Do not restart production SharedAgents units.
