#!/usr/bin/env python3
"""Drive a Telegram bot AS THE USER (Telethon) and print every reply with its
id, reply_to anchor, and body — so you can eyeball duplication, the 💭→🔧→💬
message sequence, reply-linking, and the closing stats line after a render change.

Usage:
    python drive.py <bot_username> "<msg1>" "<msg2>" ...
    # optional per-message tuning via env:
    #   SETTLE=180  QUIET=14   (seconds)  — applies to ALL messages
    # examples:
    #   python drive.py <test_bot_username> "/new" "/reasoning stream" "give me a 50 word joke"
    #   python drive.py <test_bot_username> "/tools on" "use bash to run date -u"

Creds: TELEGRAM_API_ID/HASH from ~/.sharedagents/telegram/botapi.env; StringSession
from ~/.sharedagents/telegram/telethon.session (the telegram-test-drive skill's login).
Run with that skill's venv: ~/.sharedagents/telegram/telethon-venv/bin/python
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from telethon import TelegramClient
from telethon.sessions import StringSession

_PLACEHOLDERS = ("💭 thinking", "💬 …", "🔧 …", "…", "queued", "⏳")


def _creds():
    env = {}
    for ln in (Path.home() / ".sharedagents/telegram/botapi.env").read_text().splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip()
    api_id = int(env.get("TELEGRAM_API_ID") or env["TELETHON_API_ID"])
    api_hash = env.get("TELEGRAM_API_HASH") or env["TELETHON_API_HASH"]
    sess = (Path.home() / ".sharedagents/telegram/telethon.session").read_text().strip()
    return api_id, api_hash, sess


async def _ask(client, peer, text, settle, quiet):
    sent = await client.send_message(peer, text)
    start = last = time.time()
    seen = {}
    while time.time() - start < settle:
        await asyncio.sleep(2)
        msgs = await client.get_messages(peer, min_id=sent.id, limit=50)
        cur = {m.id: (m.message or "") for m in msgs}
        if cur != seen:
            seen, last = cur, time.time()
        real = [b for b in seen.values() if b and not any(b.strip().startswith(p) for p in _PLACEHOLDERS)]
        if time.time() - last > quiet and real:
            break
    msgs = sorted(await client.get_messages(peer, min_id=sent.id, limit=50), key=lambda m: m.id)
    print(f"\n##### sent {text!r} (id {sent.id}) → {len(msgs)} message(s) [ids ascending = top→bottom] #####")
    for m in msgs:
        rt = getattr(m, "reply_to", None)
        rid = getattr(rt, "reply_to_msg_id", None) if rt else None
        print(f"--- msg {m.id}  reply_to={rid}  ({len(m.message or '')} chars) ---")
        print(m.message or "(empty)")
    return msgs


async def _main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    bot = sys.argv[1].lstrip("@")
    prompts = sys.argv[2:]
    settle = float(os.environ.get("SETTLE", "180"))
    quiet = float(os.environ.get("QUIET", "14"))
    api_id, api_hash, sess = _creds()
    client = TelegramClient(StringSession(sess), api_id, api_hash)
    await client.connect()
    peer = await client.get_entity(bot)  # resolves by username from the server
    print(f"bot: @{getattr(peer, 'username', bot)} id={peer.id}")
    for i, p in enumerate(prompts):
        # commands settle fast; only the last (or any non-/) prompt needs the full budget
        s = 25 if p.startswith("/") and i < len(prompts) - 1 else settle
        q = 5 if p.startswith("/") and i < len(prompts) - 1 else quiet
        await _ask(client, peer, p, s, q)
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(_main())
