---
name: telegram-render-verify
description: Use when you changed the sharedagents Telegram render/streaming path (sharedagents_telegram/bot.py — _BlockStream, _RollingMessage, _render_event, metrics/stats line, reply-linking) and need to confirm no duplicated messages, the correct 💭→🔧→💬 message sequence, reply-links, and the closing stats line — without deploying to the production bot.
---

# Verifying Telegram render changes

## Overview
The Telegram reply is a SEQUENCE of separate messages per turn: `💭` reasoning,
`🔧` tool calls, `💬` answer (stats line on the last). Render bugs show up as
**duplicated messages**, **cropped/fragmented answers** (an interleaving model
emits stray thinking mid-answer), wrong **order** (💭 below 💬), stray idle
`💭 thinking…`, or a missing **stats** line. test 10's `-m goal` subset asserts
these; for fast iteration, drive a worktree test-bot against the live gateway.

## Two-pronged workflow

**1. Fast interactive loop (seconds) — worktree test-bot on the LIVE gateway.**
Runs YOUR worktree code under a separate TEST bot token, so prod is
untouched and there is no long-poll conflict. After each `bot.py` edit, restart
the bot (it loads the module at launch) and re-drive.

```bash
SK=~/.sharedagents/skills/telegram-render-verify
# (re)start — kill the old one first; safe to repeat after every edit
pkill -f 'tg-render-verify/telegram' 2>/dev/null; sleep 2
bash $SK/spawn_test_bot.sh <worktree_dir>          # prints "bot connected: @<test_bot_username>"
# drive AS THE USER and print every reply (id / reply_to / body)
PY=~/.sharedagents/telegram/telethon-venv/bin/python
$PY $SK/drive.py <test_bot_username> "/new" "/reasoning stream" "give me a 50 word joke"
$PY $SK/drive.py <test_bot_username> "/tools on" "use the bash tool to run \`date -u\` and tell me the result"
```

**2. Formal gate (~2 min) — the `-m goal` subset of test 10.**
Its own Docker gateway runs the worktree code and the reasoning subtests carry
explicit no-duplication + ordering + stats assertions. Stop the test-bot first
(shared token). Run from the worktree:

```bash
pkill -f 'tg-render-verify/telegram'
TIMEOUT=300 tests/run-docker.sh tests/test_10_telegram_commands.py -m goal -n0
```

## What to check in the drive output
- **No duplication** — no two messages near-identical (model *drafting* the
  answer inside `💭` with partial overlap is OK; a whole-message twin is the bug).
- **Sequence** — message ids ascending = top→bottom: `💭` above `💬`; each `🔧`
  its own message in turn order; reasoning `on` → ONE consolidated `💭` at top.
- **reply_to** — every block's `reply_to` = the prompt's id (answers link to the
  question).
- **Stats** — `─ … N tok …` closes the answer message.
- **No stray idle** `💭 thinking…` / `💬 …` left behind.

## Gotchas
- The test-bot uses the **live gateway** → `bot.py` (sharedagents_telegram) changes
  take effect on restart, but gateway-side changes (sharedagents_core: session_reader,
  local_llm metrics/model-label) do NOT — verify those via the `-m goal` test-10
  run (its own gateway runs the worktree). 
- Telethon can't resolve the bot by raw id with no prior dialog — `drive.py`
  resolves by **username** (`<test_bot_username>`).
- One bot per token: a host test-bot and a test-10 container bot both long-poll
  `SHAREDAGENTS_TEST_TELEGRAM_TOKEN` → kill the test-bot before running test 10.
- Telethon login + chat ids are owned by the **telegram-test-drive** skill
  (`~/.sharedagents/telegram/telethon.session`); reuse it, don't re-login.
