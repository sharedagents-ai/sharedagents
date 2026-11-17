#!/usr/bin/env python3
"""Admin-only Channels allowlist mutate (same gate as the WhatsApp skill)."""

import sys

from sharedagents_whatsapp.allowlist_skill import run_allowlist_args

if __name__ == "__main__":
    raw = " ".join(sys.argv[1:]).strip() or "allowlist"
    print(run_allowlist_args(raw), end="")
