#!/usr/bin/env python3
"""Print the masked linked WhatsApp account. Never lists contacts."""

from sharedagents_whatsapp.outbound import account_snapshot

if __name__ == "__main__":
    print(account_snapshot(), end="")
