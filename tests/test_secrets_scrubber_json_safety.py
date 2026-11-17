"""Scrubbing text must never turn valid JSON into invalid JSON.

Session events and delegation digests are serialized JSON strings that the
scrubber treats as plain text. The credential value class used to accept a
trailing backslash, so ``...secret:hunter2\\"`` — an escaped quote INSIDE a
JSON string — was replaced wholesale: the escape and the closing quote went
with it, and the body no longer parsed.

Owner-visible symptom: a delegation wave wake painted its raw
23 KB digest as a "Background task" bubble instead of the collapsed card,
because the web parser could not read the damaged JSON and fell back to text.
"""

import json

from sharedagents_core.secrets_scrubber import scrub_text

RAW = '{"Resource":["arn:aws:secretsmanager:us-east-1:111122223333:secret:hunter2"]}'


def test_secret_before_an_escaped_quote_keeps_json_valid():
    line = json.dumps({"content": RAW})
    out = scrub_text(line)
    parsed = json.loads(out)
    assert "hunter2" not in out
    assert parsed["content"] == (
        '{"Resource":["arn:aws:secretsmanager:us-east-1:111122223333:secret:<<REDACTED:CREDENTIAL>>"]}'
    )


def test_scrubbing_is_idempotent_on_that_shape():
    once = scrub_text(json.dumps({"content": RAW}))
    twice = scrub_text(once)
    assert twice == once
    assert json.loads(twice)["content"].endswith('"]}')


def test_backslash_inside_a_value_is_still_redacted():
    assert scrub_text(r"secret:abc\defgh") == "secret:<<REDACTED:CREDENTIAL>>"
    assert scrub_text(r"SECRET_KEY=abc\defghij") == "SECRET_KEY=<<REDACTED:ENV_CREDENTIAL>>"
    assert scrub_text(r"--api-key abc\defgh") == "--api-key <<REDACTED:CLI_FLAG_CREDENTIAL>>"


def test_trailing_backslash_value_is_still_redacted():
    # The capture stops before an escaped QUOTE, not before a backslash that
    # belongs to the secret: a value ending in `\` is redacted whole.
    assert scrub_text("secret:abcdef\\") == "secret:<<REDACTED:CREDENTIAL>>"
    assert scrub_text("secret:abcde\\") == "secret:<<REDACTED:CREDENTIAL>>"
    assert scrub_text('password="hunter2"') == 'password="<<REDACTED:CREDENTIAL>>"'


def test_delegation_digest_round_trips_through_the_scrubber():
    members = [
        {
            "agent": "research",
            "status": "done",
            "ok": True,
            "task_id": "20260101T000000000000-aaaaaaaaaaaa",
            "content": RAW,
            "error": None,
        }
    ]
    digest = json.dumps(members)
    out = scrub_text(digest)
    parsed = json.loads(out)
    assert parsed[0]["task_id"] == members[0]["task_id"]
    assert parsed[0]["status"] == "done"
    assert "hunter2" not in out
    assert "<<REDACTED:CREDENTIAL>>" in parsed[0]["content"]


def test_a_scrubbed_wave_envelope_keeps_a_parseable_digest():
    """The envelope the web parser reads is the scrubbed one (session writer).

    If scrubbing damaged the body, the browser could not parse the wave and
    painted the whole digest as a "Background task" bubble.
    """
    from sharedagents_core.injection_guard import wrap_untrusted

    wave_id = "wave_" + "a" * 32
    members = [{"agent": "research", "status": "done", "task_id": "t", "content": RAW}]
    body = wrap_untrusted(json.dumps(members), source=f"subagent-wave:{wave_id}")
    envelope = (
        f'<delegation_wave wave_id="{wave_id}" terminal="1" total="1">\n'
        f"{body}\n</delegation_wave>\n\nAll children in this delegation wave are terminal."
    )
    scrubbed = scrub_text(envelope)
    lines = scrubbed.split("\n")
    parsed = json.loads(lines[2])
    assert parsed[0]["task_id"] == "t"
    assert "hunter2" not in scrubbed
    assert scrubbed.split("\n")[3].startswith("⟦/UNTRUSTED-DATA")
