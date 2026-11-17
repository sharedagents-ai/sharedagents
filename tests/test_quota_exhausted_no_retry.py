"""A plan-quota wall (zcode ``[1308]``) must not trigger the 5 s retry burst."""

from sharedagents_agent.agents.agent import _turn_error_is_retryable
from sharedagents_agent.agents.external_cli import _cli_error_is_quota_exhausted

_ZCODE_1308 = (
    "zcode exited 1: ZCode Built-in skipped (not-due)\n"
    "ProviderBusinessError: [1308][Usage limit reached for 5 hour. Your limit "
    "will reset at 2026-10-02 12:55:22]"
)


def test_zcode_usage_limit_is_detected():
    assert _cli_error_is_quota_exhausted(_ZCODE_1308)


def test_ordinary_cli_error_is_not_quota():
    assert not _cli_error_is_quota_exhausted("zcode exited 1: Unknown option '--x'")


_ANTIGRAVITY_INDIVIDUAL = (
    "antigravity reported error: Individual quota reached. "
    "Please upgrade your subscription to increase your limits. Resets in 167h."
)


def test_antigravity_individual_quota_is_detected():
    """The 2026-10-16 wall is a quota exhaustion, not a generic CLI error."""
    assert _cli_error_is_quota_exhausted(_ANTIGRAVITY_INDIVIDUAL)


def test_classified_quota_records_the_reset_for_the_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.delenv("SHAREDAGENTS_AGENTS_HOME", raising=False)
    from sharedagents_agent.agents.engine import EngineKind
    from sharedagents_agent.agents.external_cli import _remember_cli_quota
    from sharedagents_web import subscription_limits as sl

    fixed = sl._parse_iso_epoch("2026-10-16T23:49:00Z")
    monkeypatch.setattr(sl.time, "time", lambda: fixed)
    _remember_cli_quota(EngineKind.ANTIGRAVITY_CLI, _ANTIGRAVITY_INDIVIDUAL)
    saved = (tmp_path / "usage" / "cli_quota.json").read_text(encoding="utf-8")
    assert "2026-10-23T22:49:00Z" in saved
    assert "antigravity" in saved


def test_quota_exhausted_is_not_retried_but_cli_error_still_is():
    assert not _turn_error_is_retryable("quota_exhausted")
    assert _turn_error_is_retryable("cli_error")
