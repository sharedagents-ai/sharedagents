"""``list_accounts`` reports each account's latest sign-in activity."""

from sharedagents_web.native_auth import NativeAuthStore


def test_list_accounts_reports_last_seen(tmp_path):
    store = NativeAuthStore(tmp_path / "auth.db")
    store.create_user("active@example.com", "pw-active-123456")
    store.create_user("idle@example.com", "pw-idle-1234567")
    store.issue_token("active@example.com")

    by_email = {a["email"]: a for a in store.list_accounts()}
    assert by_email["active@example.com"]["last_seen"] > 0
    assert by_email["idle@example.com"]["last_seen"] == 0
    # A single-account lookup has no last_seen column and still reports 0.
    assert store.account("active@example.com")["last_seen"] == 0
