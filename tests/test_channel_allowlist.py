"""Channels-tab allowlist CRUD for Telegram and WhatsApp."""

import json

import pytest

from sharedagents_telegram.config import TelegramConfig
from sharedagents_web import channel_allowlist
from sharedagents_whatsapp.config import WhatsAppConfig


def _homes(monkeypatch, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setenv("SHAREDAGENTS_TELEGRAM_HOME", str(tmp_path / "telegram"))
    monkeypatch.setenv("SHAREDAGENTS_WHATSAPP_HOME", str(tmp_path / "whatsapp"))


def test_person_reply_drops_to_me(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    channel_allowlist.add("whatsapp", "person", "+5511999990001", "Lia", reply="to_me")
    channel_allowlist.add("whatsapp", "group", "120363@g.us", "Family", reply="to_me")
    snap = {c["id"]: c for c in channel_allowlist.snapshot()["channels"]}
    assert snap["whatsapp"]["people"][0]["reply"] == "any"
    assert snap["whatsapp"]["groups"][0]["reply"] == "to_me"


def test_snapshot_starts_empty(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    data = channel_allowlist.snapshot()
    ids = [c["id"] for c in data["channels"]]
    assert ids == ["telegram", "whatsapp"]
    for ch in data["channels"]:
        assert ch["people"] == []
        assert ch["groups"] == []
        assert ch["suggest_people"] == []
        assert ch["suggest_groups"] == []
        assert ch["suggest_people"] == []
        assert ch["suggest_groups"] == []


def test_telegram_and_whatsapp_crud_roundtrip(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    channel_allowlist.add("telegram", "person", "4242", "Ana")
    channel_allowlist.add("telegram", "group", "-1001", "Ops")
    channel_allowlist.add("whatsapp", "person", "+5511999990001", "Lia")
    channel_allowlist.add("whatsapp", "group", "120363@g.us", "Family")

    snap = channel_allowlist.snapshot()
    by_id = {c["id"]: c for c in snap["channels"]}
    assert by_id["telegram"]["people"][0]["id"] == "4242"
    assert by_id["telegram"]["people"][0]["allowed"] is True
    assert by_id["telegram"]["people"][0]["read"] is True
    assert by_id["telegram"]["people"][0]["send"] is True
    assert by_id["telegram"]["groups"][0]["id"] == "-1001"
    assert by_id["whatsapp"]["people"][0]["id"] == "+5511999990001"
    assert by_id["whatsapp"]["groups"][0]["id"] == "120363@g.us"

    tg = TelegramConfig.load()
    assert tg.allowed_user_ids == [4242]
    assert 4242 in tg.allowed_chat_ids
    assert -1001 in tg.allowed_chat_ids
    assert tg.bot_token == ""

    channel_allowlist.update("telegram", "person", "4242", label="Ana M")
    channel_allowlist.update("whatsapp", "person", "+5511999990001", new_id="+5511999999999")
    channel_allowlist.remove("telegram", "group", "-1001")
    channel_allowlist.remove("whatsapp", "group", "120363@g.us")

    snap = channel_allowlist.snapshot()
    by_id = {c["id"]: c for c in snap["channels"]}
    assert by_id["telegram"]["people"][0]["id"] == "4242"
    assert by_id["telegram"]["people"][0]["label"] == "Ana M"
    assert by_id["telegram"]["people"][0]["allowed"] is True
    assert by_id["telegram"]["groups"] == []
    assert by_id["whatsapp"]["people"][0]["id"] == "+5511999999999"
    assert by_id["whatsapp"]["groups"] == []


def test_rejects_bad_channel_or_telegram_id(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    try:
        channel_allowlist.add("sms", "person", "1")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unknown channel" in str(exc)
    try:
        channel_allowlist.add("telegram", "person", "not-a-number")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "integers" in str(exc)


def test_whatsapp_config_allowlist_load_save(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "default_agent": "build",
                "allowed_people": ["+5511", {"id": "x@s.whatsapp.net", "label": "X"}],
                "allowed_groups": [{"id": "g@g.us", "label": "G"}],
            }
        )
    )
    cfg = WhatsAppConfig.load(tmp_path)
    assert cfg.allowed_people[0]["id"] == "+5511"
    assert cfg.allowed_people[1]["id"] == "x@s.whatsapp.net"
    assert cfg.allowed_people[0]["read"] is True
    assert cfg.allowed_people[0]["send"] is True
    assert cfg.allows("+5511")
    assert cfg.allows("5511@s.whatsapp.net")
    assert cfg.allows("x")
    assert cfg.allows("g@g.us", group=True)
    cfg.allowed_people.append({"id": "+5521", "label": "Me"})
    cfg.save()
    again = json.loads((tmp_path / "config.json").read_text())
    assert again["default_agent"] == "build"
    assert again["allowed_people"][-1]["id"] == "+5521"
    assert "gateway_token" not in again


def test_whatsapp_directory_lists_contacts_and_groups(tmp_path):
    import sqlite3

    store = tmp_path / "neonize.sqlite3"
    con = sqlite3.connect(store)
    con.executescript(
        """
        CREATE TABLE whatsmeow_device (jid TEXT, lid TEXT, push_name TEXT);
        CREATE TABLE whatsmeow_contacts (
            our_jid TEXT, their_jid TEXT, first_name TEXT, full_name TEXT,
            push_name TEXT, business_name TEXT, redacted_phone TEXT
        );
        CREATE TABLE whatsmeow_chat_settings (
            our_jid TEXT, chat_jid TEXT, muted_until INTEGER,
            pinned INTEGER, archived INTEGER
        );
        INSERT INTO whatsmeow_device VALUES ('5512999990003:1@s.whatsapp.net', '', 'Me');
        INSERT INTO whatsmeow_contacts VALUES
            ('me', '5511999990001@s.whatsapp.net', '', 'Lia', 'Lia', '', ''),
            ('me', '5512999990003@s.whatsapp.net', '', 'Self', '', '', ''),
            ('me', '120363abc@g.us', '', 'Family', '', '', '');
        INSERT INTO whatsmeow_chat_settings VALUES
            ('me', '120363abc@g.us', 0, 0, 0),
            ('me', '120363xyz@g.us', 0, 0, 0);
        """
    )
    con.commit()
    con.close()

    from sharedagents_whatsapp.directory import list_directory

    data = list_directory(home=tmp_path)
    assert data["linked"] is True
    people_ids = [row["id"] for row in data["people"]]
    assert "5511999990001@s.whatsapp.net" in people_ids
    assert "5512999990003@s.whatsapp.net" not in people_ids
    group_ids = [row["id"] for row in data["groups"]]
    assert "120363abc@g.us" in group_ids
    assert "120363xyz@g.us" in group_ids
    family = next(row for row in data["groups"] if row["id"] == "120363abc@g.us")
    assert family["label"] == "Family"


def test_whatsapp_group_name_resolves_to_a_stable_id(monkeypatch, tmp_path):
    from sharedagents_whatsapp import directory

    _homes(monkeypatch, tmp_path)
    group_id = "120363000000000001@g.us"
    groups = [{"id": group_id, "label": "Home Group"}]
    monkeypatch.setattr(
        directory,
        "list_directory",
        lambda **kwargs: {
            "linked": True,
            "people": [],
            "groups": groups,
        },
    )
    channel_allowlist.add("whatsapp", "group", "home group", reply="mention", mention="@helper")
    cfg = WhatsAppConfig.load()
    assert cfg.allowed_groups[0]["id"] == group_id
    assert cfg.allows(group_id, group=True, access="read")
    assert cfg.allows(group_id, group=True, access="send")
    assert not cfg.allows("120363000000000002@g.us", group=True)

    # Subsequent group renames do not move the allowlist to another chat.
    groups[0]["label"] = "Renamed Group"
    groups.append({"id": "120363000000000002@g.us", "label": "Home Group"})
    assert WhatsAppConfig.load().allowed_groups[0]["id"] == group_id


@pytest.mark.parametrize("identifier", ["Unknown", "Family@s.whatsapp.net", "Family@g.us"])
def test_whatsapp_group_id_rejects_unknown_names_without_saving(monkeypatch, tmp_path, identifier):
    _homes(monkeypatch, tmp_path)
    channel_allowlist.add("whatsapp", "person", "+15555550100", "Contact")
    path = WhatsAppConfig.load().config_path
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Select a WhatsApp group"):
        channel_allowlist.add("whatsapp", "group", identifier)
    assert path.read_bytes() == before


def test_whatsapp_group_name_rejects_ambiguous_matches(monkeypatch, tmp_path):
    from sharedagents_whatsapp import directory

    _homes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        directory,
        "list_directory",
        lambda **kwargs: {
            "linked": True,
            "people": [],
            "groups": [
                {"id": "120363000000000001@g.us", "label": "Family"},
                {"id": "120363000000000002@g.us", "label": "Family"},
            ],
        },
    )
    with pytest.raises(ValueError, match="Multiple WhatsApp groups"):
        channel_allowlist.add("whatsapp", "group", "Family")
    assert not WhatsAppConfig.load().config_path.exists()


def test_whatsapp_group_name_update_repairs_legacy_entry(monkeypatch, tmp_path):
    from sharedagents_whatsapp import directory

    _homes(monkeypatch, tmp_path)
    group_id = "120363000000000001@g.us"
    monkeypatch.setattr(
        directory,
        "list_directory",
        lambda **kwargs: {
            "linked": True,
            "people": [],
            "groups": [{"id": group_id, "label": "Family"}],
        },
    )
    cfg = WhatsAppConfig.load()
    cfg.allowed_groups = [
        {
            "id": "Family",
            "label": "",
            "reply": "mention",
            "mention": "@helper",
            "read": True,
            "send": False,
            "agent": "helper",
            "model": "example-model",
        }
    ]
    cfg.save()
    channel_allowlist.update("whatsapp", "group", "Family")
    row = WhatsAppConfig.load().allowed_groups[0]
    assert row == {**cfg.allowed_groups[0], "id": group_id}


@pytest.mark.parametrize("group_id", ["120363000000000001@g.us", "15555550100-1234567890@g.us"])
def test_whatsapp_group_id_accepts_modern_and_legacy_ids(monkeypatch, tmp_path, group_id):
    _homes(monkeypatch, tmp_path)
    channel_allowlist.add("whatsapp", "group", group_id)
    assert WhatsAppConfig.load().allowed_groups[0]["id"] == group_id


def test_snapshot_keeps_directory_as_suggestions_only(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    import sqlite3

    store = tmp_path / "whatsapp" / "neonize.sqlite3"
    store.parent.mkdir(parents=True)
    con = sqlite3.connect(store)
    con.executescript(
        """
        CREATE TABLE whatsmeow_device (jid TEXT, lid TEXT, push_name TEXT);
        CREATE TABLE whatsmeow_contacts (
            our_jid TEXT, their_jid TEXT, first_name TEXT, full_name TEXT,
            push_name TEXT, business_name TEXT, redacted_phone TEXT
        );
        CREATE TABLE whatsmeow_chat_settings (
            our_jid TEXT, chat_jid TEXT, muted_until INTEGER,
            pinned INTEGER, archived INTEGER
        );
        INSERT INTO whatsmeow_device VALUES ('5512999990003:1@s.whatsapp.net', '', 'Me');
        INSERT INTO whatsmeow_contacts VALUES
            ('me', '5511999990001@s.whatsapp.net', '', 'Lia', 'Lia', '', '');
        INSERT INTO whatsmeow_chat_settings VALUES
            ('me', '120363abc@g.us', 0, 0, 0);
        """
    )
    con.commit()
    con.close()

    snap = channel_allowlist.snapshot()
    wa = next(ch for ch in snap["channels"] if ch["id"] == "whatsapp")
    assert wa["people"] == []
    assert wa["groups"] == []
    assert any(row["id"] == "5511999990001@s.whatsapp.net" for row in wa["suggest_people"])
    assert any(row["id"] == "120363abc@g.us" for row in wa["suggest_groups"])

    channel_allowlist.add(
        "whatsapp",
        "person",
        "5511999990001@s.whatsapp.net",
        "Lia",
        read=True,
        send=False,
    )
    snap = channel_allowlist.snapshot()
    wa = next(ch for ch in snap["channels"] if ch["id"] == "whatsapp")
    assert len(wa["people"]) == 1
    assert wa["people"][0]["read"] is True
    assert wa["people"][0]["send"] is False
    assert wa["people"][0]["label"] == "Lia"
    assert not any(row["id"] == "5511999990001@s.whatsapp.net" for row in wa["suggest_people"])

    cfg = WhatsAppConfig.load()
    assert cfg.allows("5511999990001@s.whatsapp.net", access="read")
    assert not cfg.allows("5511999990001@s.whatsapp.net", access="send")


def test_entry_matches_lid_alias_from_store(tmp_path):
    import sqlite3

    store = tmp_path / "neonize.sqlite3"
    con = sqlite3.connect(store)
    con.execute("CREATE TABLE whatsmeow_lid_map (lid TEXT, pn TEXT)")
    con.execute("INSERT INTO whatsmeow_lid_map VALUES ('100000000000001', '5511999990001')")
    con.commit()
    con.close()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "allowed_people": [
                    {
                        "id": "5511999990001@s.whatsapp.net",
                        "label": "Contact/M",
                        "reply": "mention",
                        "mention": "@bot",
                        "read": True,
                        "send": True,
                    }
                ],
            }
        )
    )
    cfg = WhatsAppConfig.load(tmp_path)
    assert cfg.lid_aliases["100000000000001"] == "5511999990001"
    assert cfg.entry("100000000000001@lid")["label"] == "Contact/M"
    assert cfg.allows("100000000000001@lid", access="read")


def test_telegram_read_send_flags(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    channel_allowlist.add("telegram", "group", "-1001", "Ops", read=True, send=False)
    tg = TelegramConfig.load()
    assert tg.can_read(-1001) is True
    assert tg.can_send(-1001) is False
    channel_allowlist.update("telegram", "group", "-1001", send=True)
    assert TelegramConfig.load().can_send(-1001) is True


def test_mention_and_agent_route_persist(monkeypatch, tmp_path):
    _homes(monkeypatch, tmp_path)
    snap = channel_allowlist.snapshot()
    assert snap["agents"]
    assert any(row["name"] == "jarvis" for row in snap["agents"])

    channel_allowlist.add(
        "telegram",
        "group",
        "-1001",
        "Ops",
        reply="mention",
        mention="custom_helper",
        agent="custom_helper",
        model="qwen38-fp8",
    )
    channel_allowlist.add(
        "whatsapp",
        "person",
        "+5511999990001",
        "Lia",
        reply="mention",
        mention="@bot",
        agent="build",
    )
    snap = channel_allowlist.snapshot()
    by_id = {c["id"]: c for c in snap["channels"]}
    tg_row = by_id["telegram"]["groups"][0]
    assert tg_row["reply"] == "mention"
    assert tg_row["mention"] == "@custom_helper"
    assert tg_row["agent"] == "custom_helper"
    assert tg_row["model"] == "qwen38-fp8"
    wa_row = by_id["whatsapp"]["people"][0]
    assert wa_row["reply"] == "mention"
    assert wa_row["mention"] == "@bot"
    assert wa_row["agent"] == "build"

    tg = TelegramConfig.load()
    assert tg.agent_routes["-1001"] == "custom_helper"
    assert tg.allowlist_rules["group:-1001"]["mention"] == "@custom_helper"

    wa = WhatsAppConfig.load()
    assert wa.allowed_people[0]["agent"] == "build"
    assert wa.routed_agent("+5511999990001") == "build"
