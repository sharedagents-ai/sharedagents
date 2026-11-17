"""WhatsApp self-chat reply addressing and group directory names (PR #9).

The broad neonize transport suite was retired on the OSS readiness line; these
are the regression tests PR #9 added for its fix, with only the helpers they
need.
"""

import sqlite3

import pytest

pytest.importorskip("neonize")
from neonize.proto import Neonize_pb2 as N  # noqa: E402
from neonize.proto.waE2E import WAWebProtobufsE2E_pb2 as E  # noqa: E402

from sharedagents_whatsapp.neonize_transport import NeonizeTransport  # noqa: E402

ME_PN, ME_LID, MY_DEVICE = "5511999990021", "123456789012345", 23


def jid(user, server="s.whatsapp.net", device=0):
    return N.JID(User=user, Server=server, Device=device, RawAgent=0, Integrator=0, IsEmpty=False)


def event(*, chat, sender, body, is_group=False, from_me=True, ts=1_800_000_000, mid="ABC"):
    ev = N.Message()
    src = ev.Info.MessageSource
    src.Chat.CopyFrom(chat)
    src.Sender.CopyFrom(sender)
    src.IsFromMe = from_me
    src.IsGroup = is_group
    ev.Info.ID = mid
    ev.Info.Timestamp = ts
    ev.Message.CopyFrom(body)
    return ev


@pytest.fixture
def transport(tmp_path):
    t = NeonizeTransport(tmp_path / "neonize.sqlite3", home=tmp_path)
    t._set_identity(ME_PN, ME_LID, MY_DEVICE)
    return t


@pytest.mark.asyncio
async def test_send_quotes_lid_self_chat_without_forcing_phone_jid(transport):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from neonize.aioze.client import NewAClient

    client = object.__new__(NewAClient)
    client._NewAClient__client = SimpleNamespace(IsConnected=AsyncMock(return_value=True))
    client.uuid = b"test"
    client.send_message = AsyncMock(return_value=SimpleNamespace(ID="plain"))
    client.reply_message = AsyncMock(return_value=SimpleNamespace(ID="quoted"))
    transport._client = client
    raw = event(chat=jid(ME_LID, "lid"), sender=jid(ME_LID, "lid"), body=E.Message(conversation="hi"))
    inbound = transport.normalize(raw)
    assert inbound.is_self_chat
    assert await transport.send_text("Hey", quote=inbound) == "quoted"
    client.send_message.assert_not_called()
    assert client.reply_message.await_args.args == ("Hey", raw)
    assert client.reply_message.await_args.kwargs["to"].User == ME_LID
    assert client.reply_message.await_args.kwargs["to"].Server == "lid"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [jid(ME_PN), jid(ME_LID, "lid")])
async def test_unquoted_self_chat_send_uses_the_last_inbound_address(transport, chat):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    client = SimpleNamespace(
        is_connected=True,
        send_message=AsyncMock(return_value=SimpleNamespace(ID="plain")),
    )
    transport._client = client
    inbound = transport.normalize(event(chat=chat, sender=chat, body=E.Message(conversation="/new")))
    assert inbound.is_self_chat

    assert await transport.send_text("New session") == "plain"
    target = client.send_message.await_args.args[0]
    assert (target.User, target.Server, target.Device) == (chat.User, chat.Server, 0)


@pytest.mark.asyncio
async def test_quoted_self_chat_send_keeps_its_address_after_another_inbound(transport):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    client = SimpleNamespace(
        is_connected=True,
        reply_message=AsyncMock(return_value=SimpleNamespace(ID="quoted")),
    )
    transport._client = client
    lid_chat = jid(ME_LID, "lid")
    inbound = transport.normalize(
        event(
            chat=lid_chat,
            sender=lid_chat,
            body=E.Message(conversation="hello"),
        )
    )
    transport.normalize(
        event(
            chat=jid(ME_PN),
            sender=jid(ME_PN),
            body=E.Message(conversation="next"),
            mid="NEXT",
        )
    )

    assert await transport.send_text("Reply to hello", quote=inbound) == "quoted"
    target = client.reply_message.await_args.kwargs["to"]
    assert (target.User, target.Server, target.Device) == (ME_LID, "lid", 0)


@pytest.mark.asyncio
async def test_connect_populates_group_directory_names(transport, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sharedagents_whatsapp.directory import list_directory

    con = sqlite3.connect(transport._store)
    con.executescript("""
        CREATE TABLE whatsmeow_device (jid TEXT, lid TEXT, push_name TEXT);
        CREATE TABLE whatsmeow_contacts (
            their_jid TEXT, first_name TEXT, full_name TEXT, push_name TEXT, business_name TEXT
        );
    """)
    con.execute("INSERT INTO whatsmeow_device VALUES (?, ?, ?)", (f"{ME_PN}:23@s.whatsapp.net", "", "Op"))
    con.commit()
    con.close()

    async def store_names(entries):
        with sqlite3.connect(transport._store) as con:
            con.executemany(
                "INSERT INTO whatsmeow_contacts VALUES (?, ?, ?, '', '')",
                [(f"{entry.JID.User}@{entry.JID.Server}", entry.FirstName, entry.FullName) for entry in entries],
            )

    group_id = "120363000000000001"
    transport._client = SimpleNamespace(
        me=SimpleNamespace(JID=jid(ME_PN, device=MY_DEVICE), LID=jid(ME_LID, "lid")),
        get_joined_groups=AsyncMock(
            return_value=[
                N.GroupInfo(JID=jid(group_id, "g.us"), GroupName=N.GroupName(Name="Home Group")),
                N.GroupInfo(JID=jid("120363000000000002", "g.us"), GroupName=N.GroupName(Name="")),
                N.GroupInfo(JID=jid("15555550100"), GroupName=N.GroupName(Name="Not a group")),
            ]
        ),
        contact=SimpleNamespace(put_all_contact_name=AsyncMock(side_effect=store_names)),
    )
    await transport._on_connected()
    assert transport._ready.is_set()
    data = list_directory(home=tmp_path)
    assert data["groups"] == [{"id": f"{group_id}@g.us", "label": "Home Group", "phone": ""}]
    assert data["people"] == []


@pytest.mark.asyncio
async def test_group_directory_failure_keeps_channel_ready(transport):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    transport._client = SimpleNamespace(
        me=SimpleNamespace(JID=jid(ME_PN, device=MY_DEVICE), LID=jid(ME_LID, "lid")),
        get_joined_groups=AsyncMock(side_effect=RuntimeError("unavailable")),
        contact=SimpleNamespace(put_all_contact_name=AsyncMock()),
    )
    await transport._on_connected()
    assert transport._ready.is_set()
    assert transport.is_self_chat(jid(ME_LID, "lid"))
    transport._client.contact.put_all_contact_name.assert_not_called()
