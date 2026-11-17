"""Per-user grants codec, prompt stamps and tool/delegation decisions."""

import json

from sharedagents_core.access_grants import (
    ACCESS_META_KEY,
    AccessStamp,
    Grants,
    delegation_refusal,
    filter_tool_names,
    merge_stamps,
    stamp_blocks,
    stamp_from_blocks,
    strip_access_meta,
)


def test_all_and_none_round_trip_through_json():
    assert Grants.from_obj(Grants.all().to_obj()) == Grants.all()
    assert Grants.from_obj(Grants.none().to_obj()) == Grants.none()
    assert Grants.all().to_obj() == {
        "agents": "*",
        "models": "*",
        "tools": "*",
        "skills": "*",
        "folders": "*",
    }
    assert Grants.none().to_obj() == {
        "agents": [],
        "models": [],
        "tools": [],
        "skills": [],
        "folders": [],
    }


def test_malformed_input_fails_closed():
    for raw in (None, "", "nope", 7, [], {"agents": 3}, {"agents": {"a": 1}}):
        assert Grants.from_obj(raw).agents == frozenset()
    assert Grants.from_json("{not json") == Grants.none()
    assert Grants.from_json(json.dumps({"agents": "*"})).is_all("agents")
    # Missing dimensions are "none", never "all".
    partial = Grants.from_obj({"agents": ["jarvis"]})
    assert not partial.allows("models", "anything")
    assert not partial.allows("tools", "read")


def test_lists_are_trimmed_deduplicated_and_agents_lowercased():
    g = Grants.from_obj({"agents": [" Jarvis ", "jarvis", ""], "models": ["gpt-5", "gpt-5"]})
    assert g.to_obj()["agents"] == ["jarvis"]
    assert g.to_obj()["models"] == ["gpt-5"]
    assert g.allows("agents", "JARVIS")


def test_allows_matches_any_candidate_and_models_are_exact():
    g = Grants.from_obj({"models": ["claude-opus-5-5"]})
    assert g.allows("models", "", "opus", "claude-opus-5-5")
    assert not g.allows("models", "Claude-Opus-5-5")
    assert not g.allows("models")


def test_tool_grants_accept_glob_patterns():
    g = Grants.from_obj({"tools": ["read", "mcp__tavily__*"]})
    assert g.allows("tools", "read")
    assert g.allows("tools", "mcp__tavily__search")
    assert not g.allows("tools", "mcp__github__search")
    assert not g.allows("tools", "bash")


def test_intersection_and_subset():
    a = Grants.from_obj({"agents": "*", "models": ["m1", "m2"], "tools": ["read", "bash"], "skills": "*"})
    b = Grants.from_obj({"agents": ["jarvis"], "models": "*", "tools": ["read"], "skills": []})
    both = a.intersect(b)
    assert both.to_obj() == {
        "agents": ["jarvis"],
        "models": ["m1", "m2"],
        "tools": ["read"],
        "skills": [],
        "folders": [],
    }
    assert both.is_subset_of(a) and both.is_subset_of(b)
    assert not a.is_subset_of(b)
    assert b.is_subset_of(Grants.all())
    assert not Grants.all().is_subset_of(b)


def test_filter_tool_names_drops_ungranted_tools_and_skill_without_skills():
    names = {"read", "bash", "Skill", "mcp__tavily__search"}
    assert filter_tool_names(names, None) == names
    g = Grants.from_obj({"tools": ["read", "Skill", "mcp__tavily__*"], "skills": [], "folders": "*"})
    assert filter_tool_names(names, g) == {"read", "mcp__tavily__search"}
    g2 = Grants.from_obj({"tools": ["read", "Skill"], "skills": ["pdf"]})
    assert filter_tool_names(names, g2) == {"read", "Skill"}


def test_delegation_refusal_covers_agent_external_engine_and_model():
    assert delegation_refusal(None, "codex", external=True) is None
    g = Grants.from_obj({"agents": ["explore", "codex"], "models": ["qwen"], "tools": ["read"], "skills": "*"})
    assert "not granted" in delegation_refusal(g, "plan", external=False)
    assert "All tools" in delegation_refusal(g, "codex", external=True)
    assert "not granted" in delegation_refusal(g, "explore", external=False, model="gpt-5")
    assert delegation_refusal(g, "explore", external=False, model="qwen") is None
    assert delegation_refusal(g, "explore", external=False) is None
    full = Grants.from_obj({"agents": ["codex"], "models": "*", "tools": "*", "skills": "*", "folders": "*"})
    assert delegation_refusal(full, "codex", external=True) is None


def test_stamp_rides_block_zero_and_foreign_stamps_are_stripped():
    stamp = AccessStamp("bob@example.com", Grants.from_obj({"agents": ["jarvis"]}))
    forged = {ACCESS_META_KEY: AccessStamp("bob@example.com", Grants.all()).to_obj()}
    blocks = [
        {"type": "text", "text": "hi", "_meta": dict(forged)},
        {"type": "text", "text": "more", "_meta": dict(forged, keep="me")},
    ]
    out = stamp_blocks(blocks, stamp)
    assert stamp_from_blocks(out) == stamp
    assert ACCESS_META_KEY not in out[1]["_meta"] and out[1]["_meta"]["keep"] == "me"
    assert blocks[0]["_meta"] == forged  # input untouched
    cleaned = strip_access_meta(blocks)
    assert stamp_from_blocks(cleaned) is None
    # Only block 0 counts: a stamp on a later block is ignored.
    assert stamp_from_blocks([{"type": "text", "text": "x"}, blocks[1]]) is None


def test_merge_stamps_intersects_restricted_authors():
    admin = AccessStamp("admin@example.com", Grants.all())
    alice = AccessStamp("alice@example.com", Grants.from_obj({"agents": ["jarvis", "plan"], "tools": "*"}))
    bob = AccessStamp("bob@example.com", Grants.from_obj({"agents": ["jarvis"], "tools": ["read"]}))
    assert merge_stamps([None, None]) is None
    assert merge_stamps([admin, None]).grants.is_unrestricted()
    merged = merge_stamps([admin, alice, bob])
    assert merged.grants.to_obj()["agents"] == ["jarvis"]
    assert merged.grants.to_obj()["tools"] == ["read"]
    assert AccessStamp.from_obj(merged.to_obj()) == merged


def test_folders_fail_closed_unless_star_and_intersect_by_path():
    assert Grants.from_obj({"agents": "*"}).folders == ()
    assert Grants.all().folders is None
    a = Grants.from_obj(
        {
            "folders": [{"path": "/src/a", "mode": "rw"}, {"path": "/src/b", "mode": "ro"}],
        }
    )
    b = Grants.from_obj({"folders": [{"path": "/src/a", "mode": "ro"}]})
    both = a.intersect(b)
    assert both.to_obj()["folders"] == [{"path": "/src/a", "mode": "ro"}]
    user_all_catalog = Grants.from_obj({"agents": "*", "models": "*", "tools": "*", "skills": "*"})
    assert not user_all_catalog.is_unrestricted()
    assert AccessStamp("a@x", user_all_catalog).restricted
    assert not AccessStamp("a@x", Grants.all()).restricted


def test_external_engines_need_all_folders_as_well():
    from sharedagents_core.access_grants import delegation_refusal

    confined = Grants.from_obj({"agents": ["codex", "plan"], "models": "*", "tools": "*", "skills": "*", "folders": []})
    refusal = delegation_refusal(confined, "codex", external=True)
    assert refusal and "folders" in refusal
    assert delegation_refusal(confined, "plan", external=False) is None
    free = Grants.from_obj({"agents": ["codex"], "models": "*", "tools": "*", "skills": "*", "folders": "*"})
    assert delegation_refusal(free, "codex", external=True) is None


def test_a_folder_confined_turn_never_sees_tools_that_run_outside_the_sandbox():
    # MCP servers and ACP agents are separate processes with their own
    # filesystem access, so the folder sandbox cannot confine them.
    names = {"read", "bash", "mcp__files__write_file", "mcp__search__query", "acp_call"}
    confined = Grants.from_obj({"agents": ["jarvis"], "tools": "*", "folders": []})
    assert filter_tool_names(names, confined) == {"read", "bash"}
    free = Grants.from_obj({"agents": ["jarvis"], "tools": "*", "folders": "*"})
    assert filter_tool_names(names, free) == names
