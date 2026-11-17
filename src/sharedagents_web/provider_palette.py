"""Provider colours for the Models tab — one data file plus a total fallback.

This module is the single source of truth for provider colour. The chart code
(``static/models.js``) contains no provider→colour literal at all: it paints from
the ``colors`` map the API ships, which :func:`provider_color_map` builds here.

Provenance — read this before editing
-------------------------------------
Artificial Analysis publishes NO official provider palette. Verified:
their OpenAPI spec (``https://artificialanalysis.ai/api/v2/openapi``, 196,935
bytes, no auth) contains zero colour-adjacent tokens — no ``colou?r|hex|rgb|
hsl|palette|swatch|stroke|fill``, no ``#RRGGBB`` literals, no brand/logo/icon/
tint/hue fields — the live payload's ``model_creator`` is exactly
``{"id", "name"}`` (all 200 rows of page 1), and the only provider-level
endpoints are Commercial-only ("Free and Pro tier requests return a 403
subscription error"). The site itself is not scraped: the API is the only
permitted route.

The values in :data:`PROVIDERS` are therefore NOT Artificial Analysis' official
palette and will not be pixel-identical to artificialanalysis.ai's charts. They
are derived from:

(a) the colour names observed in the owner-supplied Artificial Analysis
    reference screenshot for the Pareto-chart legend, and
(b) publicly known lab brand colours, where the observed colour matches the
    lab's own brand colour — e.g. Google green ``#34A853``, Meta blue
    ``#0866FF``, NVIDIA green ``#76B900``, IBM blue ``#0F62FE``, Microsoft blue
    ``#0078D4``, Amazon orange ``#FF9900``, Anthropic clay ``#CC7832``.

No licence obligation attaches to these values: none is copied from a
third-party palette file (brand colours are not copyrightable as such), so no
third-party notice is recorded here. If a later edit copies a value from a
palette repository, cite that repository and its licence on that entry.

This file is DATA, not code. The owner may hand-edit the table freely: keys,
alias lists and hexes are all plain literals, and nothing in the chart code
depends on any particular value.

Deliberate collisions
---------------------
The table mirrors the reference legend rather than maximising distinguishability,
so several providers sit close together on purpose: ``alibaba`` / ``mistral`` /
``xiaomi`` are all orange-family, ``zai`` / ``meta`` / ``deepseek`` / ``kimi`` /
``ibm`` are all blue-family, and ``minimax`` and ``cursor`` share the same hex
``#F26898`` exactly. Do not "improve" these without the owner's say-so.
(A judgement, reported rather than silently changed: ``minimax`` and ``cursor``
are indistinguishable by colour alone. In the live cached payload only MiniMax
appears, so the pair does not currently collide inside one legend. If both ever
appear in one selection, the legend labels — not the swatches — are what tells
them apart.)

What the resolver does
----------------------
:func:`provider_color` normalises the input (lowercase, punctuation and
whitespace stripped), resolves the alias list, and — for a lab the table has
never heard of — falls back to a DETERMINISTIC slot in
:data:`FALLBACK_PALETTE`, seeded by the creator id when the payload carries one
and by the creator name otherwise. An unknown or brand-new lab therefore always
gets a stable, non-black colour that survives reloads; it never silently becomes
another provider's colour. The fallback palette is a fixed categorical list of
colour values (presentation data), not a provider→colour table.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: ``"curated"``: owner-supplied values derived as described above — never
#: presented as Artificial Analysis' own colours.
PALETTE_STATUS = "curated"

#: Where the "AA publishes no palette" finding was recorded (date, method).
VERIFICATION = (
    "2026-09-19",
    "no colour tokens in https://artificialanalysis.ai/api/v2/openapi; "
    "model_creator is exactly {id,name} on all 200 rows of page 1; "
    "provider-level endpoints are Commercial-only",
)

#: 32 distinct hues in two lightness bands, readable on the Models tab's
#: ``#151516`` background. This is the deterministic FALLBACK palette for
#: creators the table does not know; it is a categorical presentation palette (a
#: fixed list of colour values), not a provider→colour table.
FALLBACK_PALETTE: tuple[str, ...] = (
    "#da6262",
    "#da8f62",
    "#dabc62",
    "#cbda62",
    "#9eda62",
    "#71da62",
    "#62da80",
    "#62daad",
    "#62dada",
    "#62adda",
    "#6280da",
    "#7162da",
    "#9e62da",
    "#cb62da",
    "#da62bc",
    "#da628f",
    "#d34322",
    "#d38522",
    "#d3c822",
    "#9bd322",
    "#59d322",
    "#22d32d",
    "#22d36f",
    "#22d3b1",
    "#22b1d3",
    "#226fd3",
    "#222dd3",
    "#5922d3",
    "#9b22d3",
    "#d322c8",
    "#d32285",
    "#d32243",
)


@dataclass(frozen=True)
class ProviderColor:
    """One row of the palette table: key, legend label, colour, aliases."""

    key: str
    display: str
    color: str
    aliases: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        """Every published spelling this row answers to (key and label first)."""
        return (self.key, self.display) + self.aliases


# ---------------------------------------------------------------------------
# The table (DATA — the owner may hand-edit it; keys and aliases carry meaning)
#
# The rows are kept even when no model currently exists for a provider, so that
# a colour never shifts as the catalogue changes. ``xai`` deliberately carries
# the alias ``spacexai``: AA displays xAI as "SpaceXAI" in its legend today, and
# without the alias every xAI row would fall through to the hash fallback.
# ---------------------------------------------------------------------------
PROVIDERS: tuple[ProviderColor, ...] = (
    ProviderColor("google", "Google", "#34A853", ("Google DeepMind", "DeepMind")),
    ProviderColor("openai", "OpenAI", "#202123", ("Open AI",)),
    ProviderColor("anthropic", "Anthropic", "#CC7832", ("Claude",)),
    ProviderColor("meta", "Meta", "#0866FF", ("Facebook", "Meta AI", "Llama")),
    ProviderColor("nvidia", "NVIDIA", "#76B900", ("NVIDIA Corporation",)),
    ProviderColor("alibaba", "Alibaba", "#FF6A00", ("Alibaba Cloud", "Qwen", "Alibaba Group")),
    ProviderColor("mistral", "Mistral", "#FF6B35", ("Mistral AI", "mistralai")),
    ProviderColor("deepseek", "DeepSeek", "#4DA3FF", ("Deep Seek", "DeepSeek AI")),
    ProviderColor("xai", "xAI", "#8B5CF6", ("SpaceXAI", "Space X AI", "X AI", "Grok")),
    ProviderColor("zai", "Z AI", "#3B82F6", ("Z.AI", "Z-AI", "Zhipu", "Zhipu AI", "GLM")),
    ProviderColor("kimi", "Kimi", "#5B8DEF", ("Moonshot", "Moonshot AI", "MoonshotAI")),
    ProviderColor("minimax", "MiniMax", "#F26898", ("Mini Max", "MiniMax AI")),
    ProviderColor("xiaomi", "Xiaomi", "#FF8B3D", ("Xiaomi AI", "MiMo")),
    ProviderColor("stepfun", "StepFun", "#14B8A6", ("Step Fun", "StepFun AI")),
    ProviderColor("ibm", "IBM", "#0F62FE", ("IBM Research", "Granite")),
    ProviderColor("thinkingmachines", "Thinking Machines", "#9AA0A6", ("Thinking Machines Lab", "TML")),
    ProviderColor("microsoft", "Microsoft", "#0078D4", ("MSFT", "Microsoft Azure", "Azure")),
    ProviderColor("amazon", "Amazon", "#FF9900", ("AWS", "Amazon Web Services")),
    ProviderColor("cohere", "Cohere", "#39328B", ("Cohere AI",)),
    ProviderColor("perplexity", "Perplexity", "#20808D", ("Perplexity AI", "PPLX")),
    ProviderColor("cognition", "Cognition", "#D7EF39", ("Devin", "Cognition AI")),
    ProviderColor("cursor", "Cursor", "#F26898", ("Cursor AI", "Anysphere")),
)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def canonical_key(value: Any) -> str:
    """Fold a creator name/id to its lookup key: lowercase, no punctuation.

    ``"Z AI"``, ``"Z.AI"``, ``"z-ai"`` and ``"zai"`` all fold to ``"zai"``;
    ``"SpaceXAI"`` and ``"space x ai"`` fold to ``"spacexai"`` / ``"spacexai"``
    (the second through the alias list). A UUID creator id folds to its bare hex.
    Returns ``""`` for ``None``/empty.
    """
    return _PUNCT_RE.sub("", str(value if value is not None else "").lower())


def _build_index() -> tuple[dict[str, str], tuple[tuple[str, str, str], ...]]:
    """``{folded name: colour}`` plus any accidental key collisions.

    Built from the table, so the alias lists are the only place a spelling is
    declared. A collision keeps the first row and is reported in
    :data:`PALETTE_CONFLICTS` for a test to catch — silently overwriting would
    make a colour depend on table order.
    """
    index: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    for entry in PROVIDERS:
        for name in entry.names():
            key = canonical_key(name)
            if not key:
                continue
            existing = index.get(key)
            if existing is None:
                index[key] = entry.color
            elif existing != entry.color:
                conflicts.append((key, existing, entry.color))
    return index, tuple(conflicts)


PROVIDER_INDEX, PALETTE_CONFLICTS = _build_index()

#: Every colour the table can return — the fallback must never land on one of
#: these, or an unknown lab would be indistinguishable from a known one.
TABLE_COLORS: frozenset[str] = frozenset(entry.color for entry in PROVIDERS)


def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit — stable across processes and Python versions.

    ``hash()`` is randomized per process (PYTHONHASHSEED), so it can never seed
    a colour that has to survive a reload.
    """
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def palette_size() -> int:
    """Number of slots in the deterministic fallback palette."""
    return len(FALLBACK_PALETTE)


def fallback_color(seed: Any) -> str:
    """The colour an unknown creator gets: a deterministic fallback-palette slot.

    Public so the fallback can be tested on its own; callers normally want
    :func:`provider_color`.
    """
    key = canonical_key(seed) or "unknown"
    return FALLBACK_PALETTE[_fnv1a_32(key.encode("utf-8")) % len(FALLBACK_PALETTE)]


def is_known(name: Any, creator_id: Any = None) -> bool:
    """True when the table (not the hash fallback) resolves this creator."""
    return any(canonical_key(candidate) in PROVIDER_INDEX for candidate in (name, creator_id))


def provider_color(name: Any, creator_id: Any = None) -> str:
    """The ONE colour resolver used by every chart on the Models tab.

    Precedence:

    1. the :data:`PROVIDERS` table, matched on the folded creator name and then
       on the folded creator id — so the alias lists absorb published name drift
       (``"SpaceXAI"`` → ``xai``, ``"Moonshot AI"`` → ``kimi``, ``"Z.AI"`` →
       ``zai``);
    2. :func:`fallback_color` seeded by the creator id when the payload carries
       one (a rename cannot then move the colour), else by the creator name.

    Every creator therefore has a colour — including one this module has never
    seen — and it is the same colour on every call, every reload and every chart.
    """
    for candidate in (name, creator_id):
        mapped = PROVIDER_INDEX.get(canonical_key(candidate))
        if mapped:
            return mapped
    return fallback_color(canonical_key(creator_id) or canonical_key(name))


def provider_color_map(pairs: Iterable[tuple[Any, Any]]) -> dict[str, str]:
    """``{creator name: colour}`` for ``(name, creator_id)`` pairs.

    This is the map the API ships to the browser, so the page can colour every
    point, bar and legend entry without re-deriving anything. Names are kept
    exactly as published (the legend labels are the published names); the
    resolver is what folds them.
    """
    out: dict[str, str] = {}
    for name, creator_id in pairs:
        label = str(name or "").strip()
        if label and label not in out:
            out[label] = provider_color(label, creator_id)
    return out
