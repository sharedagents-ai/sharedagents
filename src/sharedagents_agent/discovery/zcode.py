"""ZCode (Z.AI GLM Coding Plan) model catalogue.

The ZCode CLI is the authority for the id it runs; sharedagents does not invent a
wire catalogue. The **headless** engine (``zcode -p … --json``) has no
``--model`` flag — model choice lives in
``~/.zcode/v2/provider_config.json`` →
``config.defaultModelSelection.modelId`` — so the effective catalogue is
exactly that configured id (:func:`resolve_zcode_default` /
:func:`discover_zcode_models`). :data:`ZCODE_CHAT_MODELS` records the GLM chat
ids the installed CLI ships, and is used only to recognise and normalise an
operator-supplied id (:func:`canonical_zcode_model`), never to advertise a
model the engine cannot select.

Image/video/embedding-only GLM ids (``glm-4v-*``, ``glm-4.6v*``,
``glm-5v-turbo``) are rejected: ``zcode`` drives a conversational coding
session, not a vision endpoint.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: GLM ids shipped by ``zcode-app-cli 3.14.4-30`` that can drive a coding
#: session, newest first. Ids are stored lowercase (the CLI accepts the
#: uppercase ``GLM-5.3`` display form; :func:`canonical_zcode_model` maps
#: either case onto these keys).
ZCODE_CHAT_MODELS: tuple[str, ...] = (
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5-turbo",
    "glm-5.2",
    "glm-5.1",
    "glm-5.1-highspeed",
    "glm-5",
    "glm-4.7",
    "glm-4.7-flash",
    "glm-4.6",
    "glm-4.5-air",
)

_ZCODE_MODEL_ID = re.compile(r"^glm-[0-9][a-z0-9.\-]*$")

#: Engine labels that are NEVER wire model ids.
_ZCODE_LABELS = frozenset({"zcode", "z-code", "zai", "z.ai"})


def is_zcode_chat_model(model_id: str) -> bool:
    """True for a GLM chat id ``zcode --model`` can drive."""
    mid = str(model_id or "").strip().lower()
    if not mid or mid in _ZCODE_LABELS:
        return False
    if not _ZCODE_MODEL_ID.match(mid):
        return False
    # Vision / video ids are not coding-session models.
    return not (re.match(r"^glm-[0-9]+(\.[0-9]+)?v", mid) or mid.endswith("v"))


def canonical_zcode_model(model_id: str) -> str:
    """Map a user-supplied GLM id onto the CLI's own casing.

    ``GLM-5.3`` (the form the CLI writes into its config) and ``glm-5.3``
    both resolve to the catalogue entry; an unknown-but-well-formed GLM id
    is trusted verbatim (the installed CLI stays the authority for ids
    newer than this list). Returns ``""`` for anything that is not a GLM
    chat id.
    """
    mid = str(model_id or "").strip()
    if not mid:
        return ""
    lowered = mid.lower()
    if not is_zcode_chat_model(lowered):
        return ""
    for known in ZCODE_CHAT_MODELS:
        if known == lowered:
            return known
    return lowered


def _catalog_files() -> list[Any]:
    """The CLI's synced provider catalogue(s), newest first."""
    from sharedagents_agent.delegates.zcode import _zcode_home

    root = _zcode_home() / "v2" / "runtime" / "provider"
    try:
        files = list(root.glob("*/*/endpoint-*/zcode-builtin.json"))
    except OSError:
        return []
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


def _configured_provider_id() -> str:
    from sharedagents_agent.delegates.zcode import _zcode_home

    try:
        data = json.loads((_zcode_home() / "v2" / "provider_config.json").read_text(encoding="utf-8"))
        sel = data["config"]["defaultModelSelection"]
        return str(sel.get("providerId") or "")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return ""


def zcode_catalog_model_ids() -> list[str]:
    """Model ids the CLI itself offers this login (its own ``/model`` picker).

    Read from the CLI's synced catalogue (``zcode-builtin.json`` →
    ``providerConfigRules.templateRules[].config.builtinModelIds``), picking
    the template that matches the signed-in account: a ``coding-plan``
    provider id → the "Coding Plan" template, anything else → "Z.ai API".
    Original spelling is preserved (``GLM-5.3-Flash``) because the id is
    written back verbatim into a provider config. Chat-only: vision ids are
    dropped. ``[]`` when the catalogue is unreadable.
    """
    want = "coding plan" if "coding-plan" in _configured_provider_id() else "api"
    for path in _catalog_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rules = data["config"]["providerConfigRules"]["templateRules"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for rule in rules if isinstance(rules, list) else []:
            if not isinstance(rule, dict):
                continue
            names = rule.get("templateNameMap")
            label = str((names or {}).get("en-US") or "").lower()
            if want not in label:
                continue
            ids = (rule.get("config") or {}).get("builtinModelIds")
            if isinstance(ids, list):
                out = [i.strip() for i in ids if isinstance(i, str) and is_zcode_chat_model(i)]
                if out:
                    return out
    return []


def zcode_catalog_spelling(model_id: str) -> str:
    """The catalogue's exact spelling of ``model_id`` (case-insensitive), else ""."""
    want = str(model_id or "").strip().lower()
    for known in zcode_catalog_model_ids():
        if known.lower() == want:
            return known
    return ""


def discover_zcode_models() -> list[dict[str, Any]]:
    """Return the model rows for the ZCode engine's ``/model`` picker.

    Everything the CLI offers this login (:func:`zcode_catalog_model_ids`).
    Headless ``zcode -p`` has no ``--model`` flag, but the runner selects the
    model per run through an isolated provider config
    (``ZCODE_PERSONAL_PROVIDER_CONFIG_FILE`` with
    ``defaultModelSelection.modelId`` set), so every listed id is honoured.
    If the catalogue cannot be read, only the CLI's configured model is
    offered.
    """
    from sharedagents_agent.delegates.zcode import _zcode_configured_model

    configured = _zcode_configured_model()
    ids = zcode_catalog_model_ids()
    if configured and not any(i.lower() == configured.lower() for i in ids):
        ids.insert(0, configured)
    return [{"id": i, "name": i, "provider": "zai", "source": "zcode-catalog"} for i in ids]


def zcode_modelspec_label(model_id: str) -> str:
    return str(model_id or "").strip()


def resolve_zcode_default() -> str:
    """Engine-effective default: the CLI's own configured model, else ""."""
    from sharedagents_agent.delegates.zcode import _zcode_configured_model

    return _zcode_configured_model()


__all__ = [
    "ZCODE_CHAT_MODELS",
    "canonical_zcode_model",
    "discover_zcode_models",
    "is_zcode_chat_model",
    "resolve_zcode_default",
    "zcode_catalog_model_ids",
    "zcode_catalog_spelling",
]
