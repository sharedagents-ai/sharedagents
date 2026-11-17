"""Product-namespaced public ids (``<product>/<name>``).

Agents are exposed to OpenAI-compatible clients as ``sharedagents/<agent>``.
"""

from __future__ import annotations

#: Public id prefix; the only form writers emit and readers accept.
PRODUCT_ID_PREFIX = "sharedagents/"

#: Every prefix that qualifies a product-namespaced id.
ID_PREFIXES: tuple[str, ...] = (PRODUCT_ID_PREFIX,)


def bare_id(value: object) -> str:
    """Strip the product namespace prefix from ``value``.

    Loops so a doubly-prefixed value also collapses to the bare name. Ids
    namespaced by anything else (e.g. a model provider's ``openai/gpt-5``)
    are returned unchanged.
    """
    v = str(value or "").strip()
    while True:
        head, sep, tail = v.partition("/")
        if not sep or f"{head}/" not in ID_PREFIXES:
            return v
        v = tail
