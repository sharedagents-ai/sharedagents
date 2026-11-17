"""Prompt-token counting for self-served models, before the request is sent.

The character estimate (``chars / 3``) is wrong in both directions: 2.5x too low
on dense text (hex, ids, minified JSON), ~30% too high on source code and prose.
A context budget built on it either overflows or compacts early. A self-served
model ships its ``tokenizer.json``; counting with it takes ~50 ms per 100K
characters and matched vLLM's own ``/tokenize`` to 0.1% (74,686 vs 74,738).

The tokenizer is found from the model row (``tokenizerPath``) or discovered in
``$SHAREDAGENTS_MODELS_DIR`` (default ``~/models``) by model id, so a quantised
variant of a known family reuses its tokenizer. Rows with no tokenizer (hosted
APIs) fall back to the character estimate; the caller keeps a safety margin and
the reactive overflow handler as the last resort.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: ChatML-style framing per message (``<|im_start|>role\n`` + ``<|im_end|>\n``).
#: A little above the real 5 so an unusual template cannot undercount.
TEMPLATE_TOKENS_PER_MESSAGE = 6
#: The generation prompt (``<|im_start|>assistant\n`` + thinking opener).
GENERATION_PROMPT_TOKENS = 8
#: The chat template wraps ``tools`` in fixed instructions (Qwen: "# Tools ... You
#: may call one or more functions ..."); measured ~1,000 tokens against vLLM.
TOOLS_FRAMING_TOKENS = 1100
#: Image parts are billed as visual tokens, not text; charge a generous flat rate.
IMAGE_TOKENS = 1500
#: Fallback density when no tokenizer is available.
FALLBACK_CHARS_PER_TOKEN = 3.0

_lock = threading.Lock()
_loaded: dict[str, Any] = {}  # resolved path -> Tokenizer | None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _family(model_id: str) -> str:
    """``qwen3.8-27b-int4-w4a16`` -> ``qwen3827b`` (the part a quantisation keeps)."""
    match = re.match(r"^(.*?\d+b)", str(model_id or "").lower())
    return _norm(match.group(1)) if match else ""


def models_dir() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_MODELS_DIR") or "~/models").expanduser()


def tokenizer_path_for_row(row: Mapping[str, Any] | None) -> Path | None:
    """The ``tokenizer.json`` for a model row, or ``None``."""
    payload = row if isinstance(row, Mapping) else {}
    explicit = str(payload.get("tokenizerPath") or payload.get("tokenizer_path") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_dir():
            path = path / "tokenizer.json"
        return path if path.is_file() else None
    model_id = str(payload.get("id") or "")
    if not model_id:
        return None
    root = models_dir()
    try:
        candidates = [d for d in root.iterdir() if (d / "tokenizer.json").is_file()]
    except OSError:
        return None
    wanted, family = _norm(model_id), _family(model_id)
    for directory in candidates:
        if _norm(directory.name) == wanted:
            return directory / "tokenizer.json"
    if family:
        for directory in candidates:
            if _norm(directory.name).startswith(family):
                return directory / "tokenizer.json"
    return None


def tokenizer_for_row(row: Mapping[str, Any] | None) -> Any:
    """A loaded ``tokenizers.Tokenizer`` for the row, cached, or ``None``."""
    path = tokenizer_path_for_row(row)
    if path is None:
        return None
    key = str(path)
    with _lock:
        if key in _loaded:
            return _loaded[key]
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(key)
    except Exception:  # noqa: BLE001 — missing library / unreadable file: estimate instead
        tokenizer = None
    with _lock:
        _loaded[key] = tokenizer
    return tokenizer


def _texts_of(message: Mapping[str, Any]) -> tuple[list[str], int]:
    """The text pieces of one message and its image count."""
    texts: list[str] = []
    images = 0
    content = message.get("content")
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, Mapping):
                kind = str(part.get("type") or "")
                if kind == "text":
                    texts.append(str(part.get("text") or ""))
                elif kind in ("image_url", "image", "input_image"):
                    images += 1
                else:
                    texts.append(json.dumps(part, ensure_ascii=False))
            elif part:
                texts.append(str(part))
    elif content:
        texts.append(str(content))
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        texts.append(reasoning)
    for call in message.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args if args is not None else {}, ensure_ascii=False)
        texts.append(str(fn.get("name") or call.get("name") or ""))
        texts.append(args)
    return texts, images


def count_payload_tokens(
    messages: list[Mapping[str, Any]],
    tools: list[Mapping[str, Any]] | None = None,
    tokenizer: Any = None,
) -> int:
    """Tokens the upstream will see for ``messages`` + ``tools``.

    Exact (to the chat-template framing) with a tokenizer; otherwise a
    ``chars / 3`` estimate over the same text. Never raises.
    """
    texts: list[str] = []
    images = 0
    framing = GENERATION_PROMPT_TOKENS
    for message in messages or []:
        if not isinstance(message, Mapping):
            continue
        framing += TEMPLATE_TOKENS_PER_MESSAGE
        pieces, image_count = _texts_of(message)
        texts.extend(pieces)
        images += image_count
        if message.get("tool_calls"):
            framing += 8 * len(message["tool_calls"])
    if tools:
        texts.append(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
        framing += TOOLS_FRAMING_TOKENS
    total = framing + images * IMAGE_TOKENS
    texts = [t for t in texts if t]
    if tokenizer is not None:
        try:
            total += sum(len(enc.ids) for enc in tokenizer.encode_batch(texts, add_special_tokens=False))
            return int(total)
        except Exception:  # noqa: BLE001 — fall through to the estimate
            pass
    total += int(sum(len(t) for t in texts) / FALLBACK_CHARS_PER_TOKEN) + 1
    return int(total)
