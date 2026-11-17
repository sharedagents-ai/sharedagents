"""Engine-neutral parsing of explicit local output links.

Engines do not all expose the same native resource-link protocol. The shared
chat output contract therefore permits an assistant to link an existing local
deliverable with Markdown. This module only recognizes explicit, output-like
links; authorization, hashing, and Web-store materialization remain the
responsibility of the session writer.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from sharedagents_core.cwdutil import safe_getcwd

_OUTPUT_LINK_RE = re.compile(r"\[([^\]\n]{1,512})\]\(\s*(<[^>\n]+>|[^)\n]+)\s*\)")
_OUTPUT_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".csv",
        ".tsv",
        ".xls",
        ".xlsx",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".zip",
        ".7z",
        ".tgz",
        ".gz",
        ".tar",
        ".parquet",
        ".feather",
        ".mp3",
        ".m4a",
        ".wav",
        ".mp4",
        ".mov",
        ".webm",
    }
)
_OUTPUT_LABELS = (
    "download",
    "attachment",
    "attached",
    "artifact",
    "output",
    "screenshot",
    "report",
    "file",
)


def local_artifact_references(
    text: str,
    spawn_cwd: str | None,
) -> list[dict[str, Any]]:
    """Extract existing, explicit local-file output links from assistant text.

    Ordinary source citations remain prose links unless the target has a known
    deliverable suffix or the link label explicitly identifies it as an output.
    Remote URLs, query/fragment-bearing links, and missing files are ignored.
    The returned references are intentionally untrusted until the conversation
    store validates and materializes them.
    """
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _OUTPUT_LINK_RE.finditer(text or ""):
        label = match.group(1).strip()
        target = match.group(2).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        try:
            parsed = urlsplit(target)
        except ValueError:
            continue
        scheme = parsed.scheme.lower()
        if scheme and scheme != "file":
            continue
        if scheme == "file" and parsed.netloc not in {"", "localhost"}:
            continue
        if parsed.query or parsed.fragment:
            continue
        raw_path = unquote(parsed.path if scheme else target)
        if not raw_path or "\x00" in raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(spawn_cwd or safe_getcwd()) / path
        # Clickable source citations often carry :line or :line:column. Only
        # strip that suffix when the literal path is not itself a file.
        if not path.is_file():
            without_line = re.sub(r":\d+(?::\d+)?$", "", str(path))
            path = Path(without_line)
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        label_low = label.lower()
        explicitly_labelled = any(re.search(rf"\b{re.escape(word)}\b", label_low) for word in _OUTPUT_LABELS)
        if suffix not in _OUTPUT_SUFFIXES and not explicitly_labelled:
            continue
        canonical = str(path)
        if canonical in seen:
            continue
        seen.add(canonical)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        references.append(
            {
                "path": canonical,
                "locator": canonical,
                "display_name": path.name,
                "media_type": media_type,
                "storage_scope": "local_session",
            }
        )
    return references


# ── canonical artifact/permission scrubbing + local enrichment ────────────
# Pure canonical artifact and permission helpers; no secondary store state.


def _canonical_permission_decisions(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Allowlist resolved permission metadata; never retain arguments/reasons."""
    normalized: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            continue
        value = str(decision.get("decision") or "").lower()
        if value not in {"allow", "deny"}:
            continue
        criticality = str(decision.get("criticality") or "mutating").lower()
        if criticality not in {
            "read_only",
            "mutating",
            "destructive",
            "network",
            "forbidden",
        }:
            criticality = "mutating"
        source = str(decision.get("source") or "policy").lower()
        if source not in {"policy", "hook", "user", "auto"}:
            source = "policy"
        normalized.append(
            {
                "call_id": str(decision.get("call_id") or f"decision-{index}")[:256],
                "tool_name": str(decision.get("tool_name") or "")[:256],
                "decision": value,
                "criticality": criticality,
                "source": source,
            }
        )
    return normalized


def _canonical_artifact_references(
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Allowlist explicit artifact metadata and reject inline/secret bodies."""
    from urllib.parse import urlsplit, urlunsplit

    normalized: list[dict[str, Any]] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        raw_uri = str(reference.get("uri") or "")
        raw_path = str(reference.get("path") or "")
        item: dict[str, Any] = {}
        if raw_uri and len(raw_uri) <= 8192:
            try:
                parsed = urlsplit(raw_uri)
            except ValueError:
                continue
            scheme = parsed.scheme.lower()
            if scheme in {"http", "https"} and parsed.netloc:
                safe_uri = urlunsplit(
                    (
                        scheme,
                        parsed.netloc.rsplit("@", 1)[-1],
                        parsed.path,
                        "",
                        "",
                    )
                )
                item["uri"] = safe_uri
                item["locator"] = safe_uri
                item["storage_scope"] = "remote"
            elif scheme == "file":
                safe_uri = urlunsplit(
                    (
                        scheme,
                        parsed.netloc,
                        parsed.path,
                        "",
                        "",
                    )
                )
                item["uri"] = safe_uri
                item["locator"] = safe_uri
                item["storage_scope"] = "local_session"
        elif raw_path and len(raw_path) <= 4096 and not any(ord(char) < 32 for char in raw_path):
            item["path"] = raw_path
            item["locator"] = raw_path
            scope = str(reference.get("storage_scope") or "local_session")
            item["storage_scope"] = scope if scope in {"local_session", "artifact_store"} else "local_session"
        if not item:
            continue
        digest = str(reference.get("sha256") or "").lower()
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            item["sha256"] = digest
        media_type = str(reference.get("media_type") or "")[:255]
        if media_type:
            item["media_type"] = media_type
        display_name = str(reference.get("display_name") or "")[:512]
        if display_name:
            item["display_name"] = display_name
        try:
            size = max(0, int(reference.get("size_bytes") or 0))
        except (TypeError, ValueError):
            size = 0
        if size:
            item["size_bytes"] = size
        normalized.append(item)
    return normalized


def enrich_local_artifact_reference(
    reference: dict[str, Any],
    *,
    workspace_cwd: str,
    sessions_dir: Path,
    agent_home: Path,
    artifact_store_key: str = "",
) -> dict[str, Any] | None:
    """Resolve, hash, and optionally persist one authorized local output.

    Standard ACP ResourceLink has no digest field. Enrichment therefore
    happens inside sharedagents, under server-authoritative roots, before
    canonical projection. When ``artifact_store_key`` is present, the
    verified bytes are atomically copied into the authenticated Web asset
    store while the source is still available. Remote/data/opaque locators
    are never opened.
    """
    import tempfile
    from urllib.parse import unquote, urlsplit

    from sharedagents_core.layout import sharedagents_root

    scope = str(reference.get("storage_scope") or "local_session")
    if scope not in {"local_session", "artifact_store"}:
        return None
    locator = str(reference.get("locator") or reference.get("uri") or reference.get("path") or "")
    staged_path: Path | None = None
    source = None
    try:
        parsed = urlsplit(locator)
        if parsed.scheme:
            if (
                parsed.scheme.lower() != "file"
                or parsed.netloc not in {"", "localhost"}
                or parsed.query
                or parsed.fragment
            ):
                return None
            raw_path = unquote(parsed.path)
        else:
            raw_path = locator
        if not raw_path or "\x00" in raw_path:
            return None
        path = Path(raw_path).expanduser().resolve(strict=True)
        session_root = Path(sessions_dir).resolve()
        workspace_root = Path(workspace_cwd).expanduser().resolve(strict=True) if workspace_cwd else None
        if workspace_root is not None and not workspace_root.is_dir():
            workspace_root = None
        web_artifact_root = (
            Path(os.environ.get("SHAREDAGENTS_WEB_ARTIFACTS_DIR") or (sharedagents_root() / "web" / "artifacts"))
            .expanduser()
            .resolve()
        )
        artifact_roots = {
            (session_root / "artifacts").resolve(),
            (Path(agent_home).resolve() / "artifacts").resolve(),
            web_artifact_root,
        }
        roots = artifact_roots
        if scope == "local_session":
            # CLI adapters can only identify a filesystem link as local;
            # they cannot reliably know that the path was materialized by
            # an earlier turn. Accept server-owned artifact roots in both
            # scopes so a resumed conversation can re-attach a durable
            # output without granting access to arbitrary host paths.
            roots = {
                *artifact_roots,
                session_root,
                *([workspace_root] if workspace_root is not None else []),
            }
        if not path.is_file() or not any(path == root or root in path.parents for root in roots):
            return None
        store_key = str(artifact_store_key or "")
        if store_key and (
            store_key in {".", ".."}
            or "/" in store_key
            or "\\" in store_key
            or "\x00" in store_key
            or any(ord(char) < 32 for char in store_key)
        ):
            return None
        safe_name = os.path.basename(str(reference.get("display_name") or path.name or "artifact")).replace("\x00", "")[
            :240
        ]
        if not safe_name or safe_name in {".", ".."}:
            safe_name = "artifact"

        source = path.open("rb")
        size = os.fstat(source.fileno()).st_size
        declared_size = reference.get("size_bytes")
        if declared_size not in (None, "", 0, "0") and int(declared_size) != size:
            source.close()
            return None
        hasher = hashlib.sha256()
        staged_handle = None
        if store_key:
            staging_dir = web_artifact_root / store_key / ".staging"
            staging_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            staged_handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="artifact-",
                dir=staging_dir,
                delete=False,
            )
            staged_path = Path(staged_handle.name)
        try:
            with source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(chunk)
                    if staged_handle is not None:
                        staged_handle.write(chunk)
            if staged_handle is not None:
                staged_handle.flush()
                os.fsync(staged_handle.fileno())
                staged_handle.close()
                staged_handle = None
        finally:
            if staged_handle is not None:
                staged_handle.close()
        digest = hasher.hexdigest()
        declared_hash = str(reference.get("sha256") or "").lower()
        if declared_hash and declared_hash != digest:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            return None
        if staged_path is not None:
            target_dir = web_artifact_root / store_key / digest
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = target_dir / safe_name
            os.chmod(staged_path, 0o600)
            os.replace(staged_path, target)
            path = target.resolve(strict=True)
            scope = "artifact_store"
    except (OSError, TypeError, ValueError):
        try:
            if source is not None:
                source.close()
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    enriched = dict(reference)
    enriched.update(
        {
            "path": str(path),
            "locator": str(path),
            "storage_scope": scope,
            "display_name": safe_name,
            "size_bytes": size,
            "sha256": digest,
        }
    )
    enriched.pop("uri", None)
    return enriched
