"""Native Bedrock route identity and validation (pure — no SDK imports).

A model row routes to the native Amazon Bedrock Converse transport ONLY when
it explicitly declares ``api: "bedrock-converse"``. The ``baseUrl``
(``bedrock://REGION``), ``backend`` and ``provider`` fields are display /
session-routing metadata and are NEVER sufficient to infer a Bedrock route —
an OpenAI-compatible proxy that happens to sit in front of Bedrock stays on
the OpenAI wire.

``validate_bedrock_row`` fails closed: a row that declares the route but is
missing or internally inconsistent (region vs URI, auth enum, malformed
model id / inference-profile ARN, credential metadata, context window) raises
:class:`BedrockConfigError` instead of degrading to a guess.

``worker_env_for_model`` (in ``sharedagents_gateway.worker_request``) builds the
per-subprocess credential environment on top of this module:

* ``auth: "bedrock-api-key"`` — the bearer token value (resolved from the
  per-agent ``secrets.env`` first, then the inherited environment) is set
  ONLY in the child worker's ``AWS_BEARER_TOKEN_BEDROCK``; botocore reads
  exactly that variable and prefers bearer auth for the ``bedrock`` signing
  name when it is present.
* ``auth: "aws-iam"`` — any ambient ``AWS_BEARER_TOKEN_BEDROCK`` is REMOVED
  from the child environment so the SDK signs with the normal IAM chain
  (env / shared config / profile / instance role) even when the gateway
  process holds a bearer token for another agent's API-key row.

Traceability: plan/bedrock-native-converse.md (BED-01, BED-02)
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

#: The explicit route marker. The ONLY signal that selects the native
#: Converse transport — never inferred from URL/backend/provider.
BEDROCK_API = "bedrock-converse"
#: Display / session-routing backend label (kept out of the
#: ``sharedagents_model.providers`` backend registry on purpose).
BEDROCK_BACKEND = "bedrock"
#: Display URL scheme for the row's ``baseUrl`` (``bedrock://REGION``).
BEDROCK_BASE_SCHEME = "bedrock"

#: The environment variable botocore reads for the Bedrock bearer token
#: (signing name ``bedrock`` → ``AWS_BEARER_TOKEN_BEDROCK``). The child
#: worker's env is the only place it is ever set for API-key rows.
BEDROCK_BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"

AUTH_API_KEY = "bedrock-api-key"
AUTH_IAM = "aws-iam"
AUTH_VALID: tuple[str, ...] = (AUTH_API_KEY, AUTH_IAM)

#: Assistant round-message field carrying a Bedrock model's opaque / signed
#: reasoning for that round's tool calls (``{"model", "blocks"}``), so the
#: next Converse round can hand it back verbatim. Provider-private: the wire
#: sanitizer strips it from every non-Bedrock request.
BEDROCK_REASONING_FIELD = "bedrock_reasoning"

#: The row capability that opts a Bedrock row into the forced named
#: ``tool_choice`` (SDLC). Rows without it always send ``tool_choice: "auto"``
#: — the picker/docs never promise enforcement on a row that did not opt in.
NAMED_TOOL_CHOICE_CAPABILITY = "named-tool-choice"

# Validate syntax without freezing the SDK's evolving region catalogue.
_REGION_RE = re.compile(r"^[a-z]{2}(?:-[a-z]+)+-\d+$")
#: Foundation model ids: ``vendor.name-…`` with an optional ``:version``
#: suffix (e.g. ``anthropic.claude-sonnet-4-5-20250929-v1:0``).
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
#: Inference-profile ARNs (partition: aws / aws-cn / aws-us-gov).
_ARN_RE = re.compile(
    r"^arn:(?:aws|aws-cn|aws-us-gov):bedrock:([a-z0-9-]+):(\d{12})"
    r":(?:application-inference-profile|inference-profile)/[A-Za-z0-9._:-]+$"
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class BedrockConfigError(ValueError):
    """A model row declares a Bedrock route that cannot be used safely.

    Raised at validation time (config load / request build / worker start)
    with a message that names the offending FIELD but never a secret value.
    """


@dataclass(frozen=True)
class BedrockRoute:
    """The validated identity of one native Bedrock deployment row."""

    region: str
    model_id: str
    auth: Literal["bedrock-api-key", "aws-iam"]
    #: Optional boto3 named-profile (``AWS_PROFILE`` equivalent) for the
    #: child worker. ``None`` = the normal credential chain.
    profile: str | None = None


def is_bedrock_row(row: Mapping[str, Any] | None) -> bool:
    """True ONLY when the row explicitly declares ``api='bedrock-converse'``.

    A ``bedrock://`` baseUrl, a ``backend: "bedrock"`` label, or a
    ``provider`` value is display/session metadata and never routes a
    request to the native transport.
    """
    if not isinstance(row, Mapping):
        return False
    return str(row.get("api") or "").strip().lower() == BEDROCK_API


def bedrock_named_tool_choice(row: Mapping[str, Any] | None) -> bool:
    """True when the row explicitly opts into the forced named tool choice.

    Reuses the row's ``capabilities`` list (case-insensitive) so the wizard,
    doctor and picker read one declaration.
    """
    if not isinstance(row, Mapping):
        return False
    caps = row.get("capabilities") or []
    if not isinstance(caps, list):
        return False
    return any(str(c).strip().lower() == NAMED_TOOL_CHOICE_CAPABILITY for c in caps)


def is_valid_bedrock_region(region: str) -> bool:
    """True when ``region`` looks like an AWS region id (``us-east-1`` …).

    Public so the setup wizard validates a typed region with the same rule
    ``validate_bedrock_row`` enforces, instead of duplicating the regex.
    """
    return bool(_REGION_RE.match(str(region or "").strip()))


def _row_field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def validate_bedrock_row(row: Mapping[str, Any] | None) -> BedrockRoute:
    """Validate a row that declares ``api='bedrock-converse'``.

    Raises :class:`BedrockConfigError` (fail closed) on any inconsistency;
    the message names the field, never a secret value.
    """
    if not isinstance(row, Mapping):
        raise BedrockConfigError("Bedrock row is missing")
    alias = str(row.get("alias") or row.get("id") or "?")

    api = str(row.get("api") or "").strip().lower()
    if api != BEDROCK_API:
        raise BedrockConfigError(f"model row {alias!r}: api must be {BEDROCK_API!r}, got {api!r}")

    region = str(row.get("region") or "").strip()
    if not region:
        raise BedrockConfigError(f"model row {alias!r}: 'region' is required for a Bedrock route")
    if not _REGION_RE.match(region):
        raise BedrockConfigError(f"model row {alias!r}: invalid AWS region {region!r}")

    base_url = str(_row_field(row, "baseUrl", "base_url") or "").strip()
    if not base_url:
        raise BedrockConfigError(f"model row {alias!r}: 'baseUrl' is required; use {BEDROCK_BASE_SCHEME}://{region}")
    scheme, sep, rest = base_url.partition("://")
    if not sep or scheme.strip().lower() != BEDROCK_BASE_SCHEME:
        raise BedrockConfigError(
            f"model row {alias!r}: Bedrock rows use a {BEDROCK_BASE_SCHEME}:// "
            "display URL (no arbitrary endpoint URLs); got "
            f"baseUrl={base_url!r}"
        )
    if rest.rstrip("/") != region:
        raise BedrockConfigError(f"model row {alias!r}: baseUrl region {rest!r} does not match region {region!r}")

    auth = str(row.get("auth") or "").strip().lower()
    if auth not in AUTH_VALID:
        raise BedrockConfigError(f"model row {alias!r}: 'auth' must be one of {list(AUTH_VALID)}, got {auth!r}")

    model_id = str(_row_field(row, "id", "model") or "").strip()
    if not model_id or model_id != str(_row_field(row, "id", "model")):
        raise BedrockConfigError(
            f"model row {alias!r}: 'id' (modelId / inference-profile ARN) "
            "must be non-blank and free of surrounding whitespace"
        )
    if model_id.startswith("arn:"):
        m = _ARN_RE.match(model_id)
        if not m:
            raise BedrockConfigError(f"model row {alias!r}: malformed inference-profile ARN {model_id!r}")
        if m.group(1) != region:
            raise BedrockConfigError(f"model row {alias!r}: ARN region {m.group(1)!r} does not match region {region!r}")
    elif not _MODEL_ID_RE.match(model_id):
        raise BedrockConfigError(
            f"model row {alias!r}: invalid model id {model_id!r} (no whitespace or path separators)"
        )

    profile = row.get("profile")
    if profile is not None:
        profile = str(profile).strip()
        if not profile or profile != str(row.get("profile")) or any(c.isspace() for c in profile):
            raise BedrockConfigError(
                f"model row {alias!r}: 'profile' must be a non-blank boto3 profile name without whitespace"
            )
    else:
        profile = None

    if auth == AUTH_IAM and _row_field(row, "apiKeyEnv", "api_key_env"):
        raise BedrockConfigError(
            f"model row {alias!r}: auth='aws-iam' rows must not carry "
            "apiKeyEnv bearer-key metadata (ambiguous credential source)"
        )
    if auth == AUTH_API_KEY:
        source = str(_row_field(row, "apiKeyEnv", "api_key_env") or BEDROCK_BEARER_ENV).strip()
        if not _ENV_NAME_RE.match(source):
            raise BedrockConfigError(f"model row {alias!r}: invalid apiKeyEnv {source!r}")

    ctx = _row_field(row, "contextWindow", "max_context")
    if not isinstance(ctx, int) or isinstance(ctx, bool) or ctx <= 0:
        raise BedrockConfigError(f"model row {alias!r}: 'contextWindow' must be a positive integer")

    return BedrockRoute(
        region=region,
        model_id=model_id,
        auth=auth,  # type: ignore[arg-type]
        profile=profile,
    )
