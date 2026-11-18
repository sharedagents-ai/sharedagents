"""Deterministic engine errors that repeated turns cannot repair."""

from __future__ import annotations

import re

# Deliberately narrow message patterns: quota EXHAUSTION is permanent for this
# burst, ordinary HTTP 429 rate limiting is not. JSON-RPC -32602, invalid params,
# non-absolute cwd and schema validation errors require a changed request.
# Timeouts, HTTP 5xx, connection resets and unknown errors keep normal retries.
_QUOTA = re.compile(
    r"individual quota reached|quota[_ ]exhausted|insufficient[_ ]quota|"
    r"(?:quota|usage limit) (?:has been |is )?(?:reached|exceeded|exhausted)|"
    r"(?:exceeded|exhausted) (?:your |the )?quota",
    re.I,
)
_PARAMS = re.compile(
    r"(?<!\d)-32602(?!\d)|invalid[_ -]param(?:s|eters?)\b|"
    r"cwd must be an absolute path|schema[_ -]validation|"
    r"validation (?:error|failed|failure).*schema|"
    r"schema.*validation (?:error|failed|failure)",
    re.I | re.S,
)


def permanent_error_classification(classification: str, message: str = "") -> str:
    """Return a canonical permanent class, or empty for unchanged retry policy."""
    text = f"{classification}\n{message}"
    if classification == "ontology_publication":
        # The native research loop already gave two bounded repair rounds.
        # A full turn retry cannot repair a missing data-layer installation.
        return classification
    if _QUOTA.search(text):
        return "quota_exhausted"
    if _PARAMS.search(text):
        return "invalid_params"
    return ""
