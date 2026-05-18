"""Layer 2 — section + claim parsing.

Pure-Python, no PyMuPDF dependency. Operates on the assembled full-text string
produced by Pass 1 of the OCR pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any


SECTION_HEADERS = [
    "FIELD OF THE INVENTION",
    "BACKGROUND",
    "SUMMARY",
    "BRIEF DESCRIPTION OF THE DRAWINGS",
    "DETAILED DESCRIPTION",
    "CLAIMS",
    "WHAT IS CLAIMED",
    "WE CLAIM",
    "ABSTRACT",
]

_SECTION_RE = re.compile(
    r"(?P<header>" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*[:\.\n]",
    re.IGNORECASE,
)

_CLAIM_SPLIT = re.compile(r"\n\s*(\d{1,3})\.\s+")
_DEPENDENCY = re.compile(
    r"(?:the|a|an|said)\s+\w+(?:\s+\w+)?\s+of\s+claim\s+(\d{1,3})",
    re.IGNORECASE,
)
_REF_INLINE = re.compile(r"\b(\d{2,4})\b")
_REF_LABELLED = re.compile(
    r"\b(?P<ref>\d{2,4})\s*[–\-:.)]?\s*(?P<label>[A-Z][A-Za-z0-9 \-/]{2,60})"
)


@dataclass
class ParsedClaim:
    number: int
    type: str  # "independent" | "dependent"
    body: str
    refs: list[str] = field(default_factory=list)
    dependsOn: int | None = None  # noqa: N815 — JSON key

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredPatent:
    sections: dict[str, str] = field(default_factory=dict)
    claims: list[ParsedClaim] = field(default_factory=list)
    ref_definitions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": self.sections,
            "claims": [c.to_dict() for c in self.claims],
            "refDefinitions": self.ref_definitions,
        }


# ── Public entry point ──────────────────────────────────────────────────────

def parse_structure(full_text: str) -> StructuredPatent:
    sections = _split_sections(full_text)
    claims_body = (
        sections.get("claims")
        or sections.get("what is claimed")
        or sections.get("we claim")
        or ""
    )
    claims = _parse_claims(claims_body)
    ref_definitions = _extract_ref_definitions(full_text)
    return StructuredPatent(sections=sections, claims=claims, ref_definitions=ref_definitions)


# ── Section splitter ────────────────────────────────────────────────────────

def _split_sections(full_text: str) -> dict[str, str]:
    """Forward scan: each header match opens a section, next header closes it."""
    matches = list(_SECTION_RE.finditer(full_text))
    if not matches:
        return {}

    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = _normalize_header(m.group("header"))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip(" \n\r\t:.")
        if body:
            out[key] = body
    return out


def _normalize_header(h: str) -> str:
    low = h.lower().strip()
    if "claim" in low:
        return "claims"
    if "drawing" in low:
        return "briefDescriptionOfDrawings"
    if "detailed" in low:
        return "detailedDescription"
    if "field" in low:
        return "fieldOfInvention"
    if "background" in low:
        return "background"
    if "summary" in low:
        return "summary"
    if "abstract" in low:
        return "abstract"
    return low.replace(" ", "")


# ── Claim parser ────────────────────────────────────────────────────────────

def _parse_claims(claims_body: str) -> list[ParsedClaim]:
    if not claims_body.strip():
        return []

    parts = _CLAIM_SPLIT.split(claims_body)
    # parts = [preamble, "1", body1, "2", body2, ...]
    out: list[ParsedClaim] = []
    for i in range(1, len(parts) - 1, 2):
        try:
            number = int(parts[i])
        except ValueError:
            continue
        body = parts[i + 1].strip()
        if not body:
            continue

        depends_on = _detect_dependency(body)
        refs = sorted(set(_REF_INLINE.findall(body)))

        out.append(
            ParsedClaim(
                number=number,
                type="dependent" if depends_on else "independent",
                body=body,
                refs=refs,
                dependsOn=depends_on,
            )
        )
    return out


def _detect_dependency(body: str) -> int | None:
    # Look only in the opening clause — references to other claims in the body
    # often appear as examples rather than dependencies.
    head = body[:200]
    m = _DEPENDENCY.search(head)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ── Ref-number labels ───────────────────────────────────────────────────────

def _extract_ref_definitions(full_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _REF_LABELLED.finditer(full_text):
        ref = m.group("ref")
        label = m.group("label").strip().rstrip(".,;:")
        if len(label) < 3:
            continue
        # Keep the shortest plausible label.
        if ref not in out or len(label) < len(out[ref]):
            out[ref] = label
    return out


__all__ = ["StructuredPatent", "ParsedClaim", "parse_structure"]
