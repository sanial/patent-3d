"""Layer 3 — Gemma 4 claims analysis.

Two backends, selected automatically:

* **Google AI Studio (Gemma 4 API)** — active when ``GEMMA_API_KEY`` or
  ``GEMINI_API_KEY`` is set. Calls
  ``https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``.
* **Ollama** — fallback for local dev (``ollama pull gemma4:4b``).

Gemma 4 is natively multimodal — figure images are passed in the same call
as the claims text, so the model can reconcile visible ref numbers with the
textual definitions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shelve
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

try:
    from dotenv import load_dotenv

    # Load python/.env so GEMMA_API_KEY is available without manually exporting.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover — optional dependency
    pass

logger = logging.getLogger("gemma_client")


# ── Config ──────────────────────────────────────────────────────────────────


def _clean(val: str | None) -> str | None:
    if val is None:
        return None
    val = val.strip().strip('"').strip("'")
    return val or None


API_KEY = _clean(os.environ.get("GEMMA_API_KEY")) or _clean(os.environ.get("GEMINI_API_KEY"))
DEFAULT_API_BASE = os.environ.get(
    "GEMMA_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
)
DEFAULT_API_MODEL = os.environ.get("GEMMA_API_MODEL", "gemma-4-27b-it")

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:4b")

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "gemma"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "shelve"

REQUEST_TIMEOUT = 120.0


# ── Output schema ───────────────────────────────────────────────────────────

class ClaimAnalysis(BaseModel):
    number: int
    type: str
    summary: str
    key_elements: list[str] = Field(default_factory=list)
    ref_numbers: list[str] = Field(default_factory=list)
    dependsOn: int | None = None


class ComponentInfo(BaseModel):
    role: str
    appears_in_claims: list[int] = Field(default_factory=list)


class ClaimsAnalysis(BaseModel):
    claims: list[ClaimAnalysis] = Field(default_factory=list)
    component_summary: dict[str, ComponentInfo] = Field(default_factory=dict)
    novelty_keywords: list[str] = Field(default_factory=list)
    figure_ref_reconciliation: dict[str, str] = Field(default_factory=dict)


# ── Prompt ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a patent analyst. You will receive the claims of a
patent, a map of reference-number definitions, and (optionally) images of the
patent's figures.

Your task:
1. For each claim, produce: number, type ("independent" or "dependent"),
   a 1-sentence summary, key technical elements, the reference numbers it
   mentions, and dependsOn (the claim number it depends on, or null).
2. Build component_summary: for each reference number, give its role and the
   list of claim numbers it appears in.
3. Extract 5-10 novelty_keywords that capture what is new in this patent.
4. If figure images are provided, populate figure_ref_reconciliation:
   {ref_number -> a short string describing where it appears in a figure}.

Return STRICT JSON only — no commentary, no markdown fences. Use this exact
schema:

{
  "claims": [
    {"number": 1, "type": "independent", "summary": "...",
     "key_elements": ["..."], "ref_numbers": ["..."], "dependsOn": null}
  ],
  "component_summary": {
    "101": {"role": "...", "appears_in_claims": [1, 3]}
  },
  "novelty_keywords": ["..."],
  "figure_ref_reconciliation": {"101": "FIG. 2A top-left component"}
}
"""


# ── Client ──────────────────────────────────────────────────────────────────

@dataclass
class GemmaClient:
    api_key: str | None = API_KEY
    api_base: str = DEFAULT_API_BASE
    api_model: str = DEFAULT_API_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    timeout: float = REQUEST_TIMEOUT

    @property
    def provider(self) -> str:
        return "google" if self.api_key else "ollama"

    @property
    def model(self) -> str:
        return self.api_model if self.api_key else self.ollama_model

    async def analyze_claims(
        self,
        claims: list[dict[str, Any]],
        ref_definitions: dict[str, str],
        figure_pngs_b64: list[str] | None = None,
    ) -> ClaimsAnalysis:
        cache_key = _cache_key(
            f"{self.provider}:{self.model}", claims, ref_definitions, figure_pngs_b64 or []
        )
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.info("Gemma cache hit (%s)", cache_key[:10])
            return cached

        user_text = _build_user_text(claims, ref_definitions)
        figures = figure_pngs_b64 or []

        try:
            result = await self._call(user_text, figures)
        except ValidationError as e:
            logger.warning("Gemma JSON validation failed, retrying once: %s", e)
            result = await self._call(user_text, figures, retry=True)

        _cache_put(cache_key, result)
        return result

    async def _call(
        self,
        user_text: str,
        figures: list[str],
        retry: bool = False,
    ) -> ClaimsAnalysis:
        if self.api_key:
            return await self._call_google(user_text, figures, retry=retry)
        return await self._call_ollama(user_text, figures, retry=retry)

    async def _call_google(
        self,
        user_text: str,
        figures: list[str],
        retry: bool = False,
    ) -> ClaimsAnalysis:
        url = (
            f"{self.api_base.rstrip('/')}/models/{self.api_model}:generateContent"
            f"?key={self.api_key}"
        )
        parts: list[dict[str, Any]] = [{"text": user_text}]
        for b64 in figures:
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.0 if retry else 0.1,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.error("Gemma API error %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
            body = resp.json()

        try:
            content = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemma API response: {body}") from e
        return ClaimsAnalysis.model_validate_json(content)

    async def _call_ollama(
        self,
        user_text: str,
        figures: list[str],
        retry: bool = False,
    ) -> ClaimsAnalysis:
        url = f"{self.ollama_base_url}/v1/chat/completions"
        messages = _build_messages(user_text, figures)
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0 if retry else 0.1,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()

        content = body["choices"][0]["message"]["content"]
        return ClaimsAnalysis.model_validate_json(content)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_user_text(
    claims: list[dict[str, Any]],
    ref_definitions: dict[str, str],
) -> str:
    parts: list[str] = []
    parts.append("REFERENCE-NUMBER DEFINITIONS:")
    if ref_definitions:
        for ref, label in sorted(ref_definitions.items()):
            parts.append(f"  {ref}: {label}")
    else:
        parts.append("  (none provided)")

    parts.append("")
    parts.append("CLAIMS:")
    for c in claims:
        num = c.get("number")
        body = c.get("body", "")
        parts.append(f"Claim {num}: {body}")
    return "\n".join(parts)


def _build_messages(user_text: str, figure_pngs_b64: list[str]) -> list[dict[str, Any]]:
    if not figure_pngs_b64:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    # Multimodal: text + images in one user turn.
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64 in figure_pngs_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _cache_key(
    model: str,
    claims: list[dict[str, Any]],
    ref_definitions: dict[str, str],
    figure_pngs_b64: list[str],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model.encode())
    hasher.update(json.dumps(claims, sort_keys=True).encode())
    hasher.update(json.dumps(ref_definitions, sort_keys=True).encode())
    for b64 in figure_pngs_b64:
        hasher.update(hashlib.sha256(b64.encode()).digest())
    return hasher.hexdigest()


def _cache_get(key: str) -> ClaimsAnalysis | None:
    try:
        with shelve.open(str(CACHE_FILE)) as db:
            raw = db.get(key)
            if raw is None:
                return None
            return ClaimsAnalysis.model_validate(raw)
    except Exception:
        return None


def _cache_put(key: str, value: ClaimsAnalysis) -> None:
    try:
        with shelve.open(str(CACHE_FILE)) as db:
            db[key] = value.model_dump()
    except Exception as e:
        logger.warning("Gemma cache write failed: %s", e)


def run_sync(coro):
    """Helper for FastAPI handlers that prefer sync."""
    return asyncio.get_event_loop().run_until_complete(coro)


__all__ = ["GemmaClient", "ClaimsAnalysis", "ClaimAnalysis", "ComponentInfo"]
