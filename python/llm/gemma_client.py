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
DEFAULT_API_MODEL = os.environ.get("GEMMA_API_MODEL", "gemma-4-31b-it")
DEFAULT_VISION_MODEL = os.environ.get("GEMMA_VISION_MODEL", "gemini-2.5-flash")

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
    vision_model: str = DEFAULT_VISION_MODEL
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

    async def extract_claims_from_text(
        self,
        full_text: str,
        max_chars: int = 60_000,
    ) -> list[dict[str, Any]]:
        """Use Gemma to extract numbered claims from messy OCR text.

        Returns dicts shaped like ``ParsedClaim.to_dict()``:
        ``{"number", "type", "body", "refs", "dependsOn"}``. Returns ``[]``
        on failure rather than raising — the caller can fall back to whatever
        the regex parser produced.
        """
        if not full_text or not full_text.strip():
            return []

        # Patents put claims at the end. Bias the window toward the tail so
        # we don't waste tokens on the spec.
        text = full_text[-max_chars:] if len(full_text) > max_chars else full_text

        prompt = (
            "You are given the OCR text of a US patent. The text may contain "
            "OCR errors. Find the CLAIMS section (it usually starts with "
            "'What is claimed is:', 'We claim:', or 'Claims:') and extract "
            "every numbered claim.\n\n"
            "For each claim produce:\n"
            "  number      — the claim number (1, 2, 3, ...)\n"
            "  type        — 'independent' or 'dependent'\n"
            "  body        — the full claim text, with OCR errors cleaned up\n"
            "  refs        — list of reference numerals mentioned (strings like '101')\n"
            "  dependsOn   — claim number it depends on, or null\n\n"
            "Return EXACTLY one JSON object, no prose, no markdown fences:\n"
            '{"claims":[{"number":1,"type":"independent","body":"...",'
            '"refs":["101","102"],"dependsOn":null}, ...]}\n\n'
            "If no claims section is found, return {\"claims\":[]}.\n\n"
            "--- PATENT TEXT ---\n"
            f"{text}"
        )

        try:
            if self.api_key:
                # Prefer the fast Gemini vision model for this step too — it
                # supports responseMimeType=json which deterministically
                # suppresses the chain-of-thought that open Gemma emits.
                model_for_extract = (
                    self.vision_model
                    if self.vision_model.startswith("gemini")
                    else self.api_model
                )
                url = (
                    f"{self.api_base.rstrip('/')}/models/{model_for_extract}:generateContent"
                    f"?key={self.api_key}"
                )
                gen_cfg: dict[str, Any] = {
                    "temperature": 0.0,
                    "maxOutputTokens": 8192,
                }
                if model_for_extract.startswith("gemini"):
                    gen_cfg["responseMimeType"] = "application/json"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": gen_cfg,
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code >= 400:
                        logger.error(
                            "Gemma extract_claims error %s: %s",
                            resp.status_code,
                            resp.text,
                        )
                    resp.raise_for_status()
                    body = resp.json()
                content = body["candidates"][0]["content"]["parts"][0]["text"]
            else:
                url = f"{self.ollama_base_url}/v1/chat/completions"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                content = body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemma claim extraction failed: %s", exc)
            return []

        try:
            json_text = _extract_json(content)
            obj = json.loads(json_text)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Gemma claim extraction returned non-JSON: %s", exc)
            return []

        raw = obj.get("claims") if isinstance(obj, dict) else None
        if not isinstance(raw, list):
            return []

        out: list[dict[str, Any]] = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            try:
                number = int(c.get("number"))
            except (TypeError, ValueError):
                continue
            body_text = str(c.get("body") or "").strip()
            if not body_text:
                continue
            depends_on_raw = c.get("dependsOn")
            try:
                depends_on = int(depends_on_raw) if depends_on_raw is not None else None
            except (TypeError, ValueError):
                depends_on = None
            ctype = str(c.get("type") or ("dependent" if depends_on else "independent"))
            refs_raw = c.get("refs") or []
            refs = sorted({str(r) for r in refs_raw if r is not None})
            out.append(
                {
                    "number": number,
                    "type": ctype,
                    "body": body_text,
                    "refs": refs,
                    "dependsOn": depends_on,
                }
            )
        out.sort(key=lambda d: d["number"])
        return out

    async def extract_claims_from_pages(
        self,
        page_pngs_b64: list[str],
    ) -> list[dict[str, Any]]:
        """Extract claims from page images via a two-step pipeline.

        Step 1: transcribe each page in parallel (Gemma 4 vision, one call
                per page, small outputs — reliable JSON).
        Step 2: feed the concatenated transcript to ``extract_claims_from_text``.

        Avoids the failure mode where a single all-pages call blows the
        output-token budget on chain-of-thought before emitting JSON.
        """
        if not page_pngs_b64:
            return []

        transcripts = await self._transcribe_pages(page_pngs_b64, max_concurrency=2)
        for i, t in enumerate(transcripts):
            logger.info("Gemma transcribe page %d: %d chars", i, len(t))
        full_text = "\n\n".join(t for t in transcripts if t)
        # Dump for offline inspection.
        try:
            (CACHE_DIR / "last_transcript.txt").write_text(full_text, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        if not full_text.strip():
            logger.warning(
                "Gemma claim-from-pages: transcripts empty for %d pages",
                len(page_pngs_b64),
            )
            return []
        logger.info(
            "Gemma claim-from-pages: transcribed %d pages → %d chars",
            len(page_pngs_b64), len(full_text),
        )
        return await self.extract_claims_from_text(full_text)

    async def _transcribe_pages(
        self,
        page_pngs_b64: list[str],
        max_concurrency: int = 2,
    ) -> list[str]:
        sem = asyncio.Semaphore(max_concurrency)

        async def one(b64: str) -> str:
            async with sem:
                return await self._transcribe_one_page(b64)

        return await asyncio.gather(*[one(b) for b in page_pngs_b64])

    async def _transcribe_one_page(self, png_b64: str) -> str:
        prompt = (
            "Transcribe ALL readable text on this US patent page, in reading "
            "order, preserving line breaks and claim numbering. Include every "
            "word — headings, claims, footnotes — but NO commentary, NO "
            "analysis, NO markdown, NO code fences.\n\n"
            "Output format: emit the line `<<<BEGIN>>>` on its own line, then "
            "the verbatim transcribed text, then the line `<<<END>>>` on its "
            "own line. Nothing else."
        )
        timeout = max(self.timeout, 240.0)
        parts: list[dict[str, Any]] = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": png_b64}},
        ]

        try:
            if self.api_key:
                url = (
                    f"{self.api_base.rstrip('/')}/models/{self.vision_model}:generateContent"
                    f"?key={self.api_key}"
                )
                payload: dict[str, Any] = {
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
                }
                # Gemini models (not open Gemma) accept responseMimeType.
                if self.vision_model.startswith("gemini"):
                    payload["generationConfig"]["responseMimeType"] = "text/plain"
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                content = body["candidates"][0]["content"]["parts"][0]["text"]
            else:
                url = f"{self.ollama_base_url}/v1/chat/completions"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                        ],
                    }],
                    "temperature": 0.0,
                }
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    body = resp.json()
                content = body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gemma transcribe page failed: %s: %r",
                type(exc).__name__, exc,
            )
            return ""
        text = str(content or "")
        # Strip the model's reasoning preamble using the delimiters.
        begin = text.find("<<<BEGIN>>>")
        if begin != -1:
            text = text[begin + len("<<<BEGIN>>>"):]
            end = text.rfind("<<<END>>>")
            if end != -1:
                text = text[:end]
        return text.strip()

    async def classify_figure_pages(
        self,
        page_pngs_b64: list[str],
        chunk_size: int = 1,
        max_concurrency: int = 20,
    ) -> list[dict[str, Any]]:
        """Ask Gemma which pages are patent-figure pages and what their IDs are.

        Returns a list of ``{"page": int, "figure_id": str, "caption": str}``
        for pages classified as figures. ``page`` is the 0-based index into
        ``page_pngs_b64``.

        Per-page (chunk_size=1) calls are most reliable: each request has one
        image and a small expected output, so Gemma can't ramble before
        emitting JSON.
        """
        if not page_pngs_b64:
            return []

        sem = asyncio.Semaphore(max_concurrency)

        async def one(start: int, chunk: list[str]) -> list[dict[str, Any]]:
            async with sem:
                return await self._classify_pages_call(chunk, offset=start)

        tasks = []
        for start in range(0, len(page_pngs_b64), chunk_size):
            chunk = page_pngs_b64[start : start + chunk_size]
            tasks.append(one(start, chunk))

        out: list[dict[str, Any]] = []
        for results in await asyncio.gather(*tasks, return_exceptions=False):
            out.extend(results)
        out.sort(key=lambda d: d["page"])
        return out

    async def _classify_pages_call(
        self,
        chunk_pngs_b64: list[str],
        offset: int,
    ) -> list[dict[str, Any]]:
        if len(chunk_pngs_b64) == 1:
            prompt = (
                "Look at this single page of a US patent PDF.\n\n"
                "Decide whether it is a FIGURE page (its primary content is a "
                "patent drawing, usually labeled 'FIG. N' at the bottom) or a "
                "TEXT page (cover, claims, specification, etc.).\n\n"
                "Reply with EXACTLY one JSON object, no prose, no markdown:\n"
                '{"is_figure": <true|false>, "figure_id": "FIG_<id or null>", '
                '"caption": "<exact caption text or empty>"}\n\n'
                "Example for a figure page: "
                '{"is_figure":true,"figure_id":"FIG_3A","caption":"FIG. 3A"}\n'
                "Example for a text page: "
                '{"is_figure":false,"figure_id":null,"caption":""}'
            )
        else:
            prompt = (
                "TASK: Identify which of the attached patent PDF pages are FIGURE "
                "pages (pages whose primary content is one or more patent drawings, "
                "usually labeled 'FIG. N' at the bottom).\n\n"
                f"There are {len(chunk_pngs_b64)} page images attached in order, "
                f"indexed 0..{len(chunk_pngs_b64) - 1}.\n\n"
                "OUTPUT FORMAT — return EXACTLY one JSON object and NOTHING ELSE. "
                "Start with '{' and end with '}'. No prose, no fences.\n\n"
                "Schema:\n"
                '{"figures":[{"index":<int>,"figure_id":"FIG_<id>","caption":"<text>"}]}\n'
            )
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for b64 in chunk_pngs_b64:
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

        if self.api_key:
            url = (
                f"{self.api_base.rstrip('/')}/models/{self.api_model}:generateContent"
                f"?key={self.api_key}"
            )
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    logger.error("Gemma classify error %s: %s", resp.status_code, resp.text)
                resp.raise_for_status()
                body = resp.json()
            try:
                content = body["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                raise RuntimeError(f"Unexpected Gemma response: {body}") from e
        else:
            # Ollama multimodal.
            url = f"{self.ollama_base_url}/v1/chat/completions"
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b64 in chunk_pngs_b64:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            payload = {
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": content_parts}],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                body = resp.json()
            content = body["choices"][0]["message"]["content"]

        try:
            data = json.loads(_extract_json(content))
        except Exception as e:
            # Common case: Gemma rambled past the token budget on a text page.
            # If the prose clearly says "not a figure", treat as no-figure
            # silently. Otherwise log a debug message.
            low = content.lower()
            if any(s in low for s in (
                "is_figure: false",
                '"is_figure":false',
                "is_figure`: false",
                "text page",
                "not a figure",
                "no drawings",
                'no "fig.',
            )):
                return []
            logger.warning("Gemma classify JSON parse failed: %s — raw: %s", e, content[:300])
            return []

        # Per-page response shape: {"is_figure": bool, "figure_id": ..., "caption": ...}
        if "is_figure" in data:
            if not data.get("is_figure"):
                return []
            fid = data.get("figure_id")
            if not fid or fid == "null":
                fid = f"P{offset + 1}"
            return [
                {
                    "page": offset,
                    "figure_id": str(fid),
                    "caption": str(data.get("caption") or ""),
                }
            ]

        # Multi-page response shape: {"figures": [{"index":..., ...}]}
        out: list[dict[str, Any]] = []
        for entry in data.get("figures", []) or []:
            try:
                idx = int(entry["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(chunk_pngs_b64):
                continue
            out.append(
                {
                    "page": offset + idx,
                    "figure_id": str(entry.get("figure_id") or f"P{offset + idx + 1}"),
                    "caption": str(entry.get("caption") or ""),
                }
            )
        return out

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
        # Open Gemma models on Google AI Studio do NOT support
        # systemInstruction or responseMimeType — inline the system prompt
        # into the user turn and parse JSON out of free-form text.
        combined = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"--- INPUT ---\n{user_text}\n\n"
            "Respond with ONLY the JSON object. No prose, no markdown fences."
        )
        parts: list[dict[str, Any]] = [{"text": combined}]
        for b64 in figures:
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0 if retry else 0.1,
                "maxOutputTokens": 4096,
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
        json_text = _extract_json(content)
        return ClaimsAnalysis.model_validate_json(json_text)

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

def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of free-form model output.

    Handles ```json fenced blocks and prose surrounding the object.
    """
    s = text.strip()
    # Strip ``` fences.
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # Already a JSON object?
    if s.startswith("{"):
        return s
    # Find the first '{' and walk to its matching '}'.
    start = s.find("{")
    if start == -1:
        return s  # let pydantic raise
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


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
