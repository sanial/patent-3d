# Patent Analysis Pipeline — Implementation Plan

Status: **implemented (MVP)**. This doc describes what was built so future
edits stay aligned with the design.

## Goal

Take an uploaded patent PDF and produce three artefacts:

1. **Clean schematics** — figure regions with text and lead-lines removed.
2. **Structured text** — sections + per-claim parse with dependency graph.
3. **Multimodal LLM analysis** — Gemma 4 reads the claims *and* the cleaned
   figures together, returning a structured JSON summary that drives the
   Three.js scene.

## Layers

### Layer 1 — Figure extraction (`python/ocr/figure_extractor.py`)

Vector-first. For each page:

1. `page.get_drawings()` → list of vector paths with bounding rects.
2. Reject paths in the top/bottom 60 pt (running headers / page numbers).
3. DBSCAN cluster the path centroids: `eps=40pt`, `min_samples=3`.
4. Reject clusters whose bbox area < 5 % of the page.
5. Caption snap: regex `r"\bFIG\.?\s*(\d+[A-Z]?)\b"` on text spans, attach
   the nearest caption within 80 pt below.
6. Strip text:
   - Find all text spans intersecting the figure bbox.
   - Paint each text span (dilated by 8 pt) white on the rasterised PNG.
   - On the SVG, walk and `remove()` every `<text>` element.
7. Emit `ExtractedFigure(figureId, page, bbox, captionText,
   refNumbersOriginallyInside, png_base64, svg)`.

Rasterisation: `page.get_pixmap(matrix=Matrix(2,2), clip=bbox)` → ~144 dpi.
SVG: `page.get_svg_image()` then strip `<text>` and clip the viewBox.

### Layer 2 — Structure parsing (`python/ocr/structure.py`)

Pure-Python, no fitz dep. Operates on the assembled full-text string.

- **Section splitter** — forward-scans `SECTION_HEADERS` (FIELD, BACKGROUND,
  SUMMARY, BRIEF DESCRIPTION OF THE DRAWINGS, DETAILED DESCRIPTION, CLAIMS,
  WHAT IS CLAIMED, WE CLAIM, ABSTRACT). Each match opens a section that
  closes at the next match.
- **Claim parser** — split on `r"\n\s*(\d{1,3})\.\s+"`; detect dependency
  with `r"(?:the|a|an|said)\s+\w+(?:\s+\w+)?\s+of\s+claim\s+(\d{1,3})"`
  applied to the first 200 chars of the body (avoids matching example refs
  later in the claim).
- **Ref-label extractor** — `r"\b(?P<ref>\d{2,4})\s*[–\-:.)]?\s*(?P<label>[A-Z][A-Za-z0-9 \-/]{2,60})"`.
  Keeps the shortest plausible label per ref.

### Layer 3 — Gemma 4 client (`python/llm/gemma_client.py`)

Async client targeting Ollama's OpenAI-compatible endpoint at
`http://localhost:11434/v1/chat/completions`. Model default `gemma4:4b`
(set `GEMMA_MODEL=gemma4:26b` for the MoE).

- Sends system prompt + user prompt (claims + ref definitions) and, if
  provided, base64 PNGs of the cleaned figures in the same message.
- Requests `response_format={"type": "json_object"}`.
- Pydantic-validates the response (`ClaimsAnalysis`); on `ValidationError`
  retries once with `temperature=0`.
- Hash-cache (`hashlib.sha256` over model + claims + refs + per-image
  hashes) backed by `shelve` at `python/.cache/gemma/`.

Output schema:

```json
{
  "claims": [{"number": 1, "type": "...", "summary": "...",
              "key_elements": ["..."], "ref_numbers": ["..."],
              "dependsOn": null}],
  "component_summary": {"101": {"role": "...", "appears_in_claims": [1, 3]}},
  "novelty_keywords": ["..."],
  "figure_ref_reconciliation": {"101": "FIG. 2A top-left component"}
}
```

### Layer 4 — API + frontend

- `POST /api/parse-patent` — unchanged. Response now includes
  `result.extractedFigures[]` and `result.structure { sections, claims,
  refDefinitions }`.
- `POST /api/analyze-claims` — new. Body
  `{ claims, ref_definitions, figure_pngs_b64? }` → `{ analysis }`.
  Keeps OCR fast (~5 s) and the LLM call out of the upload path.
- Frontend: `PatentUploadPanel` now has three tabs.
  - **Pages** — existing pdf.js previews.
  - **Figures** — cleaned PNGs with caption and ref-chip overlays.
  - **Claims** — parsed claims by default, with an "Analyze with Gemma 4"
    button (toggle to include figure images in the call). After analysis
    the Gemma summary replaces the raw claim body, and
    `novelty_keywords` render as chips above the list.

## File map

```
python/
  ocr/
    figure_extractor.py     # Layer 1
    structure.py            # Layer 2
    patent_ocr_pipeline.py  # wires Pass 1 + Pass 2 + Layer 1 + Layer 2
  llm/
    gemma_client.py         # Layer 3
  api/
    api.py                  # /api/parse-patent, /api/analyze-claims

src/ocr/
  PatentUploadPanel.tsx     # Pages / Figures / Claims tabs
  usePatentUpload.ts        # types include ExtractedFigure, StructuredPatent
  useClaimsAnalysis.ts      # POST /api/analyze-claims
```

## Configuration

| Env var | Default | Meaning |
|--|--|--|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Gemma server |
| `GEMMA_MODEL` | `gemma4:4b` | Ollama tag |
| `VITE_OCR_API_URL` | `http://localhost:8000` | Backend base URL (frontend) |

## Next steps (not yet implemented)

- 3D scene rebuild from `component_summary` (Layer 4 follow-up).
- Per-page lazy figure rendering for large patents (currently inline base64).
- Stream Gemma responses (SSE) for incremental UI.
