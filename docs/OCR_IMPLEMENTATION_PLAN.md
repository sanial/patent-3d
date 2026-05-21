# Patent OCR — Implementation Plan

Four-layer pipeline. Each layer ships independently and feeds the next.

```
PDF
 ├─► Layer 1: figure_extractor.py  ─► figures[] (clean SVG + PNG + sidecar JSON)
 ├─► Layer 2: structure.py          ─► sections{}, claims[], refDefinitions{}
 └─► Layer 3: gemma_client.py       ─► claim analysis (multimodal: text + figures)
        └─► Layer 4: frontend       ─► Claims tab, Figures tab, 3D rebuild
```

## Layer 1 — Figure extraction (`python/ocr/figure_extractor.py`)

Vector-first, raster fallback. Patent PDFs are mostly vector; we keep the
drawing channel and discard the text channel inside figure regions.

**Steps per page:**

1. Collect all `page.get_drawings()` paths. Compute centroid + bbox per path.
2. Cluster path centroids with **DBSCAN** (`eps≈40 pt`, `min_samples=3`).
   Each cluster is a candidate figure. Reject clusters that:
   - intersect the header/footer margins (top/bottom 60 pt)
   - have a bbox area < 5 % of the page
3. Detect captions: regex `r"FIG\.\s*\d+[A-Z]?"` over `get_text("dict")`
   blocks. For each cluster, snap to the nearest caption ≤ 80 pt below.
4. Strip text from inside the figure bbox:
   - Drop any `TextBlock` whose bbox intersects the figure bbox.
   - Filter `get_drawings()` paths: keep only those fully inside the figure
     bbox and not within 8 pt of any text-span bbox (removes lead lines).
5. Re-render clean output:
   - **SVG**: `page.get_svg_image(matrix=…, clip=figureBbox)`, then strip
     `<text>` elements with an XML pass.
   - **PNG**: `page.get_pixmap(clip=figureBbox, matrix=2x)` after masking text
     bboxes to white.
6. Emit sidecar JSON per figure:
   ```json
   {
     "figureId": "FIG_2A",
     "page": 1,
     "bbox": [x0, y0, x1, y1],
     "captionText": "FIG. 2A — Sensor Assembly",
     "refNumbersOriginallyInside": ["101", "130"]
   }
   ```

PNGs are returned to the frontend as **base64-encoded data URLs** to avoid a
file-storage layer for MVP. Move to a real static-files mount if the responses
get heavy.

## Layer 2 — Structure parsing (`python/ocr/structure.py`)

Pure-Python, no PyMuPDF dependency. Operates on the assembled full-text string.

**Section splitter:**

```python
SECTION_HEADERS = [
  "FIELD OF THE INVENTION", "BACKGROUND",
  "SUMMARY", "BRIEF DESCRIPTION OF THE DRAWINGS",
  "DETAILED DESCRIPTION", "CLAIMS",
]
```

Splits full text into a dict of section name → body using a single forward
scan; tolerant of minor heading variations.

**Per-claim parser:**

- Numbered claim splitter (already in pipeline, refactor here).
- Dependency detection: regex
  `r"(?:the|a)\s+\w+\s+of\s+claim\s+(\d+)"` against the claim body — first
  match becomes `dependsOn`.
- Inline ref-number extraction per claim (not just globally).

**Output shape:**

```json
{
  "sections": { "summary": "...", "claims": "...", ... },
  "claims": [
    { "number": 1, "type": "independent", "dependsOn": null,
      "body": "...", "refs": ["101", "130"] }
  ],
  "refDefinitions": { "101": "Sensor Module" }
}
```

## Layer 3 — Gemma 4 client (`python/llm/gemma_client.py`)

Local: **Ollama** with `gemma4:4b` (E4B) or `gemma4:26b` (MoE). Cloud:
Vertex AI Model Garden, same prompt.

```bash
ollama pull gemma4:4b
```

**Client design:**

- Single async `analyze_claims(claims, ref_definitions, figure_images=None)`.
- Talks to `http://localhost:11434/v1/chat/completions` via `httpx`.
- Request body uses `response_format={"type": "json_object"}`.
- Multimodal: pass figure PNGs (already base64 from Layer 1) as
  `image_url` content parts in the same turn — Gemma 4 is natively multimodal,
  so reconciliation lives in one call.
- Response validated with Pydantic; one retry on `ValidationError`.
- Cache key = `sha256(claims_text + model_name + sorted(figure_hashes))` →
  `shelve` file under `python/.cache/gemma/` for local dev.

**Target JSON output:**

```json
{
  "claims": [
    { "number": 1, "type": "independent",
      "summary": "...",
      "key_elements": ["sensor module"],
      "ref_numbers": ["101", "130"],
      "dependsOn": null }
  ],
  "component_summary": {
    "101": { "role": "...", "appears_in_claims": [1, 3] }
  },
  "novelty_keywords": ["..."],
  "figure_ref_reconciliation": {
    "101": "FIG. 2A top-left component"
  }
}
```

The `figure_ref_reconciliation` field is the multimodal win — Gemma 4 looks
at the cleaned figure images and aligns visible ref numbers with the textual
definitions, populating richer `userData.patentData` for the Three.js scene.

**Endpoint:** `POST /api/analyze-claims` — takes the Layer 2 output (and
optional figure list), returns the JSON above. Kept separate from
`/api/parse-patent` so OCR stays fast (~5 s) while analysis is the async
follow-up (~15-30 s).

## Layer 4 — Frontend

Extend the existing `PatentUploadPanel` with two tabs after parsing
completes:

- **Claims** — list claims with Gemma summaries; chips render the ref numbers
  and call `handleSelect2(ref)` to drive 3D highlighting. Dependent claims
  visually indented under their parent (`dependsOn`).
- **Figures** — strip of cleaned PNG thumbnails; click jumps the camera via
  `scene.focusComponent(ref)` on the most prominent ref in that figure's
  sidecar JSON.

**3D bridge:** new `rebuildFromData(componentSummary)` method on
`useThreeScene` consumes `analysis.component_summary`. Each key becomes a
group in `componentMeshesRef`; `role` + `appears_in_claims` populate
`userData.patentData` so clicking a 3D object surfaces real patent context in
the `ClaimsPanel`.

## Execution order

1. **Figure extraction** — Layer 1. Needed before multimodal Gemma is useful.
2. **Structure parsing** — Layer 2. Fast, unblocks the Gemma prompt design.
3. **Gemma 4 client** — Layer 3. Text-only first turn, multimodal as soon as
   Layer 1's PNGs exist.
4. **`/api/analyze-claims` endpoint** — wire Gemma output to FastAPI.
5. **Frontend tabs** — Claims + Figures, driving the existing Three.js
   selection system.
6. **Polish** — lead-line stripping refinement; better caption matching; eval
   harness with a fixed PDF set.

## Dependencies added

```
scikit-learn>=1.5     # DBSCAN
httpx>=0.27           # Gemma client
```

(Pydantic is already pulled in by FastAPI.)

---

## Layer 5 — Fast claim extraction via Gemma 4 vision (no OCR)

Added May 2026. Bypasses the full `PatentOCRPipeline` (Tesseract / EasyOCR
+ regex parser) for the claims path on scanned patents where the PDF text
layer is empty.

### Motivation

For `12596070.pdf` (21‑page scanned plant‑monitoring patent) the regex parser
produced **zero claims** because:

- PyMuPDF `page.get_text()` returns `""` on every page (scanned PDF, no
  embedded text layer).
- Tesseract OCR on 21 pages takes >2 min and still emits noisy output the
  regex parser fails to segment.
- Gemma 4 (`gemma-4-31b-it`, open) emits chain‑of‑thought before any JSON
  payload and exhausts the token budget.

The fast path renders only the **last N pages** (where the claims live in a
US patent) directly to PNGs and feeds them to a fast vision model.

### Pipeline

Two independent Gemini API calls per upload:

```
PDF bytes
   │
   ├─ render last 4 pages → PNG @ 1.5x via PyMuPDF
   │      │
   │      ▼
   │  ┌─────────────────────────────────┐
   │  │ Step 1: per‑page transcribe     │   gemini‑2.5‑flash, vision
   │  │  - asyncio.gather, semaphore=2  │   timeout=240s, max_tokens=8192
   │  │  - prompt with <<<BEGIN>>> /    │
   │  │    <<<END>>> delimiters         │
   │  │  - strip preamble after parsing │
   │  └────────────────┬────────────────┘
   │                   │  list[str]
   │                   ▼
   │           "\n\n".join(transcripts)        ~22 KB plain text
   │                   │
   │                   ▼
   │  ┌─────────────────────────────────┐
   │  │ Step 2: text → claim JSON       │   gemini‑2.5‑flash, text
   │  │  responseMimeType=application/  │   max_tokens=8192
   │  │  json (suppresses CoT)          │
   │  └────────────────┬────────────────┘
   │                   │  list[ParsedClaim.to_dict()]
   ▼                   ▼
 frontend ◄── /api/extract-claims?last_pages=4
```

### Backend

#### `python/llm/gemma_client.py`

- New env `GEMMA_VISION_MODEL` (default `gemini-2.5-flash`). Used for OCR /
  JSON‑extraction steps independently of `GEMMA_API_MODEL` (which still
  drives `analyze_claims` and the figure classifier).
- New `vision_model` field on `GemmaClient`.
- `extract_claims_from_pages(page_pngs_b64) → list[dict]`: orchestrator.
  Calls `_transcribe_pages` then `extract_claims_from_text`. Logs per‑page
  char counts and dumps the concatenated transcript to
  `python/.cache/gemma/last_transcript.txt` for debugging.
- `_transcribe_pages(pngs, max_concurrency=2)`: `asyncio.gather` with
  `Semaphore(2)`. Concurrency is intentionally low — Gemini handles
  visual OCR fast (~7 s per page) but parallelism above 2 was hitting
  socket‑level read timeouts on this network.
- `_transcribe_one_page(png_b64) → str`: single Gemini call. Prompt asks
  for verbatim text bracketed by literal `<<<BEGIN>>>` / `<<<END>>>`
  markers; the helper strips everything outside those markers, defeating
  any preamble the model might add. Timeout `max(self.timeout, 240.0)`.
- `extract_claims_from_text` updated to route through `vision_model` when
  it is a Gemini variant, and to add `responseMimeType:
  "application/json"` to `generationConfig`. This deterministically
  suppresses chain‑of‑thought; the response is guaranteed parseable JSON.
  When `vision_model` is non‑Gemini (e.g. open Gemma) the original code
  path with `_extract_json` heuristics is preserved.

#### `python/api/api.py`

- New `POST /api/extract-claims?last_pages=4`. Validates the upload, reads
  bytes once, then:
  ```python
  pngs = await asyncio.to_thread(_render_last_pages_b64, pdf_bytes,
                                 last_pages, scale=1.5)
  claims = await _gemma.extract_claims_from_pages(pngs)
  return {"filename": file.filename, "claims": claims}
  ```
- New helper `_render_last_pages_b64(pdf_bytes, n, scale=1.5) → list[str]`:
  PyMuPDF, returns base64‑encoded PNGs for the last `n` pages.
- The endpoint is independent of `/api/parse-patent` and
  `/api/extract-figures`; all three execute concurrently in the FastAPI
  event loop because each pushes its blocking work into
  `asyncio.to_thread`.

### Frontend (`src/ocr/usePatentUpload.ts`)

Three `fetch`es start in parallel on file drop:

```
/api/extract-figures?use_gemma=true    ── figures tab populates ~25–50 s
/api/parse-patent                       ── full OCR result (slow)
/api/extract-claims?last_pages=4        ── claims tab populates ~55 s
```

Each promise updates the React result independently:

- `figuresPromise.then(...)` overrides `result.figures`.
- `claimsPromise.then(...)` writes `result.structure.claims` and
  `result.claims = claim.body[]`.
- `parsePromise.then(...)` fills the rest. After it resolves we keep the
  fast‑path claims if `parsedClaimCount === 0` (the common scanned‑PDF
  case).

UX: figures tab and claims tab each show their content the moment their
respective request returns. Wall time observed end‑to‑end on
`12596070.pdf`: ~55 s, dominated by the per‑page transcription step.

### Why two Gemini calls instead of one combined image+JSON call

Empirically the open Gemma 4 model — even with a strict JSON prompt — emits
1.5–3 KB of "the user wants me to…" reasoning before any structured output,
and on long inputs (full claims section in 4 page images) the 8192‑token
budget is exhausted before the JSON object closes. Splitting the work means:

1. The vision pass produces only **plain text** — small, naturally
   short‑circuited, immune to prose vs. JSON formatting issues.
2. The extraction pass takes plain text and emits JSON under
   `responseMimeType: application/json`, which the API enforces — no CoT
   possible.

### Diagnostic / debug surface

- `python/.cache/gemma/last_transcript.txt` — last full concatenated
  transcript (overwritten each call).
- `python/test_claim_pages.py` — direct in‑process driver:
  `python -u test_claim_pages.py <pdf> <last_pages>`. Prints per‑page PNG
  byte sizes, transcribed char counts, claim summaries, and the full JSON
  dump. Used to validate the path without going through the HTTP API.
- Logger names: `gemma_client` (INFO + WARNING).

### Failure modes & fallbacks

| Failure | Behaviour |
|---|---|
| Vision call `ReadTimeout` | logged as `WARNING gemma_client: Gemma transcribe page failed: ReadTimeout: …`; that page contributes `""`; other pages still extracted. |
| Empty transcript for all pages | `extract_claims_from_pages` returns `[]`; frontend keeps regex‑parser claims (typically also `[]` on scanned PDFs, but does not regress). |
| JSON extraction returns 0 claims | `WARNING gemma_client: Gemma claim extraction returned non‑JSON: …`. Endpoint returns `{claims: []}`; frontend keeps any claims from `/api/parse-patent`. |
| `GEMMA_API_KEY` unset | Gemini path is skipped, Ollama fallback is used (no `responseMimeType` support there — falls back to heuristic JSON extraction). |

### Tunables (env vars)

- `GEMMA_API_KEY` / `GEMINI_API_KEY` — Google AI Studio key.
- `GEMMA_API_MODEL` — model for `analyze_claims` and figure classifier
  (default `gemma-4-31b-it`).
- `GEMMA_VISION_MODEL` — model for OCR transcription and JSON claim
  extraction (default `gemini-2.5-flash`).
- `last_pages` query param on `/api/extract-claims` — default `4`. Bump to
  6–8 for patents with very long claim sections.

### Future work

- Move the figure classifier off `gemma-4-31b-it` onto `gemini-2.5-flash`
  too; the open Gemma model occasionally returns HTTP 500 on multi‑image
  prompts.
- Cache transcripts per `sha256(pdf_bytes)` so repeat uploads of the same
  PDF skip both Gemini calls.
- Detect the claims section heuristically (search transcript backward for
  `What is claimed is:` / `We claim:`) so we can shrink the input to the
  extraction call when the patent has long boilerplate after the claims.
