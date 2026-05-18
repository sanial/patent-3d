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
