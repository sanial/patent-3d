# Patent OCR Backend

Dual-pass PDF ingestion pipeline (PyMuPDF + EasyOCR) for the Patent-3D Viewer.
The React frontend lives in [../src/ocr/](../src/ocr/); this folder is
backend-only.

## Stack

| Layer            | Library            | Role                                         |
| ---------------- | ------------------ | -------------------------------------------- |
| Text extraction  | PyMuPDF (`fitz`)   | Structured text, bounding boxes, metadata    |
| Schematic OCR    | EasyOCR            | Text inside rasterised image regions         |
| API server       | FastAPI + uvicorn  | HTTP endpoint consumed by the React frontend |

## Layout

```
python/
├── requirements.txt
├── README.md
├── api/
│   └── api.py                 FastAPI app
└── ocr/
    └── patent_ocr_pipeline.py Core dual-pass pipeline
```

The matching frontend pieces live in `../src/ocr/`:

```
src/ocr/
├── usePatentUpload.ts         React hook
├── PatentUploadPanel.tsx      Drag-and-drop UI component
├── PatentUploadPanel.css      Panel styles
└── app-integration-patch.tsx.txt  App.tsx wiring notes
```

## Quick start (Windows / PowerShell)

```powershell
cd python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api.api:app --reload --port 8000
```

Then in the repo root:

```
# .env.local
VITE_OCR_API_URL=http://localhost:8000
```

## Architecture

```
User uploads PDF
      │
      ▼
POST /api/parse-patent   (FastAPI)
      │
      ├─► Pass 1: PyMuPDF ──► TextBlock list (text, bbox, page)
      │
      └─► Pass 2: EasyOCR ──► FigureHit list (text in image regions)
                │
                ▼
        PatentOCRResult
          ├── title
          ├── abstract
          ├── claims[]
          ├── refEntries { "101": { label, description, snippets, pages } }
          └── figures[]  (EasyOCR hits)
                │
                ▼
        to_patent_data_format()
                │
                ▼
        { "101": { label, description, claims, ... } }
          └── same shape as PATENT_DATA in src/data/patentData.ts
```

## Pipeline detail

### Pass 1 — PyMuPDF

- `page.get_text("dict")` returns structured blocks with per-line bounding boxes.
- Every line becomes a `TextBlock(page, text, bbox, source="pymupdf")`.
- Regex `_REF_LABELLED` scans the concatenated text for definitions like
  `"101 – Sensor Module"` and builds `RefNumEntry` records.
- A second pass over all blocks attaches context snippets and page appearances
  to every ref number found inline.

### Pass 2 — EasyOCR

- The page is rasterised at 2× scale (≈144 dpi) via `page.get_pixmap()`.
- For each embedded image reported by `page.get_images()`, the corresponding
  pixel region is cropped and sent to `reader.readtext()`.
- Results with confidence < 0.30 are discarded.
- Hits land in `PatentOCRResult.figures` — currently informational, but you can
  merge them into `ref_entries` for schematic label extraction.

## Extending

- **GPU acceleration**: pass `gpu=True` to `PatentOCRPipeline` when running on
  a CUDA machine.
- **Additional languages**: `PatentOCRPipeline(langs=["en","ch_sim"])` for
  multilingual patents.
- **Schematic label merge**: filter `figures` hits matching `_REF_INLINE` and
  upgrade their `RefNumEntry.label` from EasyOCR text.
