"""Dual-pass patent PDF OCR pipeline.

Pass 1 — PyMuPDF: structured text extraction with per-line bounding boxes.
Pass 2 — EasyOCR:  text inside rasterised image regions (schematics).

The two passes feed a single :class:`PatentOCRResult` whose shape mirrors the
``PATENT_DATA`` map consumed by the Three.js frontend (see
``src/hooks/useThreeScene.ts`` / ``src/data/patentData.ts``).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .figure_extractor import ExtractedFigure, extract_figures
from .structure import StructuredPatent, parse_structure

# EasyOCR is imported lazily — it pulls in torch which is slow to load.
_easyocr_reader: Any = None


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    source: str  # "pymupdf" | "easyocr"


@dataclass
class FigureHit:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class RefNumEntry:
    ref: str
    label: str = ""
    description: str = ""
    snippets: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


@dataclass
class PatentOCRResult:
    title: str = ""
    abstract: str = ""
    claims: list[str] = field(default_factory=list)
    ref_entries: dict[str, RefNumEntry] = field(default_factory=dict)
    figures: list[FigureHit] = field(default_factory=list)
    extracted_figures: list[ExtractedFigure] = field(default_factory=list)
    structure: StructuredPatent | None = None
    full_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "claims": self.claims,
            "refEntries": {k: asdict(v) for k, v in self.ref_entries.items()},
            "figures": [asdict(f) for f in self.figures],
            "extractedFigures": [f.to_dict() for f in self.extracted_figures],
            "structure": self.structure.to_dict() if self.structure else None,
        }

    def to_patent_data_format(self) -> dict[str, dict[str, Any]]:
        """Return a map shaped like the frontend ``PATENT_DATA`` constant."""
        out: dict[str, dict[str, Any]] = {}
        for ref, entry in self.ref_entries.items():
            out[ref] = {
                "label": entry.label or f"Reference {ref}",
                "description": entry.description or " ".join(entry.snippets[:2]),
                "claims": _claims_mentioning(ref, self.claims),
                "pages": entry.pages,
            }
        return out


# ── Regex patterns ──────────────────────────────────────────────────────────

# "101 – Sensor Module"  /  "101: Sensor Module"  /  "101 Sensor Module"
_REF_LABELLED = re.compile(
    r"\b(?P<ref>\d{2,4})\s*[–\-:.)]?\s*(?P<label>[A-Z][A-Za-z0-9 \-/]{2,60})"
)

# Inline ref mentions: "the sensor module 101" / "module (101)"
_REF_INLINE = re.compile(r"\b(\d{2,4})\b")

_TITLE_HINTS = ("title of the invention", "title:")
_ABSTRACT_HINTS = ("abstract",)
_CLAIMS_HINTS = ("what is claimed", "claims:", "we claim")


# ── Pipeline ────────────────────────────────────────────────────────────────

class PatentOCRPipeline:
    def __init__(
        self,
        langs: list[str] | None = None,
        gpu: bool = False,
        min_confidence: float = 0.30,
        image_scale: float = 2.0,
    ):
        self.langs = langs or ["en"]
        self.gpu = gpu
        self.min_confidence = min_confidence
        self.image_scale = image_scale

    # -- public entry point -----------------------------------------------

    def run(self, pdf_bytes: bytes) -> PatentOCRResult:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            # OCR scanned pages once and reuse across passes.
            page_ocr: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = {}
            for i, page in enumerate(doc):
                if not _page_has_native_text(page):
                    page_ocr[i] = self._ocr_page_lines(page)

            blocks = self._pass1_pymupdf(doc, page_ocr)
            figures = self._pass2_easyocr(doc)
            extracted = extract_figures(doc, page_ocr)
            result = self._assemble(blocks, figures)
            result.extracted_figures = extracted

            # Structure parse from the assembled full text.
            full_text = "\n".join(b.text for b in blocks)
            result.full_text = full_text
            result.structure = parse_structure(full_text)

            # Promote structure.refDefinitions into ref_entries labels.
            for ref, label in result.structure.ref_definitions.items():
                entry = result.ref_entries.setdefault(ref, RefNumEntry(ref=ref))
                if not entry.label:
                    entry.label = label
            return result
        finally:
            doc.close()

    # -- pass 1 ------------------------------------------------------------

    def _pass1_pymupdf(
        self,
        doc: "fitz.Document",
        page_ocr: dict[int, list[tuple[str, tuple[float, float, float, float]]]] | None = None,
    ) -> list[TextBlock]:
        page_ocr = page_ocr or {}
        blocks: list[TextBlock] = []
        for page_idx, page in enumerate(doc):
            had_native = False
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    text = " ".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if not text:
                        continue
                    had_native = True
                    blocks.append(
                        TextBlock(
                            page=page_idx,
                            text=text,
                            bbox=tuple(line.get("bbox", (0, 0, 0, 0))),  # type: ignore[arg-type]
                            source="pymupdf",
                        )
                    )
            # Fallback for scanned pages: use whole-page OCR.
            if not had_native and page_idx in page_ocr:
                for text, bbox in page_ocr[page_idx]:
                    blocks.append(
                        TextBlock(page=page_idx, text=text, bbox=bbox, source="easyocr")
                    )
        return blocks

    # -- per-page OCR helper ---------------------------------------------

    def _ocr_page_lines(
        self, page: "fitz.Page"
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """OCR an entire page and return ``(text, bbox-in-PDF-points)`` lines."""
        reader = _get_reader(self.langs, self.gpu)
        matrix = fitz.Matrix(self.image_scale, self.image_scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        try:
            ocr_out = reader.readtext(arr)
        except Exception:
            return []
        sx = page.rect.width / pix.width
        sy = page.rect.height / pix.height
        lines: list[tuple[str, tuple[float, float, float, float]]] = []
        for bbox_pts, text, conf in ocr_out:
            if conf < self.min_confidence or not text.strip():
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            bbox = (
                float(min(xs)) * sx,
                float(min(ys)) * sy,
                float(max(xs)) * sx,
                float(max(ys)) * sy,
            )
            lines.append((text.strip(), bbox))
        # Sort roughly top-to-bottom, left-to-right (8 pt row tolerance).
        lines.sort(key=lambda t: (round(t[1][1] / 8.0), t[1][0]))
        return _merge_ocr_lines(lines)

    # -- pass 2 ------------------------------------------------------------

    def _pass2_easyocr(self, doc: "fitz.Document") -> list[FigureHit]:
        reader = _get_reader(self.langs, self.gpu)
        hits: list[FigureHit] = []
        matrix = fitz.Matrix(self.image_scale, self.image_scale)

        for page_idx, page in enumerate(doc):
            images = page.get_images(full=True)
            if not images:
                continue

            pix = page.get_pixmap(matrix=matrix, alpha=False)
            page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            page_w, page_h = page.rect.width, page.rect.height

            for img_info in images:
                xref = img_info[0]
                rects = page.get_image_rects(xref) or []
                for rect in rects:
                    # Map PDF-space rect → rasterised pixel coords.
                    x0 = int(rect.x0 / page_w * pix.width)
                    y0 = int(rect.y0 / page_h * pix.height)
                    x1 = int(rect.x1 / page_w * pix.width)
                    y1 = int(rect.y1 / page_h * pix.height)
                    if x1 <= x0 or y1 <= y0:
                        continue

                    crop = page_img.crop((x0, y0, x1, y1))
                    arr = np.array(crop)
                    try:
                        ocr_out = reader.readtext(arr)
                    except Exception:
                        continue

                    for bbox_pts, text, conf in ocr_out:
                        if conf < self.min_confidence or not text.strip():
                            continue
                        xs = [p[0] for p in bbox_pts]
                        ys = [p[1] for p in bbox_pts]
                        hits.append(
                            FigureHit(
                                page=page_idx,
                                text=text.strip(),
                                bbox=(min(xs), min(ys), max(xs), max(ys)),
                                confidence=float(conf),
                            )
                        )
        return hits

    # -- assembly ----------------------------------------------------------

    def _assemble(self, blocks: list[TextBlock], figures: list[FigureHit]) -> PatentOCRResult:
        result = PatentOCRResult(figures=figures)

        full_text_by_page: dict[int, str] = {}
        for b in blocks:
            full_text_by_page.setdefault(b.page, "")
            full_text_by_page[b.page] += b.text + "\n"

        full_text = "\n".join(full_text_by_page[p] for p in sorted(full_text_by_page))

        result.title = _extract_title(blocks)
        result.abstract = _extract_section(full_text, _ABSTRACT_HINTS, _CLAIMS_HINTS)
        result.claims = _extract_claims(full_text)

        # Labelled ref definitions ("101 – Sensor Module")
        for m in _REF_LABELLED.finditer(full_text):
            ref = m.group("ref")
            label = m.group("label").strip().rstrip(".,;:")
            if len(label) < 3:
                continue
            entry = result.ref_entries.setdefault(ref, RefNumEntry(ref=ref))
            # Keep the shortest plausible label — usually the actual component name.
            if not entry.label or len(label) < len(entry.label):
                entry.label = label

        # Inline appearances → snippets + pages
        for b in blocks:
            for ref in set(_REF_INLINE.findall(b.text)):
                if ref not in result.ref_entries:
                    # Only track refs that already have a definition.
                    continue
                entry = result.ref_entries[ref]
                if b.page not in entry.pages:
                    entry.pages.append(b.page)
                if len(entry.snippets) < 5:
                    entry.snippets.append(b.text)

        # Promote first long snippet to description if none set yet.
        for entry in result.ref_entries.values():
            if not entry.description and entry.snippets:
                entry.description = max(entry.snippets, key=len)

        return result


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_reader(langs: list[str], gpu: bool):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # local import keeps import-time light

        _easyocr_reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    return _easyocr_reader


def _page_has_native_text(page: "fitz.Page") -> bool:
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    return True
    return False


def _merge_ocr_lines(
    lines: list[tuple[str, tuple[float, float, float, float]]],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Merge OCR fragments that sit on the same baseline into single lines.

    EasyOCR returns word-level boxes; structure/claim parsing expects lines.
    """
    if not lines:
        return []
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    cur_text = lines[0][0]
    cur_bbox = list(lines[0][1])
    cur_row_y = (cur_bbox[1] + cur_bbox[3]) / 2
    for text, bbox in lines[1:]:
        row_y = (bbox[1] + bbox[3]) / 2
        same_row = abs(row_y - cur_row_y) < 6.0
        if same_row:
            cur_text = f"{cur_text} {text}"
            cur_bbox[0] = min(cur_bbox[0], bbox[0])
            cur_bbox[1] = min(cur_bbox[1], bbox[1])
            cur_bbox[2] = max(cur_bbox[2], bbox[2])
            cur_bbox[3] = max(cur_bbox[3], bbox[3])
        else:
            out.append((cur_text, tuple(cur_bbox)))  # type: ignore[arg-type]
            cur_text = text
            cur_bbox = list(bbox)
            cur_row_y = row_y
    out.append((cur_text, tuple(cur_bbox)))  # type: ignore[arg-type]
    return out


def _extract_title(blocks: list[TextBlock]) -> str:
    # Prefer an explicit "Title of the invention" line; otherwise the first
    # non-trivial line on page 0.
    for i, b in enumerate(blocks):
        low = b.text.lower()
        if any(h in low for h in _TITLE_HINTS) and i + 1 < len(blocks):
            return blocks[i + 1].text.strip()
    for b in blocks:
        if b.page == 0 and len(b.text) > 10 and not b.text.isdigit():
            return b.text.strip()
    return ""


def _extract_section(full_text: str, start_hints: tuple[str, ...], end_hints: tuple[str, ...]) -> str:
    low = full_text.lower()
    start = -1
    for h in start_hints:
        idx = low.find(h)
        if idx >= 0:
            start = idx + len(h)
            break
    if start < 0:
        return ""
    end = len(full_text)
    for h in end_hints:
        idx = low.find(h, start)
        if idx >= 0:
            end = min(end, idx)
    return full_text[start:end].strip(" :\n\r\t")


def _extract_claims(full_text: str) -> list[str]:
    low = full_text.lower()
    start = -1
    for h in _CLAIMS_HINTS:
        idx = low.find(h)
        if idx >= 0:
            start = idx + len(h)
            break
    if start < 0:
        return []
    body = full_text[start:].strip(" :\n\r\t")
    # Split on numbered claims: "1.", "2.", ...
    parts = re.split(r"\n\s*(\d{1,3})\.\s+", body)
    claims: list[str] = []
    # parts = [pre, "1", text1, "2", text2, ...]
    for i in range(1, len(parts) - 1, 2):
        text = parts[i + 1].strip()
        if text:
            claims.append(text)
    return claims


def _claims_mentioning(ref: str, claims: list[str]) -> list[int]:
    out: list[int] = []
    pat = re.compile(rf"\b{re.escape(ref)}\b")
    for i, c in enumerate(claims, start=1):
        if pat.search(c):
            out.append(i)
    return out


__all__ = [
    "PatentOCRPipeline",
    "PatentOCRResult",
    "RefNumEntry",
    "FigureHit",
    "TextBlock",
]
