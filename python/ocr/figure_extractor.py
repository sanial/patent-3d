"""Layer 1 — figure extraction.

Vector-first pipeline. For each page:

1. Collect ``page.get_drawings()`` paths (the diagram lines).
2. Cluster their centroids with DBSCAN → candidate figure regions.
3. Reject clusters in margins or too small.
4. Snap each cluster to the nearest ``FIG. N`` caption.
5. Filter text spans and lead-line paths out of the figure bbox.
6. Render a cleaned PNG (text-masked) and a clean SVG (text-stripped).

The output is a list of :class:`ExtractedFigure` records ready for
serialization into the API response.
"""

from __future__ import annotations

import base64
import io
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN

# ── Tunables ────────────────────────────────────────────────────────────────

DBSCAN_EPS_PT = 40.0      # cluster radius in PDF points
DBSCAN_MIN_SAMPLES = 3
MARGIN_PT = 60.0          # ignore drawings inside this top/bottom margin
MIN_AREA_FRAC = 0.05      # cluster bbox must be ≥ 5% of page area
CAPTION_MAX_BELOW_PT = 80.0
TEXT_BBOX_DILATE_PT = 8.0
RASTER_SCALE = 2.0        # 144 dpi
SVG_NS = "http://www.w3.org/2000/svg"

_CAPTION_RE = re.compile(r"\bFIG\.?\s*(\d+[A-Z]?)\b", re.IGNORECASE)
_REF_INLINE = re.compile(r"\b(\d{2,4})\b")


# ── Data ────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedFigure:
    figure_id: str
    page: int
    bbox: tuple[float, float, float, float]
    caption_text: str = ""
    ref_numbers_originally_inside: list[str] = field(default_factory=list)
    png_base64: str = ""   # data URL body (no prefix)
    svg: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pngDataUrl"] = f"data:image/png;base64,{self.png_base64}" if self.png_base64 else ""
        d["figureId"] = d.pop("figure_id")
        d["captionText"] = d.pop("caption_text")
        d["refNumbersOriginallyInside"] = d.pop("ref_numbers_originally_inside")
        d.pop("png_base64", None)
        return d


# ── Public entry point ──────────────────────────────────────────────────────

def extract_figures(
    doc: "fitz.Document",
    page_ocr: dict[int, list[tuple[str, tuple[float, float, float, float]]]] | None = None,
) -> list[ExtractedFigure]:
    page_ocr = page_ocr or {}
    figures: list[ExtractedFigure] = []
    for page_idx, page in enumerate(doc):
        ocr_lines = page_ocr.get(page_idx)
        page_figs = _extract_page(page, page_idx)
        if not page_figs and ocr_lines is not None:
            # Scanned page: fall back to raster-image figure detection.
            page_figs = _extract_page_scanned(page, page_idx, ocr_lines)
        figures.extend(page_figs)
    return figures


# ── Fast path: figure-only extraction (seconds, parallel) ───────────────────

_BOTTOM_STRIP_PT = 80.0
_CAPTION_OCR_SCALE = 2.0
_CAPTION_OCR_MIN_CONF = 0.25


def extract_figures_fast(
    pdf_bytes: bytes,
    langs: list[str] | None = None,
    gpu: bool = False,
    max_workers: int = 4,
    ocr_captions: bool = True,
    use_gemma: bool = False,
    gemma_thumb_scale: float = 0.9,
) -> list[ExtractedFigure]:
    """Quickly extract figures from a (possibly scanned) patent PDF.

    Strategy per page:
      1. Try the vector-drawing extractor (cheap, deterministic).
      2. Else, treat the page as scanned: pick the largest embedded image as
         the figure, pull its raw bytes via ``doc.extract_image`` (no
         rasterisation), and OCR only the bottom strip to find the
         ``FIG. N`` caption.

    Pages are processed in parallel; each worker opens its own ``fitz.Document``
    because PyMuPDF is not thread-safe across a single document instance.

    When ``use_gemma=True``, page thumbnails are sent to Gemma 4 in one batched
    call to classify figure pages and read captions — much more accurate on
    scanned PDFs than the EasyOCR caption heuristic.
    """
    if use_gemma:
        return _extract_figures_gemma(pdf_bytes, thumb_scale=gemma_thumb_scale)
    # Cheap, single-threaded scan for candidate pages.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        candidates: list[int] = []
        for i, page in enumerate(doc):
            if page.get_images(full=True) or (page.get_drawings() or []):
                candidates.append(i)
    finally:
        doc.close()

    if not candidates:
        return []

    def _process(page_idx: int) -> list[ExtractedFigure]:
        local = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            page = local[page_idx]
            vec = _extract_page(page, page_idx)
            if vec:
                return vec
            return _extract_page_fast_scanned(
                local, page, page_idx, langs or ["en"], gpu, ocr_captions
            )
        finally:
            local.close()

    figures: list[ExtractedFigure] = []
    if max_workers > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for page_figs in ex.map(_process, candidates):
                figures.extend(page_figs)
    else:
        for idx in candidates:
            figures.extend(_process(idx))

    # Deduplicate identical captions across pages (rare but possible).
    figures.sort(key=lambda f: (f.page, f.bbox[1], f.bbox[0]))
    return figures


def _extract_page_fast_scanned(
    doc: "fitz.Document",
    page: "fitz.Page",
    page_idx: int,
    langs: list[str],
    gpu: bool,
    ocr_captions: bool,
) -> list[ExtractedFigure]:
    images = page.get_images(full=True)
    if not images:
        return []
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    # Pick the largest image rect on the page (scanned figure pages are usually
    # a single full-page raster).
    best: tuple[float, int, tuple[float, float, float, float]] | None = None
    for img_info in images:
        xref = img_info[0]
        for r in page.get_image_rects(xref) or []:
            area = max(0.0, (r.x1 - r.x0)) * max(0.0, (r.y1 - r.y0))
            if area <= 0:
                continue
            if best is None or area > best[0]:
                best = (area, xref, (r.x0, r.y0, r.x1, r.y1))

    if best is None:
        return []
    area, xref, bbox = best
    if area < page_area * _MIN_SCANNED_AREA_FRAC:
        return []

    # Caption (optional — skip to save time).
    caption_text, caption_num = ("", None)
    if ocr_captions:
        caption_text, caption_num = _ocr_caption_at_bottom(page, langs, gpu, bbox)

    # When captions are requested, drop pages without a FIG. N caption — they
    # are text pages, not diagrams.
    if ocr_captions and caption_num is None:
        return []

    figure_id = f"FIG_{caption_num}" if caption_num else f"P{page_idx + 1}"

    # Get the image bytes directly — no PDF rasterisation needed.
    png_b64 = _image_b64_from_xref(doc, xref)
    if not png_b64:
        # Fallback: rasterise the bbox.
        clip = fitz.Rect(*bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return [
        ExtractedFigure(
            figure_id=figure_id,
            page=page_idx,
            bbox=bbox,
            caption_text=caption_text,
            ref_numbers_originally_inside=[],
            png_base64=png_b64,
            svg="",
        )
    ]


def _image_b64_from_xref(doc: "fitz.Document", xref: int) -> str:
    """Return base64-encoded PNG for the embedded image at ``xref``."""
    try:
        info = doc.extract_image(xref)
    except Exception:
        return ""
    raw = info.get("image")
    ext = (info.get("ext") or "").lower()
    if not raw:
        return ""
    if ext == "png":
        return base64.b64encode(raw).decode("ascii")
    # Re-encode other formats (jpeg, jp2, jbig2, etc.) as PNG.
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _ocr_caption_at_bottom(
    page: "fitz.Page",
    langs: list[str],
    gpu: bool,
    image_bbox: tuple[float, float, float, float],
) -> tuple[str, str | None]:
    """OCR only a small strip below the figure to recover the ``FIG. N`` caption."""
    # Local import to avoid a circular dep at module load time.
    from .patent_ocr_pipeline import _get_reader  # type: ignore

    reader = _get_reader(langs, gpu)
    page_rect = page.rect
    y_top = max(image_bbox[3] - 4.0, page_rect.height - _BOTTOM_STRIP_PT)
    y_top = min(page_rect.height - 10.0, y_top)
    strip = fitz.Rect(0.0, y_top, page_rect.width, page_rect.height)
    if strip.height < 10:
        return ("", None)
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(_CAPTION_OCR_SCALE, _CAPTION_OCR_SCALE),
            clip=strip,
            alpha=False,
        )
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        out = reader.readtext(arr)
    except Exception:
        return ("", None)

    # Collect kept detections with row/col anchors so we can reconstruct
    # reading order. EasyOCR often splits ``FIG.`` and ``10`` into two tokens.
    detections: list[tuple[float, float, str]] = []
    for bbox_pts, text, conf in out:
        if conf < _CAPTION_OCR_MIN_CONF or not text.strip():
            continue
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        cy = (min(ys) + max(ys)) / 2
        cx = min(xs)
        detections.append((cy, cx, text.strip()))
    if not detections:
        return ("", None)

    # Group into rows by y (≈12 px tolerance at scale 2.0 → ~6pt).
    detections.sort(key=lambda d: (round(d[0] / 12.0), d[1]))
    rows: list[list[tuple[float, float, str]]] = []
    cur: list[tuple[float, float, str]] = []
    cur_key: float | None = None
    for d in detections:
        k = round(d[0] / 12.0)
        if cur_key is None or k == cur_key:
            cur.append(d)
            cur_key = k
        else:
            rows.append(cur)
            cur = [d]
            cur_key = k
    if cur:
        rows.append(cur)

    # Scan each row for a FIG. N caption.
    for row in rows:
        line = " ".join(t for _, _, t in row)
        m = _CAPTION_LINE_RE.search(line)
        if m:
            return (line, m.group(1))
    return ("", None)


# ── Gemma-backed fast path ──────────────────────────────────────────────────

def _extract_figures_gemma(pdf_bytes: bytes, thumb_scale: float = 0.9) -> list[ExtractedFigure]:
    """Use Gemma 4 multimodal to identify figure pages, then return their
    embedded image bytes."""
    import asyncio
    from llm.gemma_client import GemmaClient  # type: ignore

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        thumbs: list[str] = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(thumb_scale, thumb_scale), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70, optimize=True)
            thumbs.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        client = GemmaClient()
        classifications = asyncio.run(client.classify_figure_pages(thumbs))

        figures: list[ExtractedFigure] = []
        for entry in classifications:
            pidx = entry["page"]
            if pidx < 0 or pidx >= doc.page_count:
                continue
            page = doc[pidx]
            page_rect = page.rect
            images = page.get_images(full=True)
            png_b64 = ""
            bbox: tuple[float, float, float, float] = (0.0, 0.0, page_rect.width, page_rect.height)
            if images:
                best: tuple[float, int, tuple[float, float, float, float]] | None = None
                for img_info in images:
                    xref = img_info[0]
                    for r in page.get_image_rects(xref) or []:
                        area = max(0.0, (r.x1 - r.x0)) * max(0.0, (r.y1 - r.y0))
                        if area <= 0:
                            continue
                        if best is None or area > best[0]:
                            best = (area, xref, (r.x0, r.y0, r.x1, r.y1))
                if best is not None:
                    _a, xref, bbox = best
                    png_b64 = _image_b64_from_xref(doc, xref)
            if not png_b64:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            figures.append(
                ExtractedFigure(
                    figure_id=str(entry.get("figure_id") or f"P{pidx + 1}"),
                    page=pidx,
                    bbox=bbox,
                    caption_text=str(entry.get("caption") or ""),
                    ref_numbers_originally_inside=[],
                    png_base64=png_b64,
                    svg="",
                )
            )

        figures.sort(key=lambda f: (f.page, f.bbox[1]))
        return figures
    finally:
        doc.close()


# ── Per-page pipeline ───────────────────────────────────────────────────────

def _extract_page(page: "fitz.Page", page_idx: int) -> list[ExtractedFigure]:
    drawings = page.get_drawings() or []
    if not drawings:
        return []

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    # Build path bboxes (skip degenerate ones).
    path_bboxes: list[tuple[float, float, float, float]] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        if r.width < 0.5 and r.height < 0.5:
            continue
        # Skip drawings in top/bottom margin.
        if r.y0 < MARGIN_PT or r.y1 > page_rect.height - MARGIN_PT:
            continue
        path_bboxes.append((r.x0, r.y0, r.x1, r.y1))

    if len(path_bboxes) < DBSCAN_MIN_SAMPLES:
        return []

    # Cluster centroids
    centroids = np.array(
        [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in path_bboxes]
    )
    labels = DBSCAN(eps=DBSCAN_EPS_PT, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(centroids)

    # Group bboxes by cluster.
    clusters: dict[int, list[tuple[float, float, float, float]]] = {}
    for lbl, bb in zip(labels, path_bboxes):
        if lbl < 0:  # noise
            continue
        clusters.setdefault(int(lbl), []).append(bb)

    # Captions and text-span bboxes (for filtering).
    text_dict = page.get_text("dict")
    captions = _find_captions(text_dict)
    text_spans = _collect_text_spans(text_dict)

    out: list[ExtractedFigure] = []
    for cluster_idx, bbs in clusters.items():
        cluster_bbox = _union_bbox(bbs)
        cluster_area = (cluster_bbox[2] - cluster_bbox[0]) * (cluster_bbox[3] - cluster_bbox[1])
        if cluster_area < page_area * MIN_AREA_FRAC:
            continue

        caption_text, caption_num = _nearest_caption(cluster_bbox, captions)
        figure_id = f"FIG_{caption_num}" if caption_num else f"P{page_idx + 1}_C{cluster_idx}"

        # Pad bbox slightly to include thin border lines + caption.
        padded = _pad_bbox(cluster_bbox, pad=6.0, page_rect=page_rect)

        # Ref numbers that were inside this figure (before stripping).
        refs_inside = _refs_inside(padded, text_spans)

        png_b64 = _render_clean_png(page, padded, text_spans)
        svg = _render_clean_svg(page, padded)

        out.append(
            ExtractedFigure(
                figure_id=figure_id,
                page=page_idx,
                bbox=padded,
                caption_text=caption_text,
                ref_numbers_originally_inside=refs_inside,
                png_base64=png_b64,
                svg=svg,
            )
        )

    # Sort by caption number then position
    out.sort(key=lambda f: (f.page, f.bbox[1], f.bbox[0]))
    return out


# ── Scanned-page fallback ───────────────────────────────────────────────────

_CAPTION_LINE_RE = re.compile(r"\bFIG\.?\s*(\d+[A-Z]?)\b", re.IGNORECASE)
_MIN_SCANNED_AREA_FRAC = 0.04


def _extract_page_scanned(
    page: "fitz.Page",
    page_idx: int,
    ocr_lines: list[tuple[str, tuple[float, float, float, float]]],
) -> list[ExtractedFigure]:
    """Detect figures on a scanned page using image rects + OCR captions.

    For each ``FIG. N`` caption, take the connected image region just above
    it. If there are no image rects, use the page minus margins.
    """
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    # Collect all image rects on the page.
    img_rects: list[tuple[float, float, float, float]] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        for r in page.get_image_rects(xref) or []:
            img_rects.append((r.x0, r.y0, r.x1, r.y1))

    # Find caption lines.
    captions: list[tuple[str, str, tuple[float, float, float, float]]] = []
    for text, bbox in ocr_lines:
        m = _CAPTION_LINE_RE.search(text)
        if m:
            captions.append((text, m.group(1), bbox))

    if not captions:
        return []

    out: list[ExtractedFigure] = []
    for caption_text, caption_num, cap_bbox in captions:
        # Region = page minus margins, clipped above the caption.
        bbox = (
            MARGIN_PT,
            MARGIN_PT,
            page_rect.width - MARGIN_PT,
            max(MARGIN_PT + 50.0, cap_bbox[1] - 4.0),
        )

        # If any image rect intersects this region, tighten to the union.
        intersecting = [
            r for r in img_rects if _bbox_intersects(bbox, r) and r[3] <= cap_bbox[1] + 8.0
        ]
        if intersecting:
            bbox = _union_bbox(intersecting)
            # Include the caption.
            bbox = (
                min(bbox[0], cap_bbox[0]),
                bbox[1],
                max(bbox[2], cap_bbox[2]),
                max(bbox[3], cap_bbox[3]),
            )

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < page_area * _MIN_SCANNED_AREA_FRAC:
            continue

        padded = _pad_bbox(bbox, pad=6.0, page_rect=page_rect)

        # Ref numbers inside the region (from OCR).
        refs_inside = sorted(
            {
                ref
                for text, lb in ocr_lines
                if _bbox_intersects(padded, lb)
                for ref in _REF_INLINE.findall(text)
            }
        )

        # Render PNG of the region (no text masking — OCR text is part of
        # the diagram on scanned pages).
        clip = fitz.Rect(*padded)
        pix = page.get_pixmap(matrix=fitz.Matrix(RASTER_SCALE, RASTER_SCALE), clip=clip, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        out.append(
            ExtractedFigure(
                figure_id=f"FIG_{caption_num}",
                page=page_idx,
                bbox=padded,
                caption_text=caption_text,
                ref_numbers_originally_inside=refs_inside,
                png_base64=png_b64,
                svg="",
            )
        )

    out.sort(key=lambda f: (f.page, f.bbox[1], f.bbox[0]))
    return out


# ── Helpers ─────────────────────────────────────────────────────────────────

def _union_bbox(bbs: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in bbs)
    y0 = min(b[1] for b in bbs)
    x1 = max(b[2] for b in bbs)
    y1 = max(b[3] for b in bbs)
    return (x0, y0, x1, y1)


def _pad_bbox(
    bbox: tuple[float, float, float, float],
    pad: float,
    page_rect: "fitz.Rect",
) -> tuple[float, float, float, float]:
    return (
        max(0.0, bbox[0] - pad),
        max(0.0, bbox[1] - pad),
        min(page_rect.width, bbox[2] + pad),
        min(page_rect.height, bbox[3] + pad),
    )


def _find_captions(text_dict: dict) -> list[tuple[str, str, tuple[float, float, float, float]]]:
    """Return list of (full_caption, number, bbox)."""
    out: list[tuple[str, str, tuple[float, float, float, float]]] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            text = " ".join(s.get("text", "") for s in line.get("spans", [])).strip()
            m = _CAPTION_RE.search(text)
            if m:
                bbox = tuple(line.get("bbox", (0, 0, 0, 0)))  # type: ignore[arg-type]
                out.append((text, m.group(1), bbox))
    return out


def _collect_text_spans(text_dict: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                out.append((text, tuple(span.get("bbox", (0, 0, 0, 0)))))  # type: ignore[arg-type]
    return out


def _nearest_caption(
    cluster_bbox: tuple[float, float, float, float],
    captions: list[tuple[str, str, tuple[float, float, float, float]]],
) -> tuple[str, str | None]:
    """Snap to the caption that sits just below the cluster within CAPTION_MAX_BELOW_PT."""
    cx = (cluster_bbox[0] + cluster_bbox[2]) / 2
    cy_bottom = cluster_bbox[3]
    best: tuple[float, str, str] | None = None
    for full, num, bb in captions:
        bx = (bb[0] + bb[2]) / 2
        by_top = bb[1]
        dy = by_top - cy_bottom
        if dy < -10.0 or dy > CAPTION_MAX_BELOW_PT:
            continue
        score = abs(dy) + 0.3 * abs(bx - cx)
        if best is None or score < best[0]:
            best = (score, full, num)
    if best is None:
        return ("", None)
    return (best[1], best[2])


def _refs_inside(
    bbox: tuple[float, float, float, float],
    text_spans: list[tuple[str, tuple[float, float, float, float]]],
) -> list[str]:
    refs: set[str] = set()
    for text, bb in text_spans:
        if not _bbox_intersects(bbox, bb):
            continue
        for m in _REF_INLINE.findall(text):
            refs.add(m)
    return sorted(refs)


def _bbox_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (b[2] < a[0] or b[0] > a[2] or b[3] < a[1] or b[1] > a[3])


def _render_clean_png(
    page: "fitz.Page",
    bbox: tuple[float, float, float, float],
    text_spans: list[tuple[str, tuple[float, float, float, float]]],
) -> str:
    """Rasterise the figure region, mask text-span bboxes to white, return base64."""
    clip = fitz.Rect(*bbox)
    matrix = fitz.Matrix(RASTER_SCALE, RASTER_SCALE)
    pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    draw = ImageDraw.Draw(img)
    for _text, span_bb in text_spans:
        if not _bbox_intersects(bbox, span_bb):
            continue
        # Translate span bbox into local pixel coords.
        x0 = (span_bb[0] - bbox[0] - TEXT_BBOX_DILATE_PT) * RASTER_SCALE
        y0 = (span_bb[1] - bbox[1] - TEXT_BBOX_DILATE_PT) * RASTER_SCALE
        x1 = (span_bb[2] - bbox[0] + TEXT_BBOX_DILATE_PT) * RASTER_SCALE
        y1 = (span_bb[3] - bbox[1] + TEXT_BBOX_DILATE_PT) * RASTER_SCALE
        draw.rectangle([x0, y0, x1, y1], fill="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_clean_svg(page: "fitz.Page", bbox: tuple[float, float, float, float]) -> str:
    """Get the page SVG clipped to bbox, then strip <text> elements."""
    try:
        svg = page.get_svg_image(matrix=fitz.Matrix(1, 1))
    except Exception:
        return ""
    return _strip_svg_text_in_bbox(svg, bbox)


def _strip_svg_text_in_bbox(svg: str, bbox: tuple[float, float, float, float]) -> str:
    """Remove <text> nodes from the SVG. PyMuPDF SVG has all text as <text>."""
    try:
        # Default namespace makes ET awkward; register & parse.
        ET.register_namespace("", SVG_NS)
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg

    # Walk and remove text elements wherever they appear.
    for parent in root.iter():
        to_remove = [child for child in list(parent) if child.tag.endswith("}text")]
        for child in to_remove:
            parent.remove(child)

    # Clip viewBox to the figure bbox so it renders tightly.
    root.set("viewBox", f"{bbox[0]} {bbox[1]} {bbox[2] - bbox[0]} {bbox[3] - bbox[1]}")
    root.set("width", f"{bbox[2] - bbox[0]}")
    root.set("height", f"{bbox[3] - bbox[1]}")
    return ET.tostring(root, encoding="unicode")


__all__ = ["ExtractedFigure", "extract_figures"]
