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

def extract_figures(doc: "fitz.Document") -> list[ExtractedFigure]:
    figures: list[ExtractedFigure] = []
    for page_idx, page in enumerate(doc):
        for fig in _extract_page(page, page_idx):
            figures.append(fig)
    return figures


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
