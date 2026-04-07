"""
PDF layout segmentation for two-column academic papers.

Detects full-width regions (title, figures, tables) and two-column body
text, then writes each region as a separate page in a new PDF so that
the natural reading order is preserved.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default tuning constants
# ---------------------------------------------------------------------------

DEFAULT_LINE_MERGE_GAP = 3.0
DEFAULT_BAND_MERGE_GAP = 14.0
DEFAULT_BLOCK_PADDING = 2.0
DEFAULT_SEGMENT_PADDING = 3.0
DEFAULT_MIN_TEXT_HEIGHT = 5.0
DEFAULT_MIN_WORD_CHARS = 1
DEFAULT_LINE_SPLIT_GAP_X = 14.0

DEFAULT_CAPTION_MERGE_GAP = 35.0

DEFAULT_NUM_BINS = 120
DEFAULT_SINGLE_COVERAGE_THRESHOLD = 0.68
DEFAULT_MIN_SIDE_COVERAGE = 0.55
DEFAULT_MIN_CENTER_GAP_RATIO = 0.08
DEFAULT_MIN_BAND_HEIGHT_RATIO = 0.018

DEFAULT_HRULE_COVERAGE = 0.60

RectTuple = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LayoutConfig:
    """Tuning parameters for the layout segmentation algorithm."""

    line_merge_gap: float = DEFAULT_LINE_MERGE_GAP
    band_merge_gap: float = DEFAULT_BAND_MERGE_GAP
    block_padding: float = DEFAULT_BLOCK_PADDING
    segment_padding: float = DEFAULT_SEGMENT_PADDING
    min_text_height: float = DEFAULT_MIN_TEXT_HEIGHT
    min_word_chars: int = DEFAULT_MIN_WORD_CHARS
    line_split_gap_x: float = DEFAULT_LINE_SPLIT_GAP_X
    num_bins: int = DEFAULT_NUM_BINS
    single_coverage_threshold: float = DEFAULT_SINGLE_COVERAGE_THRESHOLD
    min_side_coverage: float = DEFAULT_MIN_SIDE_COVERAGE
    min_center_gap_ratio: float = DEFAULT_MIN_CENTER_GAP_RATIO
    min_band_height_ratio: float = DEFAULT_MIN_BAND_HEIGHT_RATIO
    caption_merge_gap: float = DEFAULT_CAPTION_MERGE_GAP
    hrule_coverage: float = DEFAULT_HRULE_COVERAGE


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class TextLine:
    rect: fitz.Rect
    words: List[RectTuple]


@dataclass
class TextBand:
    rect: fitz.Rect
    words: List[RectTuple]
    lines: List[TextLine]


@dataclass
class Segment:
    """A rectangular region of a PDF page in reading order."""

    page_index: int
    rect: fitz.Rect
    label: str  # "single_column", "doublecolumn_left", "doublecolumn_right"
    order: int


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _clamp_rect(rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    x0 = max(page_rect.x0, rect.x0)
    y0 = max(page_rect.y0, rect.y0)
    x1 = min(page_rect.x1, rect.x1)
    y1 = min(page_rect.y1, rect.y1)
    if x1 <= x0 or y1 <= y0:
        return fitz.Rect(0, 0, 0, 0)
    return fitz.Rect(x0, y0, x1, y1)


def _expand_rect(rect: fitz.Rect, pad: float, page_rect: fitz.Rect) -> fitz.Rect:
    expanded = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    return _clamp_rect(expanded, page_rect)


# ---------------------------------------------------------------------------
# Page spec parsing
# ---------------------------------------------------------------------------


def _parse_page_spec(page_spec: Optional[str], total_pages: int) -> List[int]:
    if not page_spec:
        return list(range(total_pages))
    result: set[int] = set()
    for part in (chunk.strip() for chunk in page_spec.split(",") if chunk.strip()):
        if "-" in part:
            left, right = (x.strip() for x in part.split("-", 1))
            start, end = int(left), int(right)
            if end < start:
                start, end = end, start
            result.update(v for v in range(start, end + 1) if 0 <= v < total_pages)
        else:
            v = int(part)
            if 0 <= v < total_pages:
                result.add(v)
    return sorted(result)


# ---------------------------------------------------------------------------
# Word / line extraction
# ---------------------------------------------------------------------------


def _extract_text_lines(page: fitz.Page, config: LayoutConfig) -> List[TextLine]:
    page_rect = page.rect
    words: List[RectTuple] = []
    for entry in page.get_text("words"):
        x0, y0, x1, y1, text = entry[:5]
        if not text or len(text.strip()) < config.min_word_chars:
            continue
        if (y1 - y0) < config.min_text_height:
            continue
        rect = _expand_rect(fitz.Rect(x0, y0, x1, y1), config.block_padding, page_rect)
        words.append((rect.x0, rect.y0, rect.x1, rect.y1))

    if not words:
        return []

    words_with_center = [(((w[1] + w[3]) / 2.0), w) for w in words]
    words_with_center.sort(key=lambda pair: (pair[0], pair[1][0]))

    lines: List[TextLine] = []
    current_words: List[RectTuple] = []
    current_center_y = 0.0

    def flush_line() -> None:
        if not current_words:
            return
        row_words = sorted(current_words, key=lambda w: w[0])
        split_gap = max(config.line_split_gap_x, page_rect.width * 0.025)
        chunk: List[RectTuple] = []

        def flush_chunk() -> None:
            if not chunk:
                return
            x0 = min(w[0] for w in chunk)
            y0 = min(w[1] for w in chunk)
            x1 = max(w[2] for w in chunk)
            y1 = max(w[3] for w in chunk)
            lines.append(TextLine(rect=fitz.Rect(x0, y0, x1, y1), words=list(chunk)))
            chunk.clear()

        for word in row_words:
            if not chunk:
                chunk.append(word)
                continue
            gap_x = word[0] - chunk[-1][2]
            true_gap = gap_x + 2 * config.block_padding
            if true_gap > split_gap:
                flush_chunk()
            chunk.append(word)
        flush_chunk()
        current_words.clear()

    for center_y, word in words_with_center:
        if not current_words:
            current_words.append(word)
            current_center_y = center_y
            continue
        if abs(center_y - current_center_y) <= config.line_merge_gap:
            current_words.append(word)
            n = len(current_words)
            current_center_y = (current_center_y * (n - 1) + center_y) / n
            continue
        flush_line()
        current_words.append(word)
        current_center_y = center_y
    flush_line()

    return lines


# ---------------------------------------------------------------------------
# Image / drawing extraction
# ---------------------------------------------------------------------------


def _extract_image_rects(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    for item in page.get_image_info():
        bbox = item.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        if (x1 - x0) > 1 and (y1 - y0) > 1:
            rects.append(fitz.Rect(x0, y0, x1, y1))
    return rects


def _cluster_image_rects(
    rects: Sequence[fitz.Rect], merge_gap: float = 30.0
) -> List[fitz.Rect]:
    if not rects:
        return []
    sorted_rects = sorted(rects, key=lambda r: r.y0)
    clusters: List[fitz.Rect] = []
    cur = fitz.Rect(sorted_rects[0])
    for rect in sorted_rects[1:]:
        if rect.y0 <= cur.y1 + merge_gap:
            cur = cur | rect
        else:
            clusters.append(cur)
            cur = fitz.Rect(rect)
    clusters.append(cur)
    return clusters


def _extract_horizontal_rules(
    page: fitz.Page, content_width: float, config: LayoutConfig
) -> List[float]:
    min_span = content_width * config.hrule_coverage
    y_positions: List[float] = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None:
            continue
        height = abs(r.y1 - r.y0)
        width = abs(r.x1 - r.x0)
        if height < 2.0 and width >= min_span:
            y_positions.append((r.y0 + r.y1) / 2.0)
    y_positions.sort()

    deduped: List[float] = []
    for y in y_positions:
        if not deduped or abs(y - deduped[-1]) > 3.0:
            deduped.append(y)

    TABLE_CLUSTER_SPAN = 300.0
    TABLE_CLUSTER_MIN = 3
    keep: List[float] = []
    for i, y in enumerate(deduped):
        neighbors = sum(1 for y2 in deduped if abs(y2 - y) <= TABLE_CLUSTER_SPAN)
        if neighbors < TABLE_CLUSTER_MIN:
            keep.append(y)
    return keep


# ---------------------------------------------------------------------------
# Band building
# ---------------------------------------------------------------------------


def _build_text_bands(
    lines: Sequence[TextLine],
    config: LayoutConfig,
    hrule_ys: Optional[List[float]] = None,
) -> List[TextBand]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda ln: (ln.rect.y0, ln.rect.x0))
    hrules = hrule_ys or []

    bands: List[TextBand] = []
    cur_rect: Optional[fitz.Rect] = None
    cur_words: List[RectTuple] = []
    cur_lines: List[TextLine] = []

    def flush_band() -> None:
        nonlocal cur_rect, cur_words, cur_lines
        if cur_rect is None:
            return
        bands.append(TextBand(rect=cur_rect, words=list(cur_words), lines=list(cur_lines)))
        cur_rect = None
        cur_words = []
        cur_lines = []

    def _hrule_between(y_top: float, y_bot: float) -> bool:
        return any(y_top < y < y_bot for y in hrules)

    for line in sorted_lines:
        if cur_rect is None:
            cur_rect = fitz.Rect(line.rect)
            cur_words = list(line.words)
            cur_lines = [line]
            continue

        y_gap = line.rect.y0 - cur_rect.y1
        crossed_rule = _hrule_between(cur_rect.y1, line.rect.y0)

        if y_gap > config.band_merge_gap or crossed_rule:
            flush_band()
            cur_rect = fitz.Rect(line.rect)
            cur_words = list(line.words)
            cur_lines = [line]
        else:
            cur_rect = cur_rect | line.rect
            cur_words.extend(line.words)
            cur_lines.append(line)
    flush_band()
    return bands


# ---------------------------------------------------------------------------
# Split bands at full-width -> two-column transitions
# ---------------------------------------------------------------------------


def _split_mixed_bands(
    bands: List[TextBand],
    content_bounds: Tuple[float, float],
    page_rect: fitz.Rect,
    page: fitz.Page,
    config: LayoutConfig,
) -> List[TextBand]:
    x0, x1 = content_bounds
    half = config.num_bins // 2
    result: List[TextBand] = []

    for band in bands:
        if len(band.lines) < 6:
            result.append(band)
            continue

        is_caption_start = _band_starts_with_caption(band, page, y_binning=True)

        if is_caption_start:
            first_text_line = min(
                (ln for ln in band.lines if ln.words),
                key=lambda ln: ln.rect.y0,
                default=None,
            )
            if first_text_line is not None:
                caption_width = first_text_line.rect.x1 - first_text_line.rect.x0
                content_width = x1 - x0
                if caption_width < content_width * 0.55:
                    is_caption_start = False

        sorted_preview = sorted(band.lines, key=lambda ln: ln.rect.y0)
        top_check = sorted_preview[: min(6, len(sorted_preview))]
        wordless_count = sum(1 for ln in top_check if not ln.words)
        if wordless_count > len(top_check) * 0.5:
            result.append(band)
            continue

        sorted_lines = sorted(band.lines, key=lambda ln: ln.rect.y0)

        page_mid_x = (x0 + x1) / 2.0
        tags: List[str] = []
        for line in sorted_lines:
            if not line.words:
                tags.append("empty")
                continue
            occ = _x_occupancy(line.words, x0, x1, config.num_bins)
            lo = float(np.mean(occ[:half])) if half else 0.0
            ro = float(np.mean(occ[half:])) if half else 0.0
            cov = float(np.mean(occ))
            line_mid = (line.rect.x0 + line.rect.x1) / 2.0

            left_only = lo >= 0.45 and ro < 0.15
            right_only = ro >= 0.45 and lo < 0.15

            if left_only:
                tags.append("left")
            elif right_only:
                tags.append("right")
            elif cov < 0.25:
                center_dist = abs(line_mid - page_mid_x)
                if center_dist < (x1 - x0) * 0.20:
                    tags.append("full")
                elif line_mid < page_mid_x:
                    tags.append("left")
                else:
                    tags.append("right")
            else:
                tags.append("full")

        WINDOW = 6
        MIN_EACH = 2
        split_idx: Optional[int] = None
        for i in range(len(tags) - WINDOW + 1):
            window = tags[i : i + WINDOW]
            n_left = window.count("left")
            n_right = window.count("right")
            if n_left >= MIN_EACH and n_right >= MIN_EACH:
                split_idx = i
                break

        min_split = 1 if is_caption_start else 3
        if split_idx is None or split_idx < min_split:
            result.append(band)
            continue

        full_before = sum(1 for t in tags[:split_idx] if t == "full")
        min_full = 1 if is_caption_start else 3
        if full_before < min_full:
            result.append(band)
            continue

        top_lines_candidate = sorted_lines[:split_idx]

        if not is_caption_start:
            top_words_all: List[RectTuple] = []
            for ln in top_lines_candidate:
                top_words_all.extend(ln.words)
            if top_words_all:
                top_x0 = min(w[0] for w in top_words_all)
                top_x1 = max(w[2] for w in top_words_all)
                top_span = top_x1 - top_x0
                content_span = x1 - x0
                if top_span < content_span * 0.60:
                    result.append(band)
                    continue

            top_y0 = min(ln.rect.y0 for ln in top_lines_candidate)
            top_y1 = max(ln.rect.y1 for ln in top_lines_candidate)
            top_height = top_y1 - top_y0
            band_height = band.rect.y1 - band.rect.y0
            if band_height > 0 and (top_height / band_height < 0.15 or top_height < 60):
                result.append(band)
                continue

        best_split = split_idx
        best_gap = 0.0
        search_range = min(4, split_idx)
        for offset in range(search_range + 1):
            candidate = split_idx - offset
            if candidate < 1:
                break
            gap = sorted_lines[candidate].rect.y0 - sorted_lines[candidate - 1].rect.y1
            if gap >= 8.0 and gap > best_gap:
                best_gap = gap
                best_split = candidate
                break
        if is_caption_start and best_split == split_idx:
            for cand in range(split_idx, 0, -1):
                g = sorted_lines[cand].rect.y0 - sorted_lines[cand - 1].rect.y1
                if g >= 2.0:
                    real_col = 0
                    real_total = 0
                    for mi in range(cand, split_idx):
                        ln = sorted_lines[mi]
                        if not ln.words or ln.rect.width < 50:
                            continue
                        real_total += 1
                        if tags[mi] in ("left", "right"):
                            real_col += 1
                    if real_total > 0 and real_col > real_total / 2:
                        best_split = cand
                        break

        split_idx = best_split

        top_lines = sorted_lines[:split_idx]
        bot_lines = sorted_lines[split_idx:]

        if not top_lines or not bot_lines:
            result.append(band)
            continue

        def make_band(lines: List[TextLine]) -> TextBand:
            r = lines[0].rect
            for ln in lines[1:]:
                r = r | ln.rect
            w: List[RectTuple] = []
            for ln in lines:
                w.extend(ln.words)
            return TextBand(rect=r, words=w, lines=lines)

        result.append(make_band(top_lines))

        bot_band = make_band(bot_lines)
        bot_is_caption = _band_starts_with_caption(bot_band, page, y_binning=True)
        if bot_is_caption and len(bot_lines) > 6:
            first_text_ln = min(
                (ln for ln in bot_lines if ln.words),
                key=lambda ln: ln.rect.y0,
                default=None,
            )
            caption_is_fullwidth = (
                first_text_ln is not None
                and (first_text_ln.rect.x1 - first_text_ln.rect.x0) >= (x1 - x0) * 0.55
            )
            if caption_is_fullwidth:
                bot_sorted = sorted(bot_lines, key=lambda ln: ln.rect.y0)
                bot_tags: List[str] = []
                for line in bot_sorted:
                    if not line.words:
                        bot_tags.append("empty")
                        continue
                    occ = _x_occupancy(line.words, x0, x1, config.num_bins)
                    lo = float(np.mean(occ[:half])) if half else 0.0
                    ro = float(np.mean(occ[half:])) if half else 0.0
                    cov = float(np.mean(occ))
                    line_mid = (line.rect.x0 + line.rect.x1) / 2.0
                    left_only = lo >= 0.45 and ro < 0.15
                    right_only = ro >= 0.45 and lo < 0.15
                    if left_only:
                        bot_tags.append("left")
                    elif right_only:
                        bot_tags.append("right")
                    elif cov < 0.25:
                        center_dist = abs(line_mid - page_mid_x)
                        if center_dist < (x1 - x0) * 0.20:
                            bot_tags.append("full")
                        elif line_mid < page_mid_x:
                            bot_tags.append("left")
                        else:
                            bot_tags.append("right")
                    else:
                        bot_tags.append("full")

                gap_split: Optional[int] = None
                for ii in range(1, len(bot_sorted)):
                    g = bot_sorted[ii].rect.y0 - bot_sorted[ii - 1].rect.y1
                    if g >= 8.0:
                        below = bot_tags[ii : ii + WINDOW]
                        if (
                            len(below) >= WINDOW
                            and below.count("left") >= MIN_EACH
                            and below.count("right") >= MIN_EACH
                        ):
                            gap_split = ii
                            break

                if gap_split is None:
                    for ii in range(len(bot_tags) - WINDOW + 1):
                        w = bot_tags[ii : ii + WINDOW]
                        if w.count("left") >= MIN_EACH and w.count("right") >= MIN_EACH:
                            gap_split = ii
                            break

                if gap_split is not None and gap_split >= 1:
                    cap_lines = bot_sorted[:gap_split]
                    body_lines = bot_sorted[gap_split:]
                    if cap_lines and body_lines:
                        result.append(make_band(cap_lines))
                        result.append(make_band(body_lines))
                        continue

        result.append(bot_band)

    return result


# ---------------------------------------------------------------------------
# Caption detection helpers
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(r"^\s*(?:Fig(?:ure)?|Tab(?:le)?)\b", re.IGNORECASE)


def _band_starts_with_caption(
    band: TextBand, page: fitz.Page, *, y_binning: bool = False
) -> bool:
    if not band.lines:
        return False
    first_line = min(band.lines, key=lambda ln: ln.rect.y0)
    clip = fitz.Rect(first_line.rect)
    clip.y1 = min(clip.y1, clip.y0 + 20)
    raw = page.get_text("words", clip=clip)
    if not raw:
        raw = page.get_text("words", clip=band.rect)
    if not raw:
        return False
    if y_binning:
        raw.sort(key=lambda w: (round(w[1] / 5.0) * 5, w[0]))
    else:
        raw.sort(key=lambda w: (w[1], w[0]))
    text = " ".join(str(w[4]) for w in raw[:6])
    return bool(_CAPTION_RE.search(text))


def _band_caption_label(band: TextBand, page: fitz.Page) -> Optional[str]:
    if not band.lines:
        return None
    first_line = min(band.lines, key=lambda ln: ln.rect.y0)
    clip = fitz.Rect(first_line.rect)
    clip.y1 = min(clip.y1, clip.y0 + 20)
    raw = page.get_text("words", clip=clip)
    if not raw:
        raw = page.get_text("words", clip=band.rect)
    if not raw:
        return None
    raw.sort(key=lambda w: (w[1], w[0]))
    text = " ".join(str(w[4]) for w in raw[:6]).strip().lower()
    if re.match(r"fig(ure)?\.?\s", text):
        return "figure"
    if re.match(r"tab(le)?\.?\s", text):
        return "table"
    return None


# ---------------------------------------------------------------------------
# Caption-aware band merging
# ---------------------------------------------------------------------------


def _merge_figure_caption_bands(
    bands: List[TextBand],
    page: fitz.Page,
    config: LayoutConfig,
    content_bounds: Tuple[float, float],
    page_rect: fitz.Rect,
) -> List[TextBand]:
    if len(bands) < 2:
        return bands

    def _combine(a: TextBand, b: TextBand) -> TextBand:
        return TextBand(
            rect=a.rect | b.rect,
            words=a.words + b.words,
            lines=a.lines + b.lines,
        )

    changed = True
    while changed:
        changed = False
        new_bands: List[TextBand] = []
        i = 0
        while i < len(bands):
            current = bands[i]
            if i + 1 < len(bands):
                nxt = bands[i + 1]
                y_gap = nxt.rect.y0 - current.rect.y1
                nxt_cls, _, _, _ = _classify_band(nxt, content_bounds, page_rect, config)
                if (
                    0 <= y_gap <= config.caption_merge_gap
                    and nxt_cls != "double"
                    and _band_starts_with_caption(nxt, page)
                ):
                    current = _combine(current, nxt)
                    i += 2
                    changed = True
                    new_bands.append(current)
                    continue
            new_bands.append(current)
            i += 1
        bands = new_bands
    return bands


# ---------------------------------------------------------------------------
# X-occupancy analysis
# ---------------------------------------------------------------------------


def _x_occupancy(words: Sequence[RectTuple], x0: float, x1: float, bins: int) -> np.ndarray:
    occupancy = np.zeros(bins, dtype=np.uint8)
    width = max(1e-6, x1 - x0)
    for wx0, _, wx1, _ in words:
        start = int(np.floor(((wx0 - x0) / width) * bins))
        end = int(np.ceil(((wx1 - x0) / width) * bins))
        start = max(0, min(bins - 1, start))
        end = max(1, min(bins, end))
        if end > start:
            occupancy[start:end] = 1
    return occupancy


def _find_center_gap(occupancy: np.ndarray) -> Tuple[int, int]:
    bins = len(occupancy)
    center = bins // 2
    zero_runs: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    for idx, value in enumerate(occupancy):
        if value == 0 and run_start is None:
            run_start = idx
        elif value == 1 and run_start is not None:
            zero_runs.append((run_start, idx))
            run_start = None
    if run_start is not None:
        zero_runs.append((run_start, bins))
    if not zero_runs:
        return center, center
    best = min(zero_runs, key=lambda run: abs(((run[0] + run[1]) // 2) - center))
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Band classification (single vs double column)
# ---------------------------------------------------------------------------


def _classify_band(
    band: TextBand,
    content_bounds: Tuple[float, float],
    page_rect: fitz.Rect,
    config: LayoutConfig,
) -> Tuple[str, float, Tuple[int, int], np.ndarray]:
    x0, x1 = content_bounds
    occupancy = _x_occupancy(band.words, x0, x1, config.num_bins)
    coverage = float(np.mean(occupancy))

    gap_start, gap_end = _find_center_gap(occupancy)
    min_band_height = config.min_band_height_ratio * page_rect.height

    left_only = 0
    right_only = 0
    mixed_full = 0
    eligible_lines = 0
    for line in band.lines:
        line_occ = _x_occupancy(line.words, x0, x1, config.num_bins)
        line_cov = float(np.mean(line_occ))
        if line_cov < 0.30:
            continue
        eligible_lines += 1
        ll = line_occ[: config.num_bins // 2]
        rr = line_occ[config.num_bins // 2 :]
        left_line_cov = float(np.mean(ll)) if len(ll) else 0.0
        right_line_cov = float(np.mean(rr)) if len(rr) else 0.0
        if left_line_cov >= 0.55 and right_line_cov <= 0.35:
            left_only += 1
        elif right_line_cov >= 0.55 and left_line_cov <= 0.35:
            right_only += 1
        elif left_line_cov >= 0.45 and right_line_cov >= 0.45:
            mixed_full += 1

    mixed_ratio = mixed_full / float(eligible_lines) if eligible_lines > 0 else 0.0
    left_only_ratio = left_only / float(eligible_lines) if eligible_lines > 0 else 0.0
    right_only_ratio = right_only / float(eligible_lines) if eligible_lines > 0 else 0.0

    band_mid = (x0 + x1) / 2.0
    width = max(1e-6, x1 - x0)
    mid_margin = width * 0.07
    left_center_count = 0
    right_center_count = 0
    for line in band.lines:
        line_mid = (line.rect.x0 + line.rect.x1) / 2.0
        if line_mid <= (band_mid - mid_margin):
            left_center_count += 1
        elif line_mid >= (band_mid + mid_margin):
            right_center_count += 1

    center_split_ratio = min(left_center_count, right_center_count) / float(
        max(1, eligible_lines)
    )

    is_double = (
        band.rect.height >= min_band_height
        and eligible_lines >= 4
        and (left_only + right_only) >= 4
        and left_only_ratio >= 0.16
        and right_only_ratio >= 0.08
        and mixed_ratio <= 0.78
        and left_center_count >= 2
        and right_center_count >= 2
        and center_split_ratio >= 0.18
    )
    return ("double" if is_double else "single", coverage, (gap_start, gap_end), occupancy)


# ---------------------------------------------------------------------------
# Splitting a double-column band into left / right rects
# ---------------------------------------------------------------------------


def _split_double_band(
    band: TextBand,
    content_bounds: Tuple[float, float],
    gap_bins: Tuple[int, int],
    page_rect: fitz.Rect,
    config: LayoutConfig,
) -> Tuple[fitz.Rect, fitz.Rect, float]:
    x0, x1 = content_bounds
    width = max(1e-6, x1 - x0)
    page_mid = x0 + (width / 2.0)
    left_words = [w for w in band.words if ((w[0] + w[2]) / 2.0) <= page_mid]
    right_words = [w for w in band.words if ((w[0] + w[2]) / 2.0) > page_mid]

    if left_words and right_words:
        left_max = float(np.percentile([w[2] for w in left_words], 95))
        right_min = float(np.percentile([w[0] for w in right_words], 5))
        if right_min > left_max:
            gutter_left = left_max
            gutter_right = right_min
        else:
            g0, g1 = gap_bins
            gutter_left = x0 + (g0 / config.num_bins) * width
            gutter_right = x0 + (g1 / config.num_bins) * width
    else:
        g0, g1 = gap_bins
        gutter_left = x0 + (g0 / config.num_bins) * width
        gutter_right = x0 + (g1 / config.num_bins) * width

    gutter_mid = (gutter_left + gutter_right) / 2.0

    if not left_words or not right_words:
        left_words = [w for w in band.words if ((w[0] + w[2]) / 2.0) <= page_mid]
        right_words = [w for w in band.words if ((w[0] + w[2]) / 2.0) > page_mid]

    def rect_from_words(
        words: Sequence[RectTuple], fallback_x0: float, fallback_x1: float
    ) -> fitz.Rect:
        if not words:
            return fitz.Rect(fallback_x0, band.rect.y0, fallback_x1, band.rect.y1)
        return fitz.Rect(
            min(w[0] for w in words),
            min(w[1] for w in words),
            max(w[2] for w in words),
            max(w[3] for w in words),
        )

    left_rect = rect_from_words(left_words, band.rect.x0, max(band.rect.x0, gutter_left))
    right_rect = rect_from_words(right_words, min(band.rect.x1, gutter_right), band.rect.x1)
    return left_rect, right_rect, gutter_mid


# ---------------------------------------------------------------------------
# Splitting single-column bands that contain distinct figures/tables
# ---------------------------------------------------------------------------


def _split_distinct_fullwidth_items(
    bands: List[TextBand],
    page: fitz.Page,
    content_bounds: Tuple[float, float] = (0, 0),
    page_rect: Optional[fitz.Rect] = None,
    config: Optional[LayoutConfig] = None,
) -> List[TextBand]:
    result: List[TextBand] = []
    for band in bands:
        if len(band.lines) < 2:
            result.append(band)
            continue

        if config is not None and page_rect is not None and content_bounds != (0, 0):
            cls, _, _, _ = _classify_band(band, content_bounds, page_rect, config)
            if cls == "double":
                result.append(band)
                continue

        sorted_lines = sorted(band.lines, key=lambda ln: ln.rect.y0)
        caption_indices: List[Tuple[int, str]] = []
        for idx, line in enumerate(sorted_lines):
            clip = fitz.Rect(line.rect)
            clip.y1 = min(clip.y1, clip.y0 + 20)
            raw = page.get_text("words", clip=clip)
            if not raw:
                continue
            raw.sort(key=lambda w: (w[1], w[0]))
            text = " ".join(str(w[4]) for w in raw[:6]).strip().lower()
            if re.match(r"fig(ure)?\.?\s", text):
                caption_indices.append((idx, "figure"))
            elif re.match(r"tab(le)?\.?\s", text):
                caption_indices.append((idx, "table"))

        if len(caption_indices) < 2:
            result.append(band)
            continue

        split_points = [ci[0] for ci in caption_indices[1:]]
        prev = 0
        for sp in split_points:
            chunk_lines = sorted_lines[prev:sp]
            if chunk_lines:
                r = chunk_lines[0].rect
                for ln in chunk_lines[1:]:
                    r = r | ln.rect
                words: List[RectTuple] = []
                for ln in chunk_lines:
                    words.extend(ln.words)
                result.append(TextBand(rect=r, words=words, lines=chunk_lines))
            prev = sp
        chunk_lines = sorted_lines[prev:]
        if chunk_lines:
            r = chunk_lines[0].rect
            for ln in chunk_lines[1:]:
                r = r | ln.rect
            words = []
            for ln in chunk_lines:
                words.extend(ln.words)
            result.append(TextBand(rect=r, words=words, lines=chunk_lines))

    return result


# ---------------------------------------------------------------------------
# Reading-order reordering
# ---------------------------------------------------------------------------


def _reorder_double_column_runs(segments: List[Segment]) -> List[Segment]:
    result: List[Segment] = []
    run_lefts: List[Segment] = []
    run_rights: List[Segment] = []

    def flush_run() -> None:
        result.extend(run_lefts)
        result.extend(run_rights)
        run_lefts.clear()
        run_rights.clear()

    i = 0
    while i < len(segments):
        seg = segments[i]
        if (
            seg.label == "doublecolumn_left"
            and i + 1 < len(segments)
            and segments[i + 1].label == "doublecolumn_right"
        ):
            run_lefts.append(seg)
            run_rights.append(segments[i + 1])
            i += 2
        else:
            flush_run()
            result.append(seg)
            i += 1
    flush_run()

    for idx, seg in enumerate(result):
        seg.order = idx
    return result


# ---------------------------------------------------------------------------
# Merge adjacent single-column segments
# ---------------------------------------------------------------------------


def _merge_adjacent_single_columns(
    segments: List[Segment], config: LayoutConfig
) -> List[Segment]:
    if len(segments) < 2:
        return segments
    result: List[Segment] = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        while (
            i + 1 < len(segments)
            and cur.label == "single_column"
            and segments[i + 1].label == "single_column"
            and cur.page_index == segments[i + 1].page_index
            and (segments[i + 1].rect.y0 - cur.rect.y1) < config.caption_merge_gap
        ):
            nxt = segments[i + 1]
            cur = Segment(
                page_index=cur.page_index,
                rect=cur.rect | nxt.rect,
                label="single_column",
                order=cur.order,
            )
            i += 1
        result.append(cur)
        i += 1
    for idx, seg in enumerate(result):
        seg.order = idx
    return result


# ---------------------------------------------------------------------------
# Resolve vertical overlaps between segments
# ---------------------------------------------------------------------------


def _resolve_vertical_overlaps(segments: List[Segment]) -> List[Segment]:
    if len(segments) < 2:
        return segments

    rows: List[List[int]] = []
    used: set[int] = set()
    for i, seg in enumerate(segments):
        if i in used:
            continue
        row = [i]
        used.add(i)
        for j in range(i + 1, len(segments)):
            if j in used:
                continue
            if seg.page_index != segments[j].page_index:
                continue
            overlap_y0 = max(seg.rect.y0, segments[j].rect.y0)
            overlap_y1 = min(seg.rect.y1, segments[j].rect.y1)
            if overlap_y1 - overlap_y0 > min(seg.rect.height, segments[j].rect.height) * 0.5:
                row.append(j)
                used.add(j)
        rows.append(row)

    page_rows: dict[int, List[List[int]]] = {}
    for row in rows:
        pi = segments[row[0]].page_index
        page_rows.setdefault(pi, []).append(row)

    for pi, p_rows in page_rows.items():
        p_rows.sort(key=lambda r: min(segments[idx].rect.y0 for idx in r))
        for ri in range(len(p_rows) - 1):
            top_row = p_rows[ri]
            bot_row = p_rows[ri + 1]
            top_y1 = max(segments[idx].rect.y1 for idx in top_row)
            bot_y0 = min(segments[idx].rect.y0 for idx in bot_row)
            if top_y1 > bot_y0:
                boundary = (top_y1 + bot_y0) / 2.0
                for idx in top_row:
                    s = segments[idx]
                    if s.rect.y1 > boundary:
                        s.rect = fitz.Rect(s.rect.x0, s.rect.y0, s.rect.x1, boundary)
                for idx in bot_row:
                    s = segments[idx]
                    if s.rect.y0 < boundary:
                        s.rect = fitz.Rect(s.rect.x0, boundary, s.rect.x1, s.rect.y1)

    return segments


def _absorb_column_images(
    segments: List[Segment],
    figure_regions: List[fitz.Rect],
    full_width_threshold: float,
    page_index: int,
    min_fig_w: float,
    min_fig_h: float,
    max_gap: float = 25.0,
) -> List[Segment]:
    col_figs = [
        r for r in figure_regions
        if r.width < full_width_threshold
        and r.width >= min_fig_w
        and r.height >= min_fig_h
    ]
    if not col_figs:
        return segments

    for fig in col_figs:
        fig_cx = (fig.x0 + fig.x1) / 2.0
        best_seg: Optional[Segment] = None
        best_dist = float("inf")

        for seg in segments:
            if seg.page_index != page_index:
                continue
            if fig_cx < seg.rect.x0 or fig_cx > seg.rect.x1:
                continue
            if fig.y1 <= seg.rect.y0:
                dist = seg.rect.y0 - fig.y1
            elif fig.y0 >= seg.rect.y1:
                dist = fig.y0 - seg.rect.y1
            else:
                dist = 0.0
            if dist < best_dist:
                best_dist = dist
                best_seg = seg

        if best_seg is not None and best_dist <= max_gap:
            best_seg.rect = fitz.Rect(
                best_seg.rect.x0,
                min(best_seg.rect.y0, fig.y0),
                best_seg.rect.x1,
                max(best_seg.rect.y1, fig.y1),
            )

    return segments


# ---------------------------------------------------------------------------
# Per-page segment extraction (main pipeline)
# ---------------------------------------------------------------------------


def _extract_segments_from_page(
    page: fitz.Page, page_index: int, config: LayoutConfig
) -> List[Segment]:
    lines = _extract_text_lines(page, config)
    page_rect = page.rect

    image_rects = _extract_image_rects(page)
    figure_regions = _cluster_image_rects(image_rects, merge_gap=30.0)
    min_fig_w = page_rect.width * 0.08
    min_fig_h = 15.0

    content_lines_pre = [ln for ln in lines if ln.words]
    if content_lines_pre:
        est_content_w = max(ln.rect.x1 for ln in content_lines_pre) - min(ln.rect.x0 for ln in content_lines_pre)
    else:
        est_content_w = page_rect.width * 0.8
    full_width_threshold = est_content_w * 0.55

    for region in figure_regions:
        if region.width >= min_fig_w and region.height >= min_fig_h:
            if region.width >= full_width_threshold:
                lines.append(TextLine(rect=region, words=[]))

    content_lines = [ln for ln in lines if ln.words]
    if content_lines:
        approx_width = max(ln.rect.x1 for ln in content_lines) - min(ln.rect.x0 for ln in content_lines)
    else:
        approx_width = page_rect.width * 0.8
    hrule_ys = _extract_horizontal_rules(page, approx_width, config)

    bands = _build_text_bands(lines, config, hrule_ys=hrule_ys)

    if not bands:
        return []

    min_x = min(b.rect.x0 for b in bands)
    max_x = max(b.rect.x1 for b in bands)
    content_bounds = (min_x, max_x)

    bands = _split_mixed_bands(bands, content_bounds, page_rect, page, config)
    bands = _merge_figure_caption_bands(bands, page, config, content_bounds, page_rect)
    bands = _split_distinct_fullwidth_items(bands, page, content_bounds, page_rect, config)

    segments: List[Segment] = []
    order = 0
    for band in sorted(bands, key=lambda b: b.rect.y0):
        cls, _, gap_bins, _ = _classify_band(band, content_bounds, page_rect, config)
        cb_x0, cb_x1 = content_bounds

        if cls == "single":
            full_rect = fitz.Rect(cb_x0, band.rect.y0, cb_x1, band.rect.y1)
            rect = _expand_rect(full_rect, config.segment_padding, page_rect)
            segments.append(Segment(page_index=page_index, rect=rect, label="single_column", order=order))
            order += 1
        else:
            left_rect, right_rect, gutter_mid = _split_double_band(band, content_bounds, gap_bins, page_rect, config)
            left_rect = fitz.Rect(cb_x0, left_rect.y0, gutter_mid, left_rect.y1)
            right_rect = fitz.Rect(gutter_mid, right_rect.y0, cb_x1, right_rect.y1)
            left_rect = _expand_rect(left_rect, config.segment_padding, page_rect)
            right_rect = _expand_rect(right_rect, config.segment_padding, page_rect)
            segments.append(Segment(page_index=page_index, rect=left_rect, label="doublecolumn_left", order=order))
            order += 1
            segments.append(Segment(page_index=page_index, rect=right_rect, label="doublecolumn_right", order=order))
            order += 1

    segments = _absorb_column_images(
        segments, figure_regions, full_width_threshold,
        page_index, min_fig_w, min_fig_h,
    )
    segments = _reorder_double_column_runs(segments)
    segments = _merge_adjacent_single_columns(segments, config)
    segments = _resolve_vertical_overlaps(segments)
    return segments


# ---------------------------------------------------------------------------
# Output: segmented PDF
# ---------------------------------------------------------------------------


def _write_segmented_pdf(
    src_doc: fitz.Document,
    segments: Sequence[Segment],
    output_pdf: str,
) -> None:
    out_doc = fitz.open()
    try:
        for segment in segments:
            clip = fitz.Rect(segment.rect)
            if clip.width <= 1 or clip.height <= 1:
                continue
            new_page = out_doc.new_page(width=clip.width, height=clip.height)
            target_rect = fitz.Rect(0, 0, clip.width, clip.height)
            new_page.show_pdf_page(
                target_rect,
                src_doc,
                segment.page_index,
                clip=clip,
                keep_proportion=False,
            )
        out_doc.save(output_pdf)
    finally:
        out_doc.close()


# ---------------------------------------------------------------------------
# Output: debug overlay images
# ---------------------------------------------------------------------------


def _render_debug_overlay(
    src_doc: fitz.Document,
    segments: Sequence[Segment],
    debug_dir: str,
) -> None:
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed; skipping debug overlays. Install with: pip install docberry[debug]")
        return

    os.makedirs(debug_dir, exist_ok=True)
    grouped: dict[int, List[Segment]] = {}
    for seg in segments:
        grouped.setdefault(seg.page_index, []).append(seg)

    for page_index, page_segments in sorted(grouped.items()):
        page = src_doc[page_index]
        scale = 150.0 / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        for seg in page_segments:
            if seg.label == "single_column":
                color = (0, 255, 0)
            elif seg.label == "doublecolumn_left":
                color = (255, 128, 0)
            else:
                color = (0, 128, 255)
            x0 = int(round(seg.rect.x0 * scale))
            y0 = int(round(seg.rect.y0 * scale))
            x1 = int(round(seg.rect.x1 * scale))
            y1 = int(round(seg.rect.y1 * scale))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            label = f"{seg.order}:{seg.label}"
            cv2.putText(img, label, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_path = os.path.join(debug_dir, f"page_{page_index:04d}_segments.png")
        cv2.imwrite(out_path, img)
        log.info("Saved debug overlay: %s", out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_pdf(
    input_pdf: str,
    output_pdf: str = "segmented_output.pdf",
    page_spec: Optional[str] = None,
    debug_dir: Optional[str] = None,
    config: Optional[LayoutConfig] = None,
) -> List[Segment]:
    """Segment a PDF into reading-order regions.

    Detects full-width regions (title, figures, tables) and two-column
    body text, then writes each region as a separate page in a new PDF
    preserving the natural reading order.

    Args:
        input_pdf: Path to the source PDF file.
        output_pdf: Path for the segmented output PDF.
        page_spec: Optional page range (0-based), e.g. ``"0,2-4"``.
        debug_dir: If provided, save debug overlay images here.
        config: Layout tuning parameters. Uses defaults if ``None``.

    Returns:
        List of :class:`Segment` objects describing each region.
    """
    config = config or LayoutConfig()
    src_doc = fitz.open(input_pdf)
    try:
        page_indexes = _parse_page_spec(page_spec, len(src_doc))
        all_segments: List[Segment] = []
        for page_index in page_indexes:
            page = src_doc[page_index]
            page_segments = _extract_segments_from_page(page, page_index, config)
            all_segments.extend(page_segments)
        _write_segmented_pdf(src_doc, all_segments, output_pdf)
        if debug_dir:
            _render_debug_overlay(src_doc, all_segments, debug_dir)
    finally:
        src_doc.close()
    return all_segments
