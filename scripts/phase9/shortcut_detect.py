"""
Phase 9 / shortcut detection module.

Per-artifact detection rules with documented parameter choices and
trade-offs. Each detector returns (present: bool, mask: np.ndarray,
score: float).

Detectors (pre-registered in `docs/phase9/04_protocol.md` §2):
  - vignette        (luminance ring threshold)
  - ruler           (Canny + Hough lines)
  - hair            (black-tophat morphology) — negative control
  - color_cast      (per-collection histogram membership) — global

Ink is *not* automated here; it is manually annotated per protocol §2.3.

All detectors operate on a 384x384 BGR image (the model's input
geometry — choosing the same scale guarantees that masks line up
exactly with Grad-CAM activation maps).

The masks are uint8, 0/255. A 0.0–1.0 score is also returned for
strength ranking when needed.
"""

from __future__ import annotations

import numpy as np
import cv2

IMG_SIZE = 384


# ---------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------

def detect_vignette(
    img_bgr: np.ndarray,
    ring_frac: float = 0.15,
    ring_lum_max: float = 0.20,
    center_lum_min: float = 0.40,
) -> tuple[bool, np.ndarray, float]:
    """
    Detect dermoscope vignette: a dark band on the image periphery.

    Parameter rationale:
      ring_frac=0.15: vignette is typically the outer 10–20% of the
                     image radius (varies by dermoscope make). 15% is
                     conservative-inclusive.
      ring_lum_max=0.20 / center_lum_min=0.40: two-sided rule.
                     A dark image overall (e.g., pigmented lesion close-up
                     against poor lighting) would trip a single-sided
                     threshold; requiring center > 0.40 prevents that
                     false positive.

    Mask: the entire ring region (regardless of detection outcome). The
    detection flag indicates whether to *use* the mask.

    Score: (center_lum - ring_lum) / center_lum — higher means stronger
    contrast between ring darkness and center brightness, i.e., more
    pronounced vignette.
    """
    h, w = img_bgr.shape[:2]
    cy, cx = h / 2, w / 2
    max_r = min(cx, cy)

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring_mask = (r >= max_r * (1 - ring_frac)).astype(np.uint8) * 255
    center_mask = (r < max_r * 0.50)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ring_lum = float(gray[ring_mask > 0].mean()) if (ring_mask > 0).any() else 1.0
    center_lum = float(gray[center_mask].mean()) if center_mask.any() else 0.0

    present = (ring_lum < ring_lum_max) and (center_lum > center_lum_min)
    score = max(0.0, (center_lum - ring_lum) / max(center_lum, 1e-6))

    return present, ring_mask, score


# ---------------------------------------------------------------------
# Ruler / surgical-marker line
# ---------------------------------------------------------------------

def detect_ruler(
    img_bgr: np.ndarray,
    canny_lo: int = 80,
    canny_hi: int = 200,
    hough_min_len: int = 100,
    hough_max_gap: int = 5,
    hough_threshold: int = 80,
    aspect_ratio_min: float = 1.5,
) -> tuple[bool, np.ndarray, float]:
    """
    Detect ruler / surgical markings via Canny + probabilistic Hough +
    aspect-ratio gate.

    Parameter rationale (post-sensitivity-sweep — see
    artifacts/phase9/03b_ruler_sensitivity.log):
      The first-pass {canny=50/150, min_len=60, threshold=50} produced
      42% positive rate, dominated by lesion-boundary false positives.
      The strict-1 configuration retained at c=70: 24% (literature
      anchor — Winkler 2019 reports rulers concentrated in SIIM-2020-
      style cohorts) while reducing c=212 / c=249 to single-digit
      percentages consistent with HAM10000 and BCN20000 lacking
      systematic ruler presence (Bissoto 2022).
      The aspect-ratio gate (>=1.5) rejects diagonal short edges that
      slip past the length threshold; true ruler lines are dominantly
      horizontal or vertical.

    Score: number of *real* lines (post-aspect-gate) normalized via
    tanh(n / 5).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_len,
        maxLineGap=hough_max_gap,
    )

    mask = np.zeros(gray.shape, dtype=np.uint8)
    n_real = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if max(dx, dy) >= aspect_ratio_min * min(dx, dy) \
                    and max(dx, dy) >= hough_min_len:
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=5)
                n_real += 1

    present = n_real >= 1
    score = float(np.tanh(n_real / 5.0))
    return present, mask, score


# ---------------------------------------------------------------------
# Hair (negative control)
# ---------------------------------------------------------------------

def detect_hair(
    img_bgr: np.ndarray,
    kernel_size: int = 11,
    bin_threshold: int = 10,
    min_components: int = 3,
    min_component_len: int = 30,
) -> tuple[bool, np.ndarray, float]:
    """
    Detect hair via black-tophat morphology (dark thin structures).

    Black-tophat = closing(img) - img. It highlights *dark structures
    thinner than the kernel*. For body hair on skin, a 11x11 disk
    kernel hits the typical hair-strand width range (4-8 px in 384x384
    dermoscopy).

    Parameter rationale:
      kernel_size=11: hair strands are 4–8 px wide; the kernel must
                       be larger than that to fully enclose them so
                       the tophat returns the strand itself.
      bin_threshold=10 (on 0-255 grayscale tophat): below this, signal
                       is mostly noise.
      min_components=3, min_component_len=30: hair appears as multiple
                       elongated components; isolated tophat blobs are
                       artifacts. The lengthy-component rule is a
                       strong false-positive filter.

    Score: fraction of pixels above bin_threshold in the tophat
    response, normalized to [0, 1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    bin_mask = (tophat > bin_threshold).astype(np.uint8) * 255

    # Component analysis on the binary tophat
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask)
    # stats[i] = [x, y, w, h, area]; require length max(w, h) > min_component_len
    n_long = sum(
        1 for i in range(1, n_components)  # skip background
        if max(stats[i, 2], stats[i, 3]) >= min_component_len
    )

    present = n_long >= min_components
    score = float((bin_mask > 0).sum()) / bin_mask.size
    return present, bin_mask, score


# ---------------------------------------------------------------------
# Color cast (global, per-collection)
# ---------------------------------------------------------------------

def compute_color_histogram(
    img_bgr: np.ndarray,
    bins: int = 8,
) -> np.ndarray:
    """
    Joint 3D histogram in HSV space, flattened. Used as the per-image
    color signature for color-cast scoring.

    bins=8 per channel → 8^3 = 512-bin histogram. 384x384 image has
    ~147K pixels so each bin has ~290 average occupancy — well-conditioned.

    HSV over BGR because hue/saturation differences are more meaningful
    than RGB-channel correlations for dermatology image-collection
    discrimination (the differences we care about are lighting cast +
    pigment-tone shift, which separate cleanly in HSV).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-8)  # normalize to a probability mass
    return hist


def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Histogram-intersection similarity in [0, 1]. 1 = identical, 0 = no
    overlap. Used as the per-image membership strength score against a
    target collection's mean histogram.
    """
    return float(np.minimum(h1, h2).sum())


# ---------------------------------------------------------------------
# Image loading helper (keeps the model's input geometry)
# ---------------------------------------------------------------------

def load_image_resized(path, size: int = IMG_SIZE) -> np.ndarray:
    """Load an image as BGR, resize to (size, size), uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img
