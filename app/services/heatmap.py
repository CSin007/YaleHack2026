from __future__ import annotations

import cv2
import numpy as np


class CrowdHeatmap:
    """
    Builds a real-time crowd density heatmap using:
    - HOG detection boxes  → hard presence signal
    - Frame differencing   → motion presence signal
    - Gaussian blur        → spatial smoothing
    - Temporal decay       → heat fades when people leave

    Call update() each frame, get_overlay() returns a BGR frame
    with the heatmap blended on top.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],   # (height, width)
        decay:       float = 0.85,      # how fast heat fades (0=instant, 1=never)
        blur_ksize:  int   = 31,        # gaussian blur kernel — bigger = smoother zones
        alpha:       float = 0.42,      # heatmap overlay opacity
        colormap:    int   = cv2.COLORMAP_TURBO,
    ) -> None:
        h, w = frame_shape
        self._h        = h
        self._w        = w
        self._decay    = decay
        self._blur     = blur_ksize | 1     # must be odd
        self._alpha    = alpha
        self._colormap = colormap

        # accumulated heat map — float32, values 0..1
        self._heat = np.zeros((h, w), dtype=np.float32)
        self._prev_gray: np.ndarray | None = None

    # ── public ────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int, float]],
    ) -> None:
        """
        Feed the current BGR frame + HOG detections.
        detections: list of (x, y, w, h, confidence) in pixel coords.
        """
        h, w = frame.shape[:2]
        new_heat = np.zeros((h, w), dtype=np.float32)

        # ── signal 1: HOG bounding boxes ─────────────────────────────
        for (bx, by, bw, bh, conf) in detections:
            # concentrate heat in the lower-centre of the box (torso/feet)
            cx  = bx + bw // 2
            cy  = by + int(bh * 0.7)
            rad = max(bw, bh) // 2

            # draw a filled ellipse — more realistic than a rectangle
            cv2.ellipse(
                new_heat,
                (cx, cy),
                (max(rad, 10), max(rad // 2, 8)),
                0, 0, 360,
                float(conf),
                thickness=-1,
            )

        # ── signal 2: frame differencing (catches missed detections) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self._prev_gray, gray).astype(np.float32) / 255.0
            # only keep significant motion
            diff[diff < 0.08] = 0
            new_heat = np.maximum(new_heat, diff * 0.6)
        self._prev_gray = gray

        # ── accumulate with temporal decay ────────────────────────────
        self._heat = self._heat * self._decay + new_heat * (1.0 - self._decay * 0.5)
        self._heat = np.clip(self._heat, 0.0, 1.0)

    def get_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with heatmap blended on top."""
        h, w = frame.shape[:2]

        # resize heat to match frame if needed
        heat = self._heat
        if heat.shape != (h, w):
            heat = cv2.resize(heat, (w, h))

        # ── smooth ────────────────────────────────────────────────────
        blurred = cv2.GaussianBlur(heat, (self._blur, self._blur), 0)

        # ── normalise to 0-255 ────────────────────────────────────────
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        # ── apply colormap ────────────────────────────────────────────
        colored = cv2.applyColorMap(norm, self._colormap)

        # ── mask: only show heatmap where there is actually heat ──────
        mask = (blurred > 0.03).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask3 = np.stack([mask, mask, mask], axis=-1)

        # ── blend ─────────────────────────────────────────────────────
        overlay = (
            frame.astype(np.float32) * (1 - mask3 * self._alpha)
            + colored.astype(np.float32) * mask3 * self._alpha
        ).astype(np.uint8)

        return overlay

    def get_density_grid(self, rows: int = 4, cols: int = 6) -> np.ndarray:
        """
        Returns a (rows x cols) float array of zone-level densities (0..1).
        Useful for feeding into the risk predictor or sending to the frontend.
        """
        blurred = cv2.GaussianBlur(self._heat, (self._blur, self._blur), 0)
        grid    = np.zeros((rows, cols), dtype=np.float32)
        cell_h  = self._h // rows
        cell_w  = self._w // cols

        for r in range(rows):
            for c in range(cols):
                cell = blurred[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                grid[r, c] = float(np.mean(cell))

        return grid

    def reset(self) -> None:
        self._heat[:] = 0
        self._prev_gray = None