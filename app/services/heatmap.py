from __future__ import annotations

import cv2
import numpy as np


class CrowdHeatmap:
    """
    Crowd heatmap: MOG2 + frame-diff for full human coverage,
    filtered by person-sized contours and weighted by detection proximity
    so heat stays on people and doesn't bleed onto objects or empty space.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int],
        decay: float = 0.55,
        blur_ksize: int = 45,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_JET,
    ) -> None:
        h, w = frame_shape
        self._h = h
        self._w = w
        self._decay = float(np.clip(decay, 0.05, 0.99))
        self._blur = blur_ksize | 1
        self._alpha = float(np.clip(alpha, 0.05, 0.95))
        self._colormap = colormap
        self._heat = np.zeros((h, w), dtype=np.float32)
        self._current_presence = np.zeros((h, w), dtype=np.float32)
        self._prev_gray: np.ndarray | None = None
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=22, detectShadows=False
        )
        self._min_blob_area = max(int(h * w * 0.0008), 300)

    # ------------------------------------------------------------------

    def _get_foreground(self, gray: np.ndarray) -> np.ndarray:
        """MOG2 + frame-diff, then keep only person-sized blobs."""
        mog = self._bg.apply(gray)

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self._prev_gray, gray)
            _, diff_bin = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            raw = cv2.bitwise_or(mog, diff_bin)
        else:
            raw = mog

        k_open  = np.ones((5,  5), np.uint8)
        k_close = np.ones((11, 11), np.uint8)
        fg = cv2.morphologyEx(raw,  cv2.MORPH_OPEN,  k_open,  iterations=1)
        fg = cv2.morphologyEx(fg,   cv2.MORPH_CLOSE, k_close, iterations=2)
        fg = cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg_clean = np.zeros_like(fg)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self._min_blob_area:
                cv2.drawContours(fg_clean, [cnt], -1, 255, -1)

        return fg_clean

    def _influence_map(
        self,
        h: int,
        w: int,
        detections: list[tuple[int, int, int, int, float]],
    ) -> np.ndarray:
        inf_map = np.zeros((h, w), dtype=np.float32)
        for bx, by, bw, bh, _ in detections:
            cx = int(bx + bw * 0.5)
            cy = int(by + bh * 0.65)
            r = max(int(np.hypot(bw, bh) * 1.8), 70)
            cv2.circle(inf_map, (cx, cy), r, 1.0, -1)

        inf_map = cv2.GaussianBlur(inf_map, (61, 61), 0)
        imax = float(inf_map.max())
        if imax > 1e-6:
            inf_map /= imax
        return inf_map

    def _detection_peaks(
        self,
        h: int,
        w: int,
        detections: list[tuple[int, int, int, int, float]],
    ) -> np.ndarray:
        """Gaussian blob at each confirmed person position."""
        peaks = np.zeros((h, w), dtype=np.float32)
        if not detections:
            return peaks

        avg_diag = float(np.mean([np.hypot(bw, bh) for bx, by, bw, bh, _ in detections]))
        sigma = int(np.clip(avg_diag * 0.28, 12, 40))
        ks = min((sigma * 4 + 1) | 1, 101)

        for bx, by, bw, bh, conf in detections:
            cx = int(bx + bw * 0.5)
            cy = int(by + bh * 0.65)
            blob = np.zeros((h, w), dtype=np.float32)
            cv2.circle(blob, (cx, cy), max(sigma // 3, 4), 1.0, -1)
            blob = cv2.GaussianBlur(blob, (ks, ks), sigma)
            bmax = float(blob.max())
            if bmax > 1e-6:
                peaks = np.maximum(peaks, blob * (float(np.clip(conf, 0.7, 1.0)) / bmax))

        return peaks

    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int, float]],
    ) -> None:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = self._get_foreground(gray)
        self._prev_gray = gray

        # Rebuild presence from scratch every frame — no stale data carries forward
        self._current_presence = np.zeros((h, w), dtype=np.float32)

        if detections:
            fg_float = fg.astype(np.float32) / 255.0
            influence = self._influence_map(h, w, detections)
            fg_weighted = fg_float * influence
            peaks = self._detection_peaks(h, w, detections)
            self._current_presence = np.maximum(fg_weighted * 0.85, peaks)

        # heat * decay + current * (1 - decay):
        # pixels with zero presence shrink by (1-decay) every frame and
        # reach <1% in ~log(0.01)/log(decay) frames — no ghost possible.
        self._heat = self._heat * self._decay + self._current_presence * (1.0 - self._decay)
        self._heat = np.clip(self._heat, 0.0, 1.0)

    def get_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        heat = self._heat
        if heat.shape != (h, w):
            heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

        blurred = cv2.GaussianBlur(heat, (self._blur, self._blur), 0)

        if not np.any(blurred > 0):
            return frame.copy()

        p98 = float(np.percentile(blurred, 98))
        norm = np.clip(blurred / max(p98, 1e-6), 0.0, 1.0)

        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), self._colormap)

        # Raised from 0.10 → 0.20 so faint decayed pixels don't paint through
        mask = (norm > 0.20).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        mask3 = np.dstack([mask, mask, mask])

        return (
            frame.astype(np.float32) * (1.0 - mask3 * self._alpha)
            + colored.astype(np.float32) * mask3 * self._alpha
        ).astype(np.uint8)

    def estimate_count(self, area_sq_m: float, max_density: float = 4.0) -> int:
        """Estimate people count from heatmap occupancy.

        Computes the fraction of the monitored area with meaningful crowd
        heat, then scales by max_density (people/m² assumed at full
        saturation).  Default 4.0 p/m² corresponds to a dense standing crowd.
        Override via env HEATMAP_MAX_DENSITY.
        """
        blurred = cv2.GaussianBlur(self._heat, (self._blur, self._blur), 0)
        occ_fraction = float(np.mean(blurred > 0.15))
        estimate = occ_fraction * area_sq_m * max_density
        return int(round(max(0.0, estimate)))

    def get_density_grid(self, rows: int = 4, cols: int = 6) -> np.ndarray:
        blurred = cv2.GaussianBlur(self._heat, (self._blur, self._blur), 0)
        grid = np.zeros((rows, cols), dtype=np.float32)
        cell_h = self._h // rows
        cell_w = self._w // cols
        for r in range(rows):
            for c in range(cols):
                cell = blurred[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                if cell.size > 0:
                    grid[r, c] = float(np.clip(np.mean(cell), 0.0, 1.0))
        return grid

    def reset(self) -> None:
        self._heat[:] = 0.0
        self._current_presence[:] = 0.0
        self._prev_gray = None
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=22, detectShadows=False
        )