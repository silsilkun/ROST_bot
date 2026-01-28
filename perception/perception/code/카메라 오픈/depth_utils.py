# depth_utils.py
import numpy as np
from coordinate import Coordinate


class FakeDepthFrameFromNpy:
    """depth_z16(mm) numpy -> get_distance(x,y)->meters"""
    __slots__ = ("depth", "h", "w")

    def __init__(self, depth_z16: np.ndarray):
        self.depth = depth_z16
        self.h, self.w = depth_z16.shape[:2]

    def get_distance(self, x: int, y: int) -> float:
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return 0.0
        d_mm = float(self.depth[int(y), int(x)])
        return (d_mm / 1000.0) if d_mm > 0.0 else 0.0


def box_center_pixel(box: np.ndarray) -> tuple[int, int]:
    """4-point polygon center pixel"""
    c = np.mean(box, axis=0)
    return int(round(float(c[0]))), int(round(float(c[1])))


def _patch_valid_count(depth_z16: np.ndarray, u: int, v: int, k: int = 2) -> int:
    h, w = depth_z16.shape[:2]
    u0, u1 = max(0, u - k), min(w, u + k + 1)
    v0, v1 = max(0, v - k), min(h, v + k + 1)
    return int(np.count_nonzero(depth_z16[v0:v1, u0:u1] > 0))


def _find_nearest_valid_pixel_in_rect(
    depth_z16: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    step: int = 2,
) -> tuple[int, int] | None:
    """
    Rect 내부에서 중심→바깥 방향으로 depth>0 픽셀 탐색
    (후보가 여러 개면 patch 유효 depth가 많은 픽셀 우선)
    """
    if depth_z16 is None or depth_z16.ndim != 2:
        return None

    h, w = depth_z16.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    step = max(1, int(step))

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))
    if depth_z16[cy, cx] > 0:
        return (cx, cy)

    max_r = int(np.hypot((x2 - x1) / 2, (y2 - y1) / 2)) + 2

    for r in range(step, max_r + 1, step):
        top, bot = cy - r, cy + r
        left, right = cx - r, cx + r

        candidates = []

        # 상/하 변
        if y1 <= top <= y2:
            for u in range(left, right + 1, step):
                if x1 <= u <= x2 and 0 <= u < w and 0 <= top < h and depth_z16[top, u] > 0:
                    candidates.append((u, top))
        if y1 <= bot <= y2:
            for u in range(left, right + 1, step):
                if x1 <= u <= x2 and 0 <= u < w and 0 <= bot < h and depth_z16[bot, u] > 0:
                    candidates.append((u, bot))

        # 좌/우 변
        for v in range(top + step, bot - step + 1, step):
            if not (y1 <= v <= y2 and 0 <= v < h):
                continue
            if x1 <= left <= x2 and 0 <= left < w and depth_z16[v, left] > 0:
                candidates.append((left, v))
            if x1 <= right <= x2 and 0 <= right < w and depth_z16[v, right] > 0:
                candidates.append((right, v))

        if not candidates:
            continue

        best, best_score = None, -1
        for (u, v) in candidates:
            score = _patch_valid_count(depth_z16, u, v, k=2)
            if score > best_score:
                best_score = score
                best = (u, v)

        if best is not None:
            return best

    return None


def blue_rect_to_world_safe(
    rect: tuple[int, int, int, int],
    depth_z16: np.ndarray,
    coord: Coordinate,
    depth_src: FakeDepthFrameFromNpy,
    search_step: int = 2,
) -> np.ndarray | None:
    """
    rect 중심 → 실패 시 rect 내부에서 유효 depth 픽셀 탐색 후 world 변환
    """
    if depth_z16 is None:
        return None

    x1, y1, x2, y2 = rect

    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))

    Pw = coord.pixel_to_world(cx, cy, depth_src)
    if Pw is not None:
        return Pw

    uv = _find_nearest_valid_pixel_in_rect(
        depth_z16, x1, y1, x2, y2, step=search_step
    )
    if uv is None:
        return None

    u, v = uv
    return coord.pixel_to_world(u, v, depth_src)
