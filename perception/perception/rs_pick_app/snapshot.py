import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


# =========================
# Snapshot params
# =========================
SNAP_N = 3
MIN_FRAMES_OK = 2


@dataclass
class Snapshot:
    color_bgr: np.ndarray
    depth_snap_m: np.ndarray


def snapshot_median_depth(get_frame: Callable[[], Optional[tuple]]) -> Optional[Snapshot]:
    depths = []
    snap_color = None

    for _ in range(SNAP_N):
        out = get_frame()
        if out is None:
            continue
        color, depth_m = out
        if snap_color is None:
            snap_color = color.copy()
        depths.append(depth_m)

    if snap_color is None or len(depths) < MIN_FRAMES_OK:
        return None

    depth_snap_m = np.median(np.stack(depths, axis=0), axis=0).astype(np.float32)
    return Snapshot(color_bgr=snap_color, depth_snap_m=depth_snap_m)
