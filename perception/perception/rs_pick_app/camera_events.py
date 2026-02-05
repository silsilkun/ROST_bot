import cv2
import numpy as np


# =========================
# Click params (이 파일 상단)
# =========================
CLICK_R = 2
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 2.00


class CameraEvents:
    def __init__(self):
        self.clicked_uv = []

    def _depth_median_around(self, u: int, v: int, depth_m: np.ndarray):
        if depth_m is None:
            return None
        H, W = depth_m.shape[:2]
        u = int(np.clip(u, 0, W - 1))
        v = int(np.clip(v, 0, H - 1))

        vals = []
        for dv in range(-CLICK_R, CLICK_R + 1):
            for du in range(-CLICK_R, CLICK_R + 1):
                uu = u + du
                vv = v + dv
                if 0 <= uu < W and 0 <= vv < H:
                    d = float(depth_m[vv, uu])
                    if d > 0.0 and (DEPTH_MIN_M <= d <= DEPTH_MAX_M):
                        vals.append(d)
        if not vals:
            return None
        return float(np.median(vals))

    def on_mouse(self, event, x, y, latest_depth_m):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        u, v = int(x), int(y)
        self.clicked_uv.append((u, v))

        d_med = self._depth_median_around(u, v, latest_depth_m)
        if d_med is None:
            print(f"Click: x={u}, y={v}, depth=N/A")
        else:
            print(f"Click: x={u}, y={v}, depth={d_med * 1000.0:.1f} mm")

    def on_key(self, key: int) -> dict:
        if key in (27, ord("q")):
            return {"quit": True, "reset": False, "do_space": False}

        if key == ord("r"):
            self.clicked_uv.clear()
            return {"quit": False, "reset": True, "do_space": False}

        if key == ord(" "):
            return {"quit": False, "reset": False, "do_space": True}

        return {"quit": False, "reset": False, "do_space": False}
