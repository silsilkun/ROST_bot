import os
import numpy as np
import cv2


# =========================
# Coordinate params
# =========================
CALIB_NPZ = "camcalib.npz"
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 2.00

M_TO_CM = 100.0
DEPTH_SAMPLE_R = 2

_CALIB_CACHE = None


def _load_calib():
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE

    base = os.path.dirname(os.path.abspath(__file__))
    path = CALIB_NPZ
    if not os.path.isabs(path):
        path = os.path.join(base, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"캘리브 파일 없음: {path}")

    data = np.load(path)
    _CALIB_CACHE = {
        "T": data["T_cam_to_work"].astype(np.float64),
        "K": data["camera_matrix"].astype(np.float64),
        "D": data["dist_coeffs"].astype(np.float64),
    }
    return _CALIB_CACHE


class Coordinate:
    def __init__(self):
        c = _load_calib()
        self.T = c["T"]
        self.K = c["K"]
        self.D = c["D"]

    def pixel_to_world_from_depthmap(self, u: int, v: int, depth_snap_m: np.ndarray):
        H, W = depth_snap_m.shape[:2]
        u = int(np.clip(u, 0, W - 1))
        v = int(np.clip(v, 0, H - 1))

        depths = []
        for du in range(-DEPTH_SAMPLE_R, DEPTH_SAMPLE_R + 1):
            for dv in range(-DEPTH_SAMPLE_R, DEPTH_SAMPLE_R + 1):
                uu = u + du
                vv = v + dv
                if 0 <= uu < W and 0 <= vv < H:
                    d = float(depth_snap_m[vv, uu])
                    if d > 0.0 and (DEPTH_MIN_M <= d <= DEPTH_MAX_M):
                        depths.append(d)

        if not depths:
            return None

        Z_cm = float(np.median(depths)) * M_TO_CM

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        pts = np.array([[[u, v]]], dtype=np.float32)
        und = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        uc, vc = float(und[0, 0, 0]), float(und[0, 0, 1])

        # v9 규약: u->Yc, v->Xc
        Yc = (uc - cx) * Z_cm / fx
        Xc = (vc - cy) * Z_cm / fy
        Pc = np.array([Xc, Yc, Z_cm, 1.0], dtype=np.float64)

        Pw = self.T @ Pc

        Pw[0] = -1 * Pw[0] + 81.5
        Pw[1] = -1 * Pw[1] + 15.9
        Pw[2] = -1 * Pw[2] + 0.0
        return Pw