# coordinate.py  (drop-in replacement, simplified)
import os
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(BASE_DIR, "camcalib.npz")

# --- constants (keep same behavior) ---
R = 2  # depth sample radius: 5x5
M_TO_CM = 100.0
FLIP_XYZ = (-1.0, -1.0, -1.0)
OFFSET_CM = (81.5, 15.9, 0.0)

# --- module-level cache (load once per process) ---
_CALIB = None  # {"T":..., "K":..., "D":...}


def _load_calib(path: str):
    global _CALIB
    if _CALIB is not None:
        return _CALIB

    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' 파일이 없습니다.")

    data = np.load(path)
    _CALIB = {
        "T": data["T_cam_to_work"].astype(np.float64),
        "K": data["camera_matrix"].astype(np.float64),
        "D": data["dist_coeffs"].astype(np.float64),
    }
    return _CALIB


class Coordinate:
    """
    Pixel(u,v) + depth_frame.get_distance(x,y)->meters  => World Pw (cm), shape (4,)
    반환: [X, Y, Z, 1]
    """

    def __init__(self, calib_path: str = SAVE_FILE):
        calib = _load_calib(calib_path)
        self.T_cam_to_work = calib["T"]
        self.camera_matrix = calib["K"]
        self.dist_coeffs = calib["D"]

    def pixel_to_world(self, u: int, v: int, depth_frame):
        u = int(u)
        v = int(v)

        # 1) depth median in 5x5 (meters -> cm)
        depths = []
        for du in range(-R, R + 1):
            for dv in range(-R, R + 1):
                d = float(depth_frame.get_distance(u + du, v + dv))
                if d > 0.0:
                    depths.append(d)
        if not depths:
            return None
        Z = float(np.median(depths)) * M_TO_CM  # cm

        # 2) intrinsics
        K = self.camera_matrix
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # 3) undistort pixel (keep same approach)
        pts = np.array([[[u, v]]], dtype=np.float32)
        undist = cv2.undistortPoints(pts, K, self.dist_coeffs, P=K)
        u_corr, v_corr = undist[0, 0]
        u_corr = float(u_corr)
        v_corr = float(v_corr)

        # 4) pixel -> camera (KEEP original axis mapping)
        Yc = (u_corr - cx) * Z / fx
        Xc = (v_corr - cy) * Z / fy
        Pc = np.array([Xc, Yc, Z, 1.0], dtype=np.float64)

        # 5) camera -> work
        Pw = self.T_cam_to_work @ Pc

        # 6) "real environment correction" (same behavior, but named)
        Pw[0] = FLIP_XYZ[0] * Pw[0] + OFFSET_CM[0]
        Pw[1] = FLIP_XYZ[1] * Pw[1] + OFFSET_CM[1]
        Pw[2] = FLIP_XYZ[2] * Pw[2] + OFFSET_CM[2]

        return Pw
