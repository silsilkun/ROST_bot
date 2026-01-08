import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "camcalib_dir")
OUT_PATH = os.path.join(BASE_DIR, "camcalib.npz")

T = np.load(os.path.join(CALIB_DIR, "T_cam_to_work.npy"))
K = np.load(os.path.join(CALIB_DIR, "camera_matrix.npy"))
D = np.load(os.path.join(CALIB_DIR, "dist_coeffs.npy"))

np.savez(
    OUT_PATH,
    T_cam_to_work=T,
    camera_matrix=K,
    dist_coeffs=D
)

print("✅ camcalib.npz 생성 완료:", OUT_PATH)
