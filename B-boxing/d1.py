import cv2
import json
import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass

COLOR_PATH = "color.png"
DEPTH_PATH = "depth.npy"
CALIB_PATH = "camcalib.npz"

@dataclass
class DetectedObject:
    def __init__(self):
        self.color = cv2.imread(COLOR_PATH)
        self.depth = np.load(DEPTH_PATH)
        calib = np.load(CALIB_PATH)
        
        if self.color is None or self.depth is None:
            raise RuntimeError("❌ color / depth 로드 실패")
        
        self.camera_matrix = calib["camera_matrix"]
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        
        self.depth_scale = 0.001 # mm → m
        self.roi = (460, 150, 830, 490)  # ROI (u1, v1, u2, v2)
        
        print("✅ Detector / Geometry / Labeling 준비 완료")
        
    # -------------------------
    # Utils
    # -------------------------
    def in_roi(self, u, v):
        x1, y1, x2, y2 = self.roi
        return x1 <= u <= x2 and y1 <= v <= y2
    
    def bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def depth_cm(self, u, v):
        d = self.depth[v, u]
        if d <= 0:
            return None
        return d * self.depth_scale * 100.0  # mm → cm
    
    def pixel_to_camera_cm(self, u, v, z):
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return X, Y, z
    
    # -------------------------
    # 1️⃣ Solid objects (DBSCAN)
    # -------------------------
    def detect_solid_objects(self, eps=2.0, min_samples=50):
        h, w = self.depth.shape
        pts_px, pts_world = [], []
        for v in range(0, h, 2):
            for u in range(0, w, 2):
    bbox: tuple  # (x1, y1, x2, y2)
    kind: str    # "solid" | "transparent"