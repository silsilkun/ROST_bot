import cv2
import numpy as np
from sklearn.cluster import DBSCAN

COLOR_PATH = "color.png"
DEPTH_PATH = "depth.npy"
CALIB_PATH = "camcalib.npz"




class DepthDBSCANVisualizer:
    def __init__(self):
        self.color = cv2.imread(COLOR_PATH)
        self.depth = np.load(DEPTH_PATH)
        calib = np.load(CALIB_PATH)

        
        if self.color is None or self.depth is None:
            raise RuntimeError("❌ color.png 또는 depth.npy 로드 실패")

        self.T_cam_to_work = calib["T_cam_to_work"]
        self.camera_matrix = calib["camera_matrix"]
    
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        self.depth_scale = 0.001 # mm → m
        self.Z_floor = -2.0

        # ROI (u1, v1, u2, v2)
        self.roi = (460, 150, 830, 490)

        print("✅ ROI + Depth + DBSCAN + Depth-discontinuity Transparent 준비 완료")

    # -------------------------------
    # Utils
    # -------------------------------
    def in_roi(self, u, v):
        x1, y1, x2, y2 = self.roi
        return x1 <= u <= x2 and y1 <= v <= y2

    def pixel_to_world(self, u, v, depth_cm):
        Xc = (u - self.cx) * depth_cm / self.fx
        Yc = (v - self.cy) * depth_cm / self.fy
        Zc = depth_cm - 1.4 # 카메라 높이 보정
        Pc = np.array([Xc, Yc, Zc, 1.0])
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]

    # -------------------------------
    # Depth → DBSCAN → Rotated Box
    # -------------------------------
    #eps=2.0: 월드 좌표기준 거리, min_samples=50: 최소포인트수
      
    def extract_objects_dbscan_rotated(self, eps=2.0, min_samples=50):  
        h, w = self.depth.shape
        points_2d = []
        points_world = []

        for v in range(0, h, 2):
            for u in range(0, w, 2):
                if not self.in_roi(u, v):
                    continue

                d = self.depth[v, u]
                if d <= 0:
                    continue

                depth_cm = d * self.depth_scale * 100.0  # mm → cm
                Pw = self.pixel_to_world(u, v, depth_cm)

                if Pw[2] <= self.Z_floor:  # 작업 평면을 기준으로 돌출된 물체의 포인트만 사용
                    points_2d.append([u, v])
                    points_world.append(Pw)

        if len(points_world) == 0:
            return []

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_world)
        labels = clustering.labels_

        rotated_boxes = []

        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx], dtype=np.float32)

            if len(pixels) < 20:
                continue

            rect = cv2.minAreaRect(pixels)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            rotated_boxes.append(box)

        return rotated_boxes

    # -------------------------------
    # RGB + Depth Discontinuity → Transparent
    # -------------------------------
    def extract_transparent_rgb(self):
        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY) #그레이스케일 + 밝기 + 에지 기반

        # 1️⃣ Bright + Edge
        bright = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31, -5
        )

        edges1 = cv2.Canny(gray, 20, 60)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        rgb_mask = cv2.bitwise_or(bright, edges)

        # 2️⃣ Depth 불연속 영역 생성 (핵심)
        depth = self.depth.astype(np.float32)
        depth_blur = cv2.medianBlur(depth, 5)

        grad = cv2.Laplacian(depth_blur, cv2.CV_32F)
        grad_abs = np.abs(grad)

        depth_hole = np.zeros_like(depth, dtype=np.uint8)
        depth_hole[(depth == 0) | (grad_abs > 20)] = 255 
        # Rgb의 밝기, 에지, Depth 불연속 영역을 결합하여 깊이 센서가 인식하지 못한 영역

        # 3️⃣ 통합
        mask = cv2.bitwise_and(rgb_mask, depth_hole)

        # ROI 제한
        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        # Morphology
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        roi_area = (x2 - x1) * (y2 - y1)

        for cnt in contours:
            if cv2.contourArea(cnt) < roi_area * 0.002:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

        return boxes

    # -------------------------------
    # 실행
    # -------------------------------
    def run(self):
        vis = self.color.copy()

        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        green_boxes = self.extract_objects_dbscan_rotated()
        for box in green_boxes:
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)

        blue_boxes = self.extract_transparent_rgb()
        blue_boxes = suppress_blue_boxes(blue_boxes, green_boxes)

        for bx1, by1, bx2, by2 in blue_boxes:
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        cv2.imshow("Depth Discontinuity Transparent Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -------------------------------
# 초록 박스 내부 파랑 박스 제거
# -------------------------------
def suppress_blue_boxes(blue_boxes, green_rotated_boxes):
    filtered = []

    for bx1, by1, bx2, by2 in blue_boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        keep = True
        for box in green_rotated_boxes:
            if cv2.pointPolygonTest(box, (cx, cy), False) >= 0:
                keep = False
                break

        if keep:
            filtered.append((bx1, by1, bx2, by2))

    return filtered


if __name__ == "__main__":
    app = DepthDBSCANVisualizer()
    app.run()