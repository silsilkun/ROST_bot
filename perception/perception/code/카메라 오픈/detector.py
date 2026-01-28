# detector.py
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from coordinate import Coordinate
from depth_utils import FakeDepthFrameFromNpy, box_center_pixel


def long_side_angle_0_180(rect) -> float:
    """
    minAreaRect에서 가장 긴 변의 방향을 기준으로
    각도를 0~90도 범위로 정규화
    """
    box = cv2.boxPoints(rect).astype(np.float32)
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    lens = [float(np.hypot(v[0], v[1])) for v in edges]

    # 가장 긴 변 선택
    v = edges[int(np.argmax(lens))]
    ang = float(np.degrees(np.arctan2(v[0], v[1])))

    # 각도 범위 정리
    if ang < 0:
        ang += 180.0
    if ang > 90.0:
        ang = 180.0 - ang
    return ang


def suppress_blue_boxes(blue_boxes, green_polys):
    """
    파란 박스의 중심점이
    초록 폴리곤 내부에 있으면 제거
    (depth가 있는 물체를 우선 신뢰)
    """
    filtered = []
    for bx1, by1, bx2, by2 in blue_boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        keep = True
        for poly in green_polys:
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                keep = False
                break

        if keep:
            filtered.append((bx1, by1, bx2, by2))
    return filtered


class DepthDBSCANVisualizer:
    """
    color(BGR) + depth(z16, mm)를 입력으로 받아
    초록/파랑 객체를 검출하고 시각화 결과를 반환하는 클래스
    """

    def __init__(self):
        # 입력 프레임
        self.color = None
        self.depth_z16 = None
        self.depth_src = None

        # 좌표 변환은 Coordinate로 통일
        self.coord = Coordinate()

        # 관심 영역(ROI)
        self.roi = (470, 85, 800, 313)

        # ===============================
        # 🔧 하드코딩된 객체 규정 (절대값)
        # ===============================

        # 초록(GREEN) 규정
        self.GREEN_Z_MIN = 5.0          # 높이 Z 최소값 (cm 이상)
        self.GREEN_EDGE_MIN = 20.0      # 짧은 변 최소 길이 (px)
        self.GREEN_EDGE_MAX = 990.0   # 짧은 변 최대 길이
        self.GREEN_AREA_MIN = 250.0     # 면적 최소값 (px^2)
        self.GREEN_AREA_MAX = 999999.0  # 면적 최대값

        # 파랑(BLUE) 규정
        self.BLUE_AREA_MIN = 1500.0     # 면적 최소값 (px^2)
        self.BLUE_AREA_MAX = 999999.0   # 면적 최대값

        # DBSCAN 파라미터
        self.dbscan_eps = 2.0
        self.dbscan_min_samples = 50

        print("Detector ready (하드코딩 규정 기반)")

    def update(self, color, depth_z16):
        """
        스냅샷 입력
        """
        self.color = color
        self.depth_z16 = depth_z16
        self.depth_src = FakeDepthFrameFromNpy(depth_z16)

    def in_roi(self, u, v):
        """
        ROI 내부 픽셀 여부
        """
        x1, y1, x2, y2 = self.roi
        return x1 <= u <= x2 and y1 <= v <= y2

    # -----------------------------
    # 초록(GREEN) 통과 규칙
    # -----------------------------
    def _pass_green(self, Z_cm: float, area_px: float, min_edge_px: float) -> bool:
        """
        초록 객체 규정:
        - 높이 Z >= 기준
        - 짧은 변 길이 범위 내
        - 면적 범위 내
        """
        if Z_cm < self.GREEN_Z_MIN:
            return False
        if not (self.GREEN_EDGE_MIN <= min_edge_px <= self.GREEN_EDGE_MAX):
            return False
        if not (self.GREEN_AREA_MIN <= area_px <= self.GREEN_AREA_MAX):
            return False
        return True

    # -----------------------------
    # 파랑(BLUE) 통과 규칙
    # -----------------------------
    def _pass_blue(self, area_px: float) -> bool:
        """
        파랑 객체 규정:
        - 면적만 사용
        """
        return self.BLUE_AREA_MIN <= area_px <= self.BLUE_AREA_MAX

    def extract_objects_dbscan_rotated(self):
        """
        depth 기반(DBSCAN) 초록 객체 검출
        """
        if self.depth_z16 is None:
            return []

        h, w = self.depth_z16.shape
        world_pts = []
        pixel_pts = []

        # ROI 내부를 2픽셀 간격으로 샘플링
        for v in range(0, h, 2):
            for u in range(0, w, 2):
                if not self.in_roi(u, v):
                    continue
                if self.depth_z16[v, u] <= 0:
                    continue

                Pw = self.coord.pixel_to_world(u, v, self.depth_src)
                if Pw is None:
                    continue

                X, Y, Z = Pw[:3]

                # 높이 조건 1차 필터
                if Z >= self.GREEN_Z_MIN:
                    world_pts.append([X, Y, Z])
                    pixel_pts.append([u, v])

        if not world_pts:
            return []

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples
        ).fit(world_pts).labels_

        green_items = []
        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            if len(idx) < 20:
                continue

            pixels = np.array([pixel_pts[i] for i in idx], dtype=np.float32)
            rect = cv2.minAreaRect(pixels)
            w_rect, h_rect = rect[1]

            area_px = float(w_rect * h_rect)
            min_edge_px = float(min(w_rect, h_rect))

            # 클러스터의 대표 높이(Z)는 중앙값 사용
            z_vals = [world_pts[i][2] for i in idx]
            Z_cm = float(np.median(z_vals))

            if not self._pass_green(Z_cm, area_px, min_edge_px):
                continue

            angle = long_side_angle_0_180(rect)
            box = cv2.boxPoints(rect).astype(np.int32)
            green_items.append({"box": box, "angle": angle})

        return green_items

    def extract_transparent_rgb(self):
        """
        RGB + depth hole 기반 파랑 객체 검출
        """
        if self.color is None or self.depth_z16 is None:
            return []

        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        # 밝기 기반 마스크
        bright = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31, -5
        )

        # 에지 기반 마스크
        edges = cv2.bitwise_or(
            cv2.Canny(gray, 20, 60),
            cv2.Canny(gray, 50, 150),
        )

        rgb_mask = cv2.bitwise_or(bright, edges)

        # depth hole 검출
        depth = self.depth_z16.astype(np.float32)
        depth_blur = cv2.medianBlur(depth, 5)
        grad = np.abs(cv2.Laplacian(depth_blur, cv2.CV_32F))

        depth_hole = np.zeros_like(depth, dtype=np.uint8)
        depth_hole[(depth == 0) | (grad > 20)] = 255

        mask = cv2.bitwise_and(rgb_mask, depth_hole)

        # ROI 적용
        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))

            # 파랑 객체 면적 규정
            if not self._pass_blue(area):
                continue

            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            # 중심점 계산
            dist = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, 5)
            _, _, _, (cx, cy) = cv2.minMaxLoc(dist)

            size = int(np.sqrt(area) * 0.6)
            boxes.append((
                max(cx - size // 2, 0),
                max(cy - size // 2, 0),
                min(cx + size // 2, mask.shape[1]),
                min(cy + size // 2, mask.shape[0]),
            ))

        return boxes

    def run(self):
        """
        검출 실행 및 시각화
        """
        if self.color is None or self.depth_z16 is None:
            raise RuntimeError("run() 전에 update()를 호출하세요.")

        vis = self.color.copy()

        # ROI 표시
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 초록 / 파랑 검출
        green_items = self.extract_objects_dbscan_rotated()
        green_polys = [it["box"] for it in green_items]
        blue_boxes = suppress_blue_boxes(self.extract_transparent_rgb(), green_polys)

        items = []
        idx = 0

        # 초록 객체 추가
        for it in green_items:
            items.append({
                "id": idx,
                "type": "green",
                "poly": it["box"],
                "angle": float(it["angle"]),
                "rect": None,
            })
            idx += 1

        # 파랑 객체 추가
        for (x1b, y1b, x2b, y2b) in blue_boxes:
            poly = np.array(
                [[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]],
                dtype=np.int32
            )
            items.append({
                "id": idx,
                "type": "blue",
                "poly": poly,
                "angle": 0.0,
                "rect": (x1b, y1b, x2b, y2b),
            })
            idx += 1

        # 시각화
        for item in items:
            color = (0, 255, 0) if item["type"] == "green" else (255, 0, 0)
            box = item["poly"]

            cv2.drawContours(vis, [box], 0, color, 2)

            cx, cy = box_center_pixel(box)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

            cv2.putText(
                vis, f"{item['id']}",
                (cx, max(0, cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        return vis, items


if __name__ == "__main__":
    raise SystemExit("이 파일은 단독 실행용이 아닙니다. main.py에서 import 해서 사용하세요.")
