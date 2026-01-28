import cv2
import numpy as np
import pyrealsense2 as rs
import os
from collections import deque

SAVE_FILE = "camcalib.npz"

# 마커 중심 월드 좌표 (cm), 테이블 평면 Z=0
WORLD_MARKER_CENTER = {
    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 40.0, 0.0], dtype=np.float32),
    2: np.array([36.0, 0.0, 0.0], dtype=np.float32),
    3: np.array([36.0, 40.0, 0.0], dtype=np.float32),
}

MARKER_SIZE_CM = 9.3  # <<<< 반드시 실측해서 정확히!
MIN_MARKERS_PER_FRAME = 2   # 프레임당 최소 마커 수 (코너=8점 이상 확보)
ACCUM_FRAMES = 60           # 누적 프레임 수
REPROJ_OK_MEAN_PX = 2.0     # 평균 재투영 오차 허용 기준(참고)

def make_marker_corners_world(center_xyz_cm, marker_size_cm):
    """center를 기준으로 4 코너의 월드 3D 좌표 생성 (Z=0)"""
    s = marker_size_cm
    half = s / 2.0
    # OpenCV ArUco 코너 순서: TL, TR, BR, BL
    offsets = np.array([
        [-half, -half, 0.0],  # TL
        [ half, -half, 0.0],  # TR
        [ half,  half, 0.0],  # BR
        [-half,  half, 0.0],  # BL
    ], dtype=np.float32)
    return center_xyz_cm.reshape(1, 3) + offsets

def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(err.mean()), float(err.max())

class RealsenseCalibratorAccurate:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                       [0, intr.fy, intr.ppy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.T_cam_to_work = None
        self.load_calibration()

        # 누적 버퍼
        self.acc_obj = deque(maxlen=ACCUM_FRAMES)
        self.acc_img = deque(maxlen=ACCUM_FRAMES)

    def load_calibration(self):
        if os.path.exists(SAVE_FILE):
            data = np.load(SAVE_FILE)
            self.T_cam_to_work = data["T_cam_to_work"]
            if "camera_matrix" in data and "dist_coeffs" in data:
                self.camera_matrix = data["camera_matrix"]
                self.dist_coeffs = data["dist_coeffs"]
            print(f"✅ 기존 캘리브레이션 로드: {SAVE_FILE}")
        else:
            print("⚠️ 저장된 설정 없음. 'Space'를 눌러 캘리브레이션")

    def accumulate_points(self, corners, ids):
        """현재 프레임에서 코너들을 월드/이미지 대응점으로 변환해 누적"""
        if ids is None:
            return 0, 0

        ids = ids.flatten()
        obj_pts_list = []
        img_pts_list = []
        used_markers = 0

        for i, mid in enumerate(ids):
            if mid not in WORLD_MARKER_CENTER:
                continue
            # 이미지 코너 4개 (TL,TR,BR,BL), shape (4,2)
            img_c = corners[i][0].astype(np.float32)
            # 월드 코너 4개 (cm)
            obj_c = make_marker_corners_world(WORLD_MARKER_CENTER[mid], MARKER_SIZE_CM)

            obj_pts_list.append(obj_c)
            img_pts_list.append(img_c)
            used_markers += 1

        if used_markers >= MIN_MARKERS_PER_FRAME:
            obj_pts = np.concatenate(obj_pts_list, axis=0)  # (4*M,3)
            img_pts = np.concatenate(img_pts_list, axis=0)  # (4*M,2)
            self.acc_obj.append(obj_pts)
            self.acc_img.append(img_pts)

        return used_markers, len(self.acc_obj)

    def solve_from_accumulated(self):
        """누적된 대응점으로 RANSAC+Refine로 최종 자세 추정"""
        if len(self.acc_obj) < max(10, ACCUM_FRAMES // 3):
            print(f"❌ 누적 프레임 부족: {len(self.acc_obj)}")
            return False

        obj_pts = np.concatenate(list(self.acc_obj), axis=0).astype(np.float32)
        img_pts = np.concatenate(list(self.acc_img), axis=0).astype(np.float32)

        # 1) RANSAC로 초기 (아웃라이어 제거)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
            iterationsCount=200,
            reprojectionError=3.0,   # px, 상황에 따라 2~5 조정
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok or inliers is None or len(inliers) < 20:
            print("❌ solvePnPRansac 실패 또는 inlier 부족")
            return False

        in_obj = obj_pts[inliers.flatten()]
        in_img = img_pts[inliers.flatten()]

        # 2) LM refine로 정밀화 (OpenCV 4.1+)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                in_obj, in_img, self.camera_matrix, self.dist_coeffs, rvec, tvec
            )
        except Exception:
            # 버전에 따라 Refine가 없을 수 있으니 fallback
            pass

        mean_px, max_px = reprojection_error(in_obj, in_img, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        # 변환행렬 구성 (월드->카메라)
        R, _ = cv2.Rodrigues(rvec)
        T_w2c = np.eye(4, dtype=np.float32)
        T_w2c[:3, :3] = R.astype(np.float32)
        T_w2c[:3, 3] = tvec.flatten().astype(np.float32)

        self.T_cam_to_work = np.linalg.inv(T_w2c).astype(np.float32)

        np.savez(
            SAVE_FILE,
            T_cam_to_work=self.T_cam_to_work,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_cm=np.float32(MARKER_SIZE_CM),
            reproj_mean_px=np.float32(mean_px),
            reproj_max_px=np.float32(max_px),
            inliers=np.int32(len(inliers)),
            total_points=np.int32(len(obj_pts)),
        )

        print("✅ 캘리브레이션 완료")
        print(f"   - 누적 프레임: {len(self.acc_obj)}")
        print(f"   - 총 점: {len(obj_pts)}, inlier: {len(inliers)}")
        print(f"   - 재투영 오차(mean/max): {mean_px:.2f}px / {max_px:.2f}px")
        print(f"   - 저장: {SAVE_FILE}")

        if mean_px > REPROJ_OK_MEAN_PX:
            print("⚠️ 평균 재투영 오차가 큰 편입니다. (마커 인쇄/실측/조명/흔들림/왜곡계수) 점검 권장.")

        return True

    def run(self):
        cv2.namedWindow("Calibration View")
        print("\n[작동법]")
        print("- SPACE: 누적된 프레임으로 캘리브레이션 수행 및 저장")
        print("- R: 누적 버퍼 리셋")
        print("- ESC: 종료\n")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_f = aligned.get_color_frame()
                if not color_f:
                    continue

                img = np.asanyarray(color_f.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)

                used_markers, acc_n = self.accumulate_points(corners, ids)

                # 상태 표시
                cv2.putText(img, f"markers:{used_markers}  acc_frames:{acc_n}/{ACCUM_FRAMES}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    self.solve_from_accumulated()
                elif key == ord('r'):
                    self.acc_obj.clear()
                    self.acc_img.clear()
                    print("🧹 누적 버퍼 리셋")
                elif key == 27:
                    break

                cv2.imshow("Calibration View", img)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealsenseCalibratorAccurate()
    app.run()
