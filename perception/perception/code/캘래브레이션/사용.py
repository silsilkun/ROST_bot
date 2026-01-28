import cv2
import numpy as np
import pyrealsense2 as rs
import os

# ===============================
# 설정 및 파일 경로
# ===============================
SAVE_FILE = "camcalib.npz"

class RealsenseCoordinatePicker:
    def __init__(self):
        # 1. RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # 2. 내장 파라미터(Intrinsics) 추출
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 3. 저장된 변환 행렬 로드
        self.T_cam_to_work = self.load_calibration()

    def load_calibration(self):
        if os.path.exists(SAVE_FILE):
            data = np.load(SAVE_FILE)
            matrix = data["T_cam_to_work"]
            print(f"✅ '{SAVE_FILE}' 로드 완료. 클릭하여 월드 좌표를 확인하세요.")
            return matrix
        else:
            print(f"❌ '{SAVE_FILE}' 파일을 찾을 수 없습니다. 먼저 캘리브레이션 코드를 실행하세요.")
            exit()

    def pixel_to_world(self, u, v, depth_frame):
        # 5x5 median depth (meters)
        depth_list = []
        W, H = 1280, 720  # 스트림 해상도와 일치
        for du in range(-2, 3):
            for dv in range(-2, 3):
                uu = u + du
                vv = v + dv
                if uu < 0 or uu >= W or vv < 0 or vv >= H:
                    continue
                d = depth_frame.get_distance(uu, vv)  # meters
                if d > 0:
                    depth_list.append(d)

        if not depth_list:
            return None

        Z_m = float(np.median(depth_list))  # meters

        fx, fy = float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])
        cx, cy = float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])

        # pinhole: u=x, v=y
        X_m = (u - cx) * Z_m / fx
        Y_m = (v - cy) * Z_m / fy

        # meters -> cm (월드가 cm라면)
        Pc = np.array([X_m * 100.0, Y_m * 100.0, Z_m * 100.0, 1.0], dtype=np.float32)
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]


        
    def run(self):
        clicked_pixel = None
        last_world_pos = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_pixel
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_pixel = (x, y)

        cv2.namedWindow("World Coordinate Picker")
        cv2.setMouseCallback("World Coordinate Picker", mouse_callback)

        print("\n[사용법]\n- 마우스 왼쪽 클릭: 해당 지점의 월드 좌표(cm) 출력\n- ESC: 종료\n")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_f = aligned.get_color_frame()
                depth_f = aligned.get_depth_frame()
                
                if not color_f or not depth_f: continue

                img = np.asanyarray(color_f.get_data())

                # 클릭된 지점이 있으면 좌표 변환 수행
                if clicked_pixel:
                    u, v = clicked_pixel
                    res = self.pixel_to_world(u, v, depth_f)
                    if res is not None:
                        last_world_pos = (u, v, res)
                        print(f"📍 클릭 위치({u}, {v}) -> 월드: X={res[0]:.2f}cm, Y={res[1]:.2f}cm, Z={res[2]:.2f}cm")
                    clicked_pixel = None

                # 화면에 마지막 클릭 지점 표시
                if last_world_pos:
                    u, v, pos = last_world_pos
                    cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(img, f"X:{pos[0]:.1f} Y:{pos[1]:.1f} Z:{pos[2]:.1f}", (u + 10, v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("World Coordinate Picker", img)
                if cv2.waitKey(1) == 27: break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealsenseCoordinatePicker()
    app.run()