import cv2
import numpy as np
import pyrealsense2 as rs
import os

SAVE_FILE = "camcalib.npz"

class RealsenseCoordinatePicker:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        self.T_cam_to_work = self.load_calibration()

        # ✅ 바닥 Z 샘플 저장용
        self.z_samples = []

    def load_calibration(self):
        if os.path.exists(SAVE_FILE):
            data = np.load(SAVE_FILE, allow_pickle=True)
            matrix = data["T_cam_to_work"].astype(np.float32)
            print(f"✅ '{SAVE_FILE}' 로드 완료.")
            return matrix
        else:
            raise FileNotFoundError(f"❌ '{SAVE_FILE}' 파일이 없습니다. 먼저 캘리브레이션 실행 필요.")

    def save_calibration(self, T_new):
        # 기존 npz에 다른 키들이 있을 수 있으니 모두 보존해서 overwrite
        data = np.load(SAVE_FILE, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        out["T_cam_to_work"] = T_new.astype(np.float32)
        np.savez(SAVE_FILE, **out)
        print(f"✅ 보정된 T_cam_to_work 저장 완료: {SAVE_FILE}")

    def pixel_to_world(self, u, v, depth_frame):
        # 5x5 median depth (meters)
        depth_list = []
        W, H = 1280, 720
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

        Z_m = float(np.median(depth_list))

        fx, fy = float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])
        cx, cy = float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])

        # ✅ 올바른 pinhole
        X_m = (u - cx) * Z_m / fx
        Y_m = (v - cy) * Z_m / fy

        # 월드가 cm라면 cm로 변환
        Pc = np.array([X_m * 100.0, Y_m * 100.0, Z_m * 100.0, 1.0], dtype=np.float32)
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]

    def apply_z_zero_correction(self):
        if len(self.z_samples) < 10:
            print(f"❌ Z 샘플이 부족합니다: {len(self.z_samples)}개 (최소 10개 권장)")
            return

        z_bias = float(np.median(self.z_samples))  # cm
        T_shift = np.eye(4, dtype=np.float32)
        T_shift[2, 3] = -z_bias

        T_new = T_shift @ self.T_cam_to_work

        print("==== Z=0 보정 ====")
        print(f"- z_bias (median): {z_bias:.3f} cm")
        print("- 적용: T_new = T_shift @ T_cam_to_work")
        print("==================")

        self.T_cam_to_work = T_new
        self.save_calibration(T_new)

        # 샘플 리셋(원하면 유지해도 됨)
        self.z_samples.clear()
        print("✅ 보정 완료. 이제 바닥 클릭 시 Z≈0 확인하세요.")

    def run(self):
        clicked_pixel = None
        last_world_pos = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_pixel
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_pixel = (x, y)

        cv2.namedWindow("World Coordinate Picker")
        cv2.setMouseCallback("World Coordinate Picker", mouse_callback)

        print("\n[사용법]")
        print("- 좌클릭: 월드 좌표 출력 + Z 샘플 누적")
        print("- B: 누적된 Z로 '바닥=Z0' 보정 후 저장")
        print("- R: Z 샘플 리셋")
        print("- ESC: 종료\n")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_f = aligned.get_color_frame()
                depth_f = aligned.get_depth_frame()
                if not color_f or not depth_f:
                    continue

                img = np.asanyarray(color_f.get_data())

                if clicked_pixel:
                    u, v = clicked_pixel
                    res = self.pixel_to_world(u, v, depth_f)
                    if res is not None:
                        last_world_pos = (u, v, res)
                        self.z_samples.append(float(res[2]))
                        print(f"📍 ({u},{v}) -> X={res[0]:.2f}cm, Y={res[1]:.2f}cm, Z={res[2]:.2f}cm | Zsamples={len(self.z_samples)}")
                    else:
                        print("⚠️ depth 없음(0). 다른 지점 클릭")
                    clicked_pixel = None

                if last_world_pos:
                    u, v, pos = last_world_pos
                    cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(img, f"X:{pos[0]:.1f} Y:{pos[1]:.1f} Z:{pos[2]:.1f}",
                                (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 키 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('b'):
                    self.apply_z_zero_correction()
                elif key == ord('r'):
                    self.z_samples.clear()
                    print("🧹 Z 샘플 리셋")
                elif key == 27:
                    break

                cv2.imshow("World Coordinate Picker", img)

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = RealsenseCoordinatePicker()
    app.run()
