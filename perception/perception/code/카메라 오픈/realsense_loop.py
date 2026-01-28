# realsense_loop.py
import pyrealsense2 as rs
import numpy as np
import cv2


def _draw_points(img: np.ndarray, points):
    """
    points: iterable of (x, y) or (x, y, ...)
    """
    if not points:
        return img

    # overlay가 없으면 굳이 copy 안 뜨기 위해 여기서 copy
    out = img.copy()
    for p in points:
        if len(p) < 2:
            continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
    return out


def run(
    width=1280,
    height=720,
    fps=30,
    on_save=None,
    on_reset=None,
    on_click=None,
    update_depth_frame=None,
    update_color_image=None,
    get_points=None,
):
    """
    책임:
      - RealSense 스트림 설정/루프/종료
      - 프레임을 외부 state로 전달(update_*)
      - UI 입력(스페이스/r/esc) → 외부 콜백 호출
      - 마우스 콜백 연결(on_click)

    외부 모듈 책임:
      - 클릭/저장/검출/좌표변환 로직
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    align = rs.align(rs.stream.color)

    window_name = "RealSense Color"
    cv2.namedWindow(window_name)

    # ✅ on_click은 OpenCV가 호출. 기존처럼 그대로 연결
    if on_click is not None:
        cv2.setMouseCallback(window_name, on_click)

    print("스페이스바: 계산/저장 | r: 리셋 | esc: 종료")

    try:
        pipeline.start(config)

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # 갱신
            if update_depth_frame is not None:
                update_depth_frame(depth_frame)
            if update_color_image is not None:
                update_color_image(color_image)

            # 포인트 오버레이가 있을 때만 copy
            display = color_image
            if get_points is not None:
                pts = get_points()
                if pts:
                    display = _draw_points(color_image, pts)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("r") and on_reset:
                on_reset()
                continue
            elif key == ord(" ") and on_save:
                on_save()
                continue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense 종료")
