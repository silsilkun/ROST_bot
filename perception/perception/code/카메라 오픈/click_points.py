# click_points.py
import cv2
import numpy as np

from coordinate import Coordinate
from depth_utils import FakeDepthFrameFromNpy  # ✅ 중복 제거: 여기서만 import


class _State:
    __slots__ = ("depth_frame", "color_image", "points")

    def __init__(self):
        self.depth_frame = None   # RealSense depth_frame
        self.color_image = None   # np.ndarray(BGR)
        self.points = []          # [(x, y, depth_mm_click), ...]  # depth_mm는 디버깅 표시용


_S = _State()  #전역

def update_depth_frame(frame):
    _S.depth_frame = frame


def update_color_image(frame):
    _S.color_image = frame


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if _S.depth_frame is None:
        print("Depth 프레임 없음")
        return

    depth_mm = float(_S.depth_frame.get_distance(x, y)) * 1000.0  # 디버깅 표시용
    _S.points.append((int(x), int(y), depth_mm))
    print(f"Click: x={x}, y={y}, depth={depth_mm:.1f} mm")


def Save_Cam(picker=None):
    """
    스페이스바 순간:
      - color/depth 스냅샷(copy)
      - 스냅샷 depth(z16) 기준으로 클릭점 world 변환
    return: (color_image, depth_image_z16, points_3d)
    """
    if _S.depth_frame is None or _S.color_image is None:
        print("프레임 없음")
        return None, None, []

    color_image = _S.color_image.copy()
    depth_z16 = np.asanyarray(_S.depth_frame.get_data()).copy()  # z16(mm) snapshot

    # 스냅샷 ndarray를 Coordinate에 넣기 위한 어댑터(get_distance 제공)
    depth_src = FakeDepthFrameFromNpy(depth_z16)

    # 가능하면 외부에서 1회 생성한 Coordinate를 picker로 주입 권장
    picker = picker or Coordinate()

    points_3d = []
    for u, v, _depth_mm_click in _S.points:
        Pw = picker.pixel_to_world(u, v, depth_src)
        if Pw is not None:
            points_3d.append(Pw[:3])

    print(f"3D 좌표 계산 완료 ({len(points_3d)}개 점)" if points_3d else "저장할 3D 좌표 없음")
    return color_image, depth_z16, points_3d


def reset_points():
    _S.points.clear()
    print("포인트 리셋 완료")


def get_saved_points():
    # 외부에서 리스트를 수정해도 내부 상태가 오염되지 않게 복사본 반환
    return list(_S.points)


# flat_clicked_xy가 필요하면 이 함수 호출해서 쓰면 됨
def get_clicked_xy():
    return [(x, y) for (x, y, _d) in _S.points]
