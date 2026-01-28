# pipeline.py
import os
import numpy as np
import cv2

import click_points
from detector import DepthDBSCANVisualizer
import depth_utils
from coordinate import Coordinate
from settings import OUTPUT_DIR, COLOR_PATH, DEPTH_PATH

# --- singletons (불변/서비스 객체는 1회 생성) ---
detector = DepthDBSCANVisualizer()
coord = Coordinate()


def _angle_for_green(item_angle_long_0_180: float) -> float:
    """
    기존 로직 유지:
    - 긴변 각도(0~180) -> 0~90으로 접고
    - 집게 접근 각도 = 긴변 + 90 (mod 180)
    """
    ang_long = float(item_angle_long_0_180)
    if ang_long > 90.0:
        ang_long = 180.0 - ang_long
    return (ang_long + 90.0) % 180.0


def save_cam():
    """
    Space 순간:
      1) click_points에서 color/depth snapshot + 클릭점 3D 계산
      2) detector로 vis/items 생성
      3) 결과 저장(vis 이미지 + depth.npy)
      4) items(초록/파랑)를 world_list로 변환 + flat 리스트 생성
    return: processed_result dict
    """

    # ✅ 호출마다 새 dict로 초기화 (이전 결과 섞임 방지)
    processed_result = {
        "color": None,
        "depth": None,
        "points_3d": None,
        "vis": None,
        "boxes": None,
        "green_items": None,              # 현재 사용 안 함(호환 키 유지)
        "world_list": None,
        "clicked_world_xy_list": None,
        "flat_clicked_xy": None,
        "flat_world_list": None,
    }

    # 1) 스냅샷 + 클릭 기반 world 좌표(3D)
    color, depth_z16, points_3d = click_points.Save_Cam()

    processed_result["color"] = color
    processed_result["depth"] = depth_z16
    processed_result["points_3d"] = points_3d

    if color is None or depth_z16 is None:
        print("save_cam: color/depth 없음")
        # 키는 유지된 상태로 그대로 반환
        processed_result["clicked_world_xy_list"] = []
        processed_result["flat_clicked_xy"] = []
        processed_result["world_list"] = []
        processed_result["boxes"] = []
        processed_result["flat_world_list"] = []
        return processed_result

    # 1-1) 클릭 world에서 XY만 추출 (요구사항 유지)
    clicked_world_xy_list = [[float(p[0]), float(p[1])] for p in points_3d]
    flat_clicked_xy = clicked_world_xy_list

    processed_result["clicked_world_xy_list"] = clicked_world_xy_list
    processed_result["flat_clicked_xy"] = flat_clicked_xy

    # 2) detect 실행 (vis/items)
    detector.update(color, depth_z16)      # ✅ depth는 z16 ndarray(mm)
    vis, items = detector.run()

    processed_result["vis"] = vis
    processed_result["boxes"] = [it["poly"] for it in items]

    # 3) 저장 (ID가 포함된 vis를 저장)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(COLOR_PATH, vis)
    np.save(DEPTH_PATH, depth_z16)

    # 4) world_list 생성 (초록 + 파랑, id 유지)
    # ✅ green은 여기서 depth_src를 1번만 만들어 재사용
    depth_src = depth_utils.FakeDepthFrameFromNpy(depth_z16)

    world_list = []
    for it in items:
        obj_id = int(it["id"])
        obj_type = it["type"]

        Pw = None
        ang = 0.0

        if obj_type == "green":
            cx, cy = depth_utils.box_center_pixel(it["poly"])
            Pw = coord.pixel_to_world(cx, cy, depth_src)
            ang = _angle_for_green(it.get("angle", 0.0))

        else:
            # blue는 depth hole이 많아서 "safe" 탐색 사용
            Pw = depth_utils.blue_rect_to_world_safe(it["rect"], depth_z16, coord, depth_src, search_step=2)
            ang = 0.0

        # 실패해도 id 유지 (기존 동작 유지)
        if Pw is None:
            X = Y = Z = 0.0
        else:
            X, Y, Z = map(float, Pw[:3])

        world_list.append({
            "id": obj_id,
            "type": obj_type,
            "world": (float(X), float(Y), float(Z)),
            "angle": float(ang),
        })

    processed_result["world_list"] = world_list

    # 4-1) flat_world_list (ID 순서 유지)
    flat_world_list = []
    for it in sorted(world_list, key=lambda d: d["id"]):
        X, Y, Z = it["world"]
        flat_world_list.extend([it["id"], X, Y, Z, float(it["angle"])])

    processed_result["flat_world_list"] = flat_world_list
    processed_result["green_items"] = None

    # 5) 결과 창 표시 (기존 유지)
    cv2.imshow("Detect Result", vis)
    cv2.waitKey(1)

    # 유지 요구사항(로그)
    print("flat_world_list:", flat_world_list)
    print("flat_clicked_xy:", flat_clicked_xy)

    return processed_result