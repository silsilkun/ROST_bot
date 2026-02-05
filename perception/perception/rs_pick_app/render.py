import cv2


def _draw_roi(img, roi_xyxy, color=(0, 255, 0), th=2):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
    return out


def render_live_view(color_img, roi_xyxy, clicked_uv):
    out = _draw_roi(color_img, roi_xyxy)
    for (u, v) in clicked_uv:
        cv2.circle(out, (int(u), int(v)), 5, (0, 0, 255), -1)
    return out


def render_result_view(color_img, roi_xyxy, ok, u, v, xyz_angle, u2v2):
    out = _draw_roi(color_img, roi_xyxy)

    if (not ok) or u is None or v is None or xyz_angle is None or u2v2 is None:
        cv2.putText(out, "FAIL", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
        return out

    X, Y, Zapp, angle = xyz_angle
    u2, v2 = u2v2

    cv2.circle(out, (int(u), int(v)), 8, (0, 0, 255), -1)  # pick (red)
    if (u2, v2) != (u, v):
        cv2.arrowedLine(out, (int(u), int(v)), (int(u2), int(v2)), (255, 0, 0), 2, tipLength=0.2)  # axis (blue)

    cv2.putText(out, f"uv=({int(u)},{int(v)})", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    txt = f"xyz=[{X:.1f},{Y:.1f},{Zapp:.1f},{angle:.1f}]"
    cv2.putText(out, txt, (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return out
