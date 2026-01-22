# main.py
from ..runtime import realsense_runtime
from ..runtime import click_input_ui
from ..pipelines import perception_pipeline

# RealSense 카메라 실행 (스페이스바 누르면 pipeline.save_cam 호출)
realsense_runtime.run(
    on_save=perception_pipeline.save_cam,
    on_reset=click_input_ui.reset_points,
    on_click=click_input_ui.mouse_callback,
    update_depth_frame=click_input_ui.update_depth_frame,
    update_color_image=click_input_ui.update_color_image,
    get_points=click_input_ui.get_saved_points,
)
