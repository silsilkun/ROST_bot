# main.py
import realsense_loop
import click_points
import pipeline


def main():
    realsense_loop.run(
        on_save=pipeline.save_cam,
        on_reset=click_points.reset_points,
        on_click=click_points.mouse_callback,
        update_depth_frame=click_points.update_depth_frame,
        update_color_image=click_points.update_color_image,
        get_points=click_points.get_saved_points,
    )


if __name__ == "__main__":
    main()
