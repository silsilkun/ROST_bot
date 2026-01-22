from perception.perception.pipelines import perception_pipeline
from perception.perception.runtime import realsense_runtime
from perception.perception.runtime import click_input_ui

from rost_interfaces.srv import PerceptionToEstimation

import rclpy
from rclpy.node import Node




class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        self.srv = self.create_client(PerceptionToEstimation, 'perception_to_estimation')
        while not self.srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = PerceptionToEstimation.Request()

        self.processed_result = realsense_runtime.run(
                on_save=perception_pipeline.save_cam,
                on_reset=click_input_ui.reset_points,
                on_click=click_input_ui.mouse_callback,
                update_depth_frame=click_input_ui.update_depth_frame,
                update_color_image=click_input_ui.update_color_image,
                get_points=click_input_ui.click_points.get_saved_points,
        )

    def send_request(self):
        self.req.image = self.processed_result['vis'].tobytes()

        
