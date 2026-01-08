import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from perception.msg import DetectedObject, DetectedObjectArray

from perception.utils.camera import capture_frame
from perception.utils.detection import detect_objects


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.publisher_ = self.create_publisher(
            DetectedObjectArray,
            'perception/objects',
            10
        )

        self.timer = self.create_timer(
            1.0,   # 1초에 한 번
            self.timer_callback
        )

        self.get_logger().info('Perception node started')

    def timer_callback(self):
        frame = capture_frame()

        detections = detect_objects(frame)
        # detections 예시:
        # [
        #   {"id": 0, "label": "can", "score": 0.9,
        #    "cx": 120, "cy": 200, "w": 50, "h": 80}
        # ]

        msg = DetectedObjectArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        for d in detections:
            obj = DetectedObject()
            obj.id = d['id']
            obj.label = d['label']
            obj.score = d['score']
            obj.center_x = d['cx']
            obj.center_y = d['cy']
            obj.width = d['w']
            obj.height = d['h']
            msg.objects.append(obj)

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published {len(msg.objects)} objects')


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
