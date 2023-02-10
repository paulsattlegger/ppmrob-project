from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

BASE_DIR = Path("data/images")


class ReplayNode(Node):
    def __init__(self):
        super().__init__("replay_node")
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, "/image", 10)
        timer_period = 1 / 30  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        path = BASE_DIR / f"picture5_{self.i}.png"
        self.get_logger().error(f"Publishing image {path}")

        cv_bgr8_image = cv2.imread(str(path))

        cv2.imshow("Live", cv_bgr8_image)
        cv2.waitKey(1)

        msg = self.bridge.cv2_to_imgmsg(cv_bgr8_image, encoding="bgr8")
        self.publisher_.publish(msg)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    replay_node = ReplayNode()

    limit = int(max(BASE_DIR.iterdir()).stem.rsplit("_", 1)[1])

    while replay_node.i <= limit:
        rclpy.spin_once(replay_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    replay_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
