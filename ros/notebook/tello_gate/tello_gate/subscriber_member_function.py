import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from tello_gate_msgs.srv import CurrentImage


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.sub = self.create_subscription(Image, "/image", self.listener_callback, 10)
        self.srv = self.create_service(
            CurrentImage, "get_current_image", self.get_current_image_callback
        )
        self.msg = None

    def listener_callback(self, msg):
        self.get_logger().info("Update msg")
        self.msg = msg

    def get_current_image_callback(self, _, response):
        response.image = self.msg
        self.get_logger().info("Serving current image")
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
