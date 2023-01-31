# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

import numpy as np
import rclpy
import torchvision
import torchvision.transforms as transforms
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes

import torch

IMG_CENTER_X = 480
IMG_CENTER_Y = 360
SCORE_THRESHOLD = 0.2


class Label(Enum):
    FRONT = 1
    BACK = 2


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/image", self.listener_callback, 10
        )
        self.publisher_distance_movement = self.create_publisher(String, "distance", 10)
        self.publisher_bounding_box = self.create_publisher(Image, "bounding_box", 10)
        self.model = self.load_model()
        # inference on the GPU or on the CPU, if a GPU is not available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # move model to the right device
        self.model.to(self.device)

    def listener_callback(self, msg: Image):
        self.get_logger().info("Image received")
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError:
            return

        to_tensor = transforms.ToTensor()
        tensor = to_tensor(image)

        prediction = self.inference(tensor)
        box = prediction[0]["boxes"].cpu().numpy()[0]
        self.publish_distance_movement(box)
        self.publish_bounding_box(box)

    def load_model(self):
        # our dataset has three classes - background, front and back
        num_classes = 3  # use our dataset and defined transformations
        model = (
            torchvision.models.detection.fasterrcnn_resnet50_fpn()
        )  # No pretrained weights are used as they are loaded below
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # TODO: data/model/fasterrcnn_resnet50_fpn.pt
        model.load_state_dict(
            torch.load("fasterrcnn_resnet50_fpn.pt", map_location=torch.device("cpu"))
        )

        # Use if GPU is available
        # model.load_state_dict(torch.load("data/model/fasterrcnn_resnet50_fpn.pt"))

        return model

    def inference(self, image):
        # put the model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([image.to(self.device)])
        return prediction

    def calc_distance(box):
        _, ytl, _, ybr = box
        f = 2
        real_height = 300
        image_height = 235
        object_height = abs(ytl - ybr)
        sensor_height = 2.77
        distance = (f * real_height * image_height) / (object_height * sensor_height)
        return distance

    def calc_center(box):
        xtl, ytl, xbr, ybr = box
        center = [(xtl + xbr) / 2, (ytl + ybr) / 2]
        return center

    def publish_distance_movement(self, box):
        distance = self.calc_distance(box)
        box_center_x, _ = self.calc_center(box)
        if box_center_x > IMG_CENTER_X:
            msg = f"""Distance to object (mm): {distance:.2f}
            Move right {box_center_x - IMG_CENTER_X:.0f}px"""
        else:
            msg = f"""Distance to object (mm): {distance:.2f}
            Move left {IMG_CENTER_X - box_center_x:.0f}px"""
        self.publisher_dist_mov.publish(msg)

    def extract_labels(self, prediction):
        length = len(prediction[0]["labels"])
        labels = []
        for i in range(length):
            if prediction[0]["scores"].cpu().numpy()[i] > SCORE_THRESHOLD:
                label = Label(prediction[0]["labels"].cpu().numpy()[i]).name
                score = prediction[0]["scores"].cpu().numpy()[i]
                txt = "{}:{}".format(label, score)
                labels.append(txt)
        return labels

    def publish_bounding_box(self, prediction):
        labels = self.extract_labels(prediction)
        tensor = draw_bounding_boxes(
            image,
            boxes=prediction[0]["boxes"][prediction[0]["scores"] > SCORE_THRESHOLD],
            labels=labels,
            width=4,
            font_size=150,
        )

        image = np.transpose(tensor.numpy(), (1, 2, 0))
        msg = self.bridge.cv2_to_imgmsg(image, encoding="passthrough")
        self.publisher_bounding_box.publish(msg)


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
