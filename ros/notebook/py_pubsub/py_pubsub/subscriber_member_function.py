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

import cv2
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
REAL_HEIGHT = 300  # mm
IMAGE_HEIGHT = 720  # px
FOCAL_LENGTH = 2  # mm
SENSOR_HEIGHT = 2.77  # mm


class Label(Enum):
    FRONT = 1
    BACK = 2


def cv2_to_pt(rgb8_image):
    # reshape the image from (H x W x C) to (C x H x W) format
    np_rgb8_image = rgb8_image.transpose((2, 0, 1))
    # create a tensor from a numpy.ndarray
    pt_rgb8_image = torch.from_numpy(np_rgb8_image)
    return pt_rgb8_image


def pt_to_cv2(rgb8_image):
    # reshape the image from (C x H x W) to (H x W x C) format
    cv_rgb8_image = rgb8_image.numpy().transpose((1, 2, 0))
    # convert rgb8 to bgr8
    cv_bgr8_image = cv2.cvtColor(cv_rgb8_image, cv2.COLOR_RGB2BGR)
    return cv_bgr8_image


def calc_distance(box):
    _, ytl, _, ybr = box
    object_height = abs(ytl - ybr)  # px
    distance = (FOCAL_LENGTH * REAL_HEIGHT * IMAGE_HEIGHT) / (
        object_height * SENSOR_HEIGHT
    )
    return distance


def calc_center(box):
    xtl, ytl, xbr, ybr = box
    center = [(xtl + xbr) / 2, (ytl + ybr) / 2]
    return center


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("minimal_subscriber")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/image", self.listener_callback, 10
        )
        self.publisher_distance_movement = self.create_publisher(
            String, "/distance", 10
        )
        self.publisher_bounding_box = self.create_publisher(Image, "/bounding_box", 10)
        # inference on the GPU or on the CPU, if a GPU is not available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.load_model()
        # move model to the right device
        self.model.to(self.device)

    def listener_callback(self, msg: Image):
        self.get_logger().info("Image received")
        try:
            # convert the image from BGR to RGB format
            cv_rgb8_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except CvBridgeError:
            return

        to_tensor = transforms.ToTensor()
        tensor = to_tensor(cv_rgb8_image)

        prediction = self.inference(tensor)
        if len(prediction[0]["boxes"]) > 0:
            box = prediction[0]["boxes"][0]
            self.publish_distance_movement(box)
            self.publish_bounding_box(cv_rgb8_image, prediction)

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
        model.load_state_dict(
            torch.load(
                "data/model/fasterrcnn_resnet50_fpn.pt", map_location=self.device
            )
        )
        return model

    def inference(self, image):
        # put the model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([image.to(self.device)])
        return prediction

    def publish_distance_movement(self, box):
        _, ytl, _, ybr = box
        distance = calc_distance(box)
        box_center_x, _ = calc_center(box)
        msg = String()
        object_height = abs(ytl - ybr)
        scale = object_height / REAL_HEIGHT
        if box_center_x > IMG_CENTER_X:
            error = box_center_x - IMG_CENTER_X
            msg.data = f"Distance to object (mm): {distance:.2f}, Move right {error:.0f}px {error*scale:.0f}mm"
        else:
            error = IMG_CENTER_X - box_center_x
            msg.data = f"Distance to object (mm): {distance:.2f}, Move left {error:.0f}px {error*scale:.0f}mm"
        self.get_logger().info(msg.data)
        self.publisher_distance_movement.publish(msg)

    def extract_labels(self, prediction):
        length = len(prediction[0]["labels"])
        labels = []
        for i in range(length):
            if prediction[0]["scores"][i] > SCORE_THRESHOLD:
                label = Label(prediction[0]["labels"][i].item()).name
                score = prediction[0]["scores"][i]
                txt = "{}:{}".format(label, score)
                labels.append(txt)
        return labels

    def publish_bounding_box(self, cv_rgb8_image, prediction):
        labels = self.extract_labels(prediction)
        pt_rgb8_image = draw_bounding_boxes(
            cv2_to_pt(cv_rgb8_image),
            boxes=prediction[0]["boxes"][prediction[0]["scores"] > SCORE_THRESHOLD],
            labels=labels,
            width=4,
            font_size=150,
        )

        self.get_logger().info("Converting image")
        msg = self.bridge.cv2_to_imgmsg(pt_to_cv2(pt_rgb8_image), encoding="bgr8")
        self.get_logger().info("Publishing bounding box")
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
