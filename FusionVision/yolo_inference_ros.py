import cv2
import numpy as np
import torch
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from utils import perform_yolo_inference, get_color

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Inference com câmera ROS')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--bbox_color', type=str, default="red")
    parser.add_argument('--font_scale', type=float, default=0.5)
    parser.add_argument('--font_thickness', type=int, default=1)
    return parser.parse_args()

class YoloRosNode(Node):
    def __init__(self, args):
        super().__init__('yolo_inference_ros')
        self.args = args
        self.bridge = CvBridge()
        self.get_logger().info(f'Carregando modelo YOLO: {args.weights}')
        self.yolo_model = YOLO(args.weights)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f'Usando device: {self.device}')
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        cv2.namedWindow('YOLO Inference - TurtleBot3', cv2.WINDOW_AUTOSIZE)
        self.get_logger().info('Aguardando imagens em /camera/image_raw ...')

    def image_callback(self, msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter imagem: {e}')
            return
        detections, _ = perform_yolo_inference(color_image, self.yolo_model, confidence_threshold=self.args.confidence_threshold)
        for detection in detections:
            x1, y1, x2, y2 = detection['bounding_box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = get_color(self.args.bbox_color)
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 3)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.args.font_scale, color, self.args.font_thickness)
        cv2.imshow('YOLO Inference - TurtleBot3', color_image)
        cv2.waitKey(1)

def main():
    args = parse_args()
    rclpy.init()
    node = YoloRosNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
