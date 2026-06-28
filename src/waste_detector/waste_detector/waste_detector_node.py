#!/usr/bin/env python3
"""
Tópicos publicados:
  /waste_detector/detections  (vision_msgs/Detection2DArray)
  /waste_detector/position    (geometry_msgs/PointStamped) — objeto mais próximo
"""

import sys
sys.path.insert(0, '/home/lauraluna/FusionVision')

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

import numpy as np
from cv_bridge import CvBridge
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from ultralytics import YOLO
import open3d as o3d
import cv2


class WasteDetectorNode(Node):

    def __init__(self):
        super().__init__('waste_detector')

        # ── configs via ROS2 ─────────────────────────────
        self.declare_parameter('model_path',
            '/home/lauraluna/FusionVision/runs/detect/runs/train/waste_yolov8n_combined_v2/weights/best.pt')
        self.declare_parameter('conf_threshold', 0.6)

        model_path    = self.get_parameter('model_path').value
        self.conf_thr = self.get_parameter('conf_threshold').value

        
        self.FX, self.FY = 615.0, 615.0
        self.CX, self.CY = 320.0, 240.0
        self.W,  self.H  = 640, 480

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.W, self.H, self.FX, self.FY, self.CX, self.CY)

    
        self.bridge            = CvBridge()
        self.latest_depth_img  = None

       
        self.get_logger().info(f'Carregando YOLO: {model_path}')
        self.yolo    = YOLO(model_path)
        self.fastsam = FastSAM('/home/lauraluna/FusionVision/FastSAM-x.pt')
        self.get_logger().info('Modelos carregados.')

        # publishers ─────────────────────────────────────────────────────
        self.pub_detections = self.create_publisher(
            Detection2DArray, '/waste_detector/detections', 10)
        self.pub_position = self.create_publisher(
            PointStamped, '/waste_detector/position', 10)

        # ── subscribers ────────────────────────────────────────────────────
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.sub_rgb = self.create_subscription(
            Image, '/camera/image', self.rgb_callback, 10)

        self.get_logger().info('Nó Live do FusionVision iniciado e aguardando tópicos...')
        self.create_timer(2.0, lambda: self.get_logger().info(
            f'Status: depth_recebido={self.latest_depth_img is not None}'))

   
    def depth_callback(self, msg):
        self.get_logger().info(
            f'Depth recebido! encoding={msg.encoding} size={msg.width}x{msg.height}',
            throttle_duration_sec=2.0)
        try:
            if msg.encoding == '32FC1':
                self.latest_depth_img = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='32FC1')
            else:
                cv_depth = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='passthrough')
                self.latest_depth_img = cv_depth.astype(np.float32) / 1000.0
        except Exception as e:
            self.get_logger().error(f'Erro ao converter profundidade: {e}')


    def rgb_callback(self, msg):
        if self.latest_depth_img is None:
            self.get_logger().warning(
                'RGB recebido, mas aguardando primeiro frame de profundidade...',
                throttle_duration_sec=2.0)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Erro ao converter imagem RGB: {e}')
            return

        
        yolo_res = self.yolo(
            cv_image, device='cpu', conf=self.conf_thr, verbose=False, imgsz=640)
        CLASSES_ALVO = {'PLASTIC', 'GLASS', 'METAL', 'CARDBOARD', 'BIODEGRADABLE'}
        boxes = [b for b in yolo_res[0].boxes
                 if yolo_res[0].names[int(b.cls[0])] in CLASSES_ALVO]

        annotated_frame = yolo_res[0].plot()
        cv2.imshow('Visao do Robo - FusionVision', annotated_frame)
        cv2.waitKey(1)

        if len(boxes) == 0:
            return

        orig_h, orig_w = cv_image.shape[:2]
        sx = self.W / orig_w
        sy = self.H / orig_h

        bounding_boxes = []
        box_meta       = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            x1, x2 = max(0, x1), min(self.W, x2)
            y1, y2 = max(0, y1), min(self.H, y2)
            bounding_boxes.append([x1, y1, x2, y2])
            box_meta.append({
                'cls_name': yolo_res[0].names[int(box.cls[0])],
                'conf':     float(box.conf[0]),
                'bbox':     (x1, y1, x2, y2),
            })


        sam_masks = None
        try:
            fsam_res = self.fastsam(
                cv_image, device='cpu', retina_masks=True,
                verbose=False, imgsz=640, conf=0.4, iou=0.9)

            if fsam_res and fsam_res[0].masks is not None:
                prompt = FastSAMPrompt(cv_image, fsam_res, device='cpu')
                ann = prompt.box_prompt(bboxes=bounding_boxes)  
                sam_masks = np.array(ann).astype(np.uint8)
                self.get_logger().info(
                    f'YOLO: {len(boxes)} obj | FastSAM: {len(sam_masks)} máscaras')
            else:
                self.get_logger().warning(
                    'FastSAM não gerou máscaras — usando bbox como fallback.',
                    throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f'Erro no FastSAM: {e}')


        np_depth = np.array(self.latest_depth_img)

        header            = Header(frame_id='camera_link')
        header.stamp      = msg.header.stamp
        det_array         = Detection2DArray()
        det_array.header  = header

        closest_z   = float('inf')
        closest_pt  = None
        closest_cls = ''

        kernel = np.ones((20, 20), np.uint8)  

        for i, meta in enumerate(box_meta):
            x1, y1, x2, y2 = meta['bbox']

 
            if sam_masks is not None and i < len(sam_masks):
                mask = cv2.erode(sam_masks[i], kernel, iterations=1)
                roi  = np.where(
                    (mask > 0) & (~np.isnan(np_depth)) & (np_depth > 0) & (np_depth < 5.0),
                    np_depth,
                    np.nan)
            else:
                roi_raw = np_depth[y1:y2, x1:x2]
                roi     = np.where(
                    ~np.isnan(roi_raw) & (roi_raw > 0) & (roi_raw < 5.0),
                    roi_raw,
                    np.nan)

            valid = roi[~np.isnan(roi)]
            if len(valid) == 0:
                continue

            z   = float(np.median(valid))
            x_m = ((x1 + x2) / 2 - self.CX) * z / self.FX
            y_m = ((y1 + y2) / 2 - self.CY) * z / self.FY

            self.get_logger().info(
                f'  [{meta["cls_name"]} {meta["conf"]:.2f}] XYZ=({x_m:.3f}, {y_m:.3f}, {z:.3f}) m')

            det        = Detection2D()
            det.header = header
            hyp        = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id  = meta['cls_name']
            hyp.hypothesis.score     = meta['conf']
            hyp.pose.pose.position.x = x_m
            hyp.pose.pose.position.y = y_m
            hyp.pose.pose.position.z = z
            det.results.append(hyp)
            det_array.detections.append(det)

            if z < closest_z:
                closest_z   = z
                closest_pt  = (x_m, y_m, z)
                closest_cls = meta['cls_name']

      
        if det_array.detections:
            self.pub_detections.publish(det_array)
            self.get_logger().info(
                f'  Publicado: {len(det_array.detections)} detecção(ões)')

        if closest_pt:
            pt_msg         = PointStamped()
            pt_msg.header  = header
            pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = closest_pt
            self.pub_position.publish(pt_msg)
            self.get_logger().info(
                f'  Mais próximo: {closest_cls} @ Z={closest_z:.3f} m')


def main(args=None):
    rclpy.init(args=args)
    node = WasteDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()