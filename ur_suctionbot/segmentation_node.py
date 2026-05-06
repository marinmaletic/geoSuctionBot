#!/usr/bin/env python3
"""
segmentation_node

Gemini detection + SAM2 segmentation pipeline.
Subscribes to camera topics, on service call runs Gemini to detect objects,
SAM2 to refine masks, then publishes masked points and visualization.

Subscribes:
  /ur_suctionbot/camera/color/image_raw   (sensor_msgs/Image)
  /ur_suctionbot/camera/depth/image_raw   (sensor_msgs/Image)
  /ur_suctionbot/camera/camera_info       (sensor_msgs/CameraInfo)

Publishes:
  /ur_suctionbot/segmentation/overlay     (sensor_msgs/Image)  BGR visualization
  /ur_suctionbot/segmentation/points      (sensor_msgs/PointCloud2)  masked points only

Services:
  /ur_suctionbot/segmentation/segment     (ur_suctionbot/srv/Segment)
"""

import os
import json
import struct
import threading

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header

from ur_suctionbot.srv import Segment


# SAM2 checkpoint path
SAM2_CHECKPOINT = os.path.expanduser('~/sam2/checkpoints/sam2.1_hiera_tiny.pt')
SAM2_CFG        = 'configs/sam2.1/sam2.1_hiera_t.yaml'

COLORS_BGR = [
    (0, 220, 80),
    (255, 200, 0),
    (0, 200, 255),
    (255, 100, 0),
    (200, 0, 255),
    (0, 255, 200),
]


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        # Parameters
        self.declare_parameter('sam2_checkpoint', SAM2_CHECKPOINT)
        self.declare_parameter('sam2_cfg',        SAM2_CFG)
        self.declare_parameter('gemini_model',    'gemini-robotics-er-1.6-preview')     # "gemini-3-flash-preview"
        self.declare_parameter('depth_scale',     0.001)
        self.declare_parameter('gemini_prompt',
            'Detect all aseptic beverage containers / tetrapaks in this image, including crushed or deformed ones. Be sure to differentiate between cartons and bottles, especially white. '
            'Return ONLY a JSON array: '
            '[{"label": "object", "confidence": 0.0, "box_2d": [y0, x0, y1, x1]}] '
            'box_2d must be normalized integers 0-1000. '
            'If nothing found return: []')

        self.checkpoint  = self.get_parameter('sam2_checkpoint').value
        self.sam2_cfg    = self.get_parameter('sam2_cfg').value
        self.gemini_model = self.get_parameter('gemini_model').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.prompt      = self.get_parameter('gemini_prompt').value

        # State
        self._last_color  = None
        self._last_depth  = None
        self._last_header = None
        self._fx = self._fy = self._cx = self._cy = None
        self._predictor   = None
        self._gemini      = None
        self._lock        = threading.Lock()

        # Subscriptions
        self.create_subscription(Image,      '/ur_suctionbot/camera/color/image_raw', self._color_cb, 5)
        self.create_subscription(Image,      '/ur_suctionbot/camera/depth/image_raw', self._depth_cb, 5)
        self.create_subscription(CameraInfo, '/ur_suctionbot/camera/camera_info',     self._info_cb,  1)

        # Publishers
        self.pub_overlay = self.create_publisher(Image,       '/ur_suctionbot/segmentation/overlay', 5)
        self.pub_points  = self.create_publisher(PointCloud2, '/ur_suctionbot/segmentation/points',  5)

        # Service
        self.create_service(Segment, '/ur_suctionbot/segmentation/segment', self._segment_cb)

        self.get_logger().info('Segmentation node ready — loading models lazily on first call')

    # ── Subscribers ────────────────────────────────────────────────────────────

    def _color_cb(self, msg: Image):
        self._last_color  = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self._last_header = msg.header

    def _depth_cb(self, msg: Image):
        self._last_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def _info_cb(self, msg: CameraInfo):
        self._fx = msg.k[0]
        self._cx = msg.k[2]
        self._fy = msg.k[4]
        self._cy = msg.k[5]

    # ── Lazy model loading ─────────────────────────────────────────────────────

    def _load_models(self):
        """Load SAM2 and Gemini on first service call to avoid slow startup."""
        if self._predictor is not None:
            return

        self.get_logger().info('Loading SAM2...')
        import torch
        import sys
        sys.modules.setdefault('open3d.ml', type(sys)('open3d.ml'))

        # SAM2 needs to be loaded from its own directory
        sam2_dir = os.path.expanduser('~/sam2')
        orig_dir = os.getcwd()
        os.chdir(sam2_dir)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model  = build_sam2(self.sam2_cfg, self.checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(model)
        os.chdir(orig_dir)
        self.get_logger().info(f'SAM2 ready on {device}')

        self.get_logger().info('Loading Gemini...')
        from google import genai
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY environment variable not set')
        self._gemini = genai.Client(api_key=api_key)
        self.get_logger().info('Gemini ready')

    # ── Gemini detection ───────────────────────────────────────────────────────

    def _gemini_detect(self, img_rgb: np.ndarray) -> tuple[list[dict], list[tuple]]:
        """Run Gemini on RGB image, return (detections, pixel_boxes)."""
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_rgb)
        H, W = img_rgb.shape[:2]

        response = self._gemini.models.generate_content(
            model=self.gemini_model,
            contents=[self.prompt, pil_img],
        )

        detections = self._parse_gemini(response.text)
        boxes_px   = []
        for d in detections:
            box = d.get('box_2d')
            if not box or len(box) != 4:
                continue
            y0n, x0n, y1n, x1n = box
            x0 = int(x0n / 1000 * W)
            y0 = int(y0n / 1000 * H)
            x1 = int(x1n / 1000 * W)
            y1 = int(y1n / 1000 * H)
            boxes_px.append((x0, y0, x1, y1))

        return detections, boxes_px

    def _parse_gemini(self, text: str) -> list[dict]:
        try:
            text = text.strip()
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            result = json.loads(text.strip())
            return result if isinstance(result, list) else []
        except Exception as e:
            self.get_logger().warn(f'Gemini parse error: {e}')
            return []

    # ── SAM2 segmentation ──────────────────────────────────────────────────────

    def _sam2_segment(self, img_rgb: np.ndarray, boxes: list[tuple]) -> list[np.ndarray]:
        """Run SAM2 on each box, return list of boolean masks."""
        if not boxes:
            return []

        self._predictor.set_image(img_rgb)
        masks = []
        for box in boxes:
            x0, y0, x1, y1 = box
            m, scores, _ = self._predictor.predict(
                box=np.array([[x0, y0, x1, y1]], dtype=np.float32),
                multimask_output=True,
            )
            masks.append(m[int(np.argmax(scores))].astype(bool))
        return masks

    # ── Combined mask ──────────────────────────────────────────────────────────

    def _combine_masks(self, masks: list[np.ndarray], H: int, W: int) -> np.ndarray:
        """Union of all masks into a single boolean mask."""
        combined = np.zeros((H, W), dtype=bool)
        for m in masks:
            combined |= m
        return combined

    # ── Visualization ──────────────────────────────────────────────────────────

    def build_overlay(
        self,
        img_rgb: np.ndarray,
        detections: list[dict],
        boxes_px: list[tuple],
        masks: list[np.ndarray],
    ) -> np.ndarray:
        """Draw Gemini bboxes and SAM2 masks on image, return BGR."""
        H, W = img_rgb.shape[:2]
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

        # Draw semi-transparent mask overlays
        overlay = vis.copy()
        for i, mask in enumerate(masks):
            color = COLORS_BGR[i % len(COLORS_BGR)]
            overlay[mask] = color
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

        # Draw contours and bounding boxes
        for i, (det, box) in enumerate(zip(detections, boxes_px)):
            color = COLORS_BGR[i % len(COLORS_BGR)]
            x0, y0, x1, y1 = box

            # Bounding box
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            # Label
            label = f"{det.get('label','?')} {det.get('confidence', 0):.2f}"
            cv2.putText(vis, label, (x0, max(y0 - 8, 12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw mask contours
        for i, mask in enumerate(masks):
            color = COLORS_BGR[i % len(COLORS_BGR)]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

        return vis

    # ── Point cloud (masked) ───────────────────────────────────────────────────

    def build_masked_pointcloud(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        header: Header,
    ) -> PointCloud2:
        """Backproject only masked depth pixels into a PointCloud2."""
        d     = depth.astype(np.float64) * self.depth_scale
        valid = mask & (d > 0) & np.isfinite(d)

        v_idx, u_idx = np.where(valid)
        z = d[valid]
        x = (u_idx - self._cx) * z / self._fx
        y = (v_idx - self._cy) * z / self._fy

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='u', offset=12, datatype=PointField.INT32,   count=1),
            PointField(name='v', offset=16, datatype=PointField.INT32,   count=1),
        ]

        data = bytearray()
        for xi, yi, zi, ui, vi in zip(x, y, z, u_idx, v_idx):
            data += struct.pack('fffii', float(xi), float(yi), float(zi), int(ui), int(vi))

        msg = PointCloud2()
        msg.header     = header
        msg.height     = 1
        msg.width      = len(x)
        msg.fields     = fields
        msg.is_bigendian = False
        msg.point_step = 20
        msg.row_step   = 20 * len(x)
        msg.data       = bytes(data)
        msg.is_dense   = True
        return msg

    # ── Service callback ───────────────────────────────────────────────────────

    def _segment_cb(self, request, response):
        if self._last_color is None or self._last_depth is None:
            response.success = False
            response.message = 'No frames received yet'
            return response

        if self._fx is None:
            response.success = False
            response.message = 'Waiting for camera intrinsics'
            return response

        try:
            # Load models on first call
            self._load_models()

            img_rgb = self._last_color.copy()
            depth   = self._last_depth.copy()
            header  = self._last_header
            H, W    = img_rgb.shape[:2]

            # Use custom prompt if provided
            if request.prompt:
                self.prompt = request.prompt

            self.get_logger().info('Running Gemini detection...')
            detections, boxes_px = self._gemini_detect(img_rgb)
            self.get_logger().info(f'Gemini found {len(detections)} object(s)')

            self.get_logger().info('Running SAM2 segmentation...')
            masks = self._sam2_segment(img_rgb, boxes_px)
            self.get_logger().info(f'SAM2 produced {len(masks)} mask(s)')

            # Combined mask
            combined_mask = self._combine_masks(masks, H, W)

            # Build and publish overlay image
            overlay_bgr = self.build_overlay(img_rgb, detections, boxes_px, masks)
            overlay_msg = Image()
            overlay_msg.header   = header
            overlay_msg.height   = H
            overlay_msg.width    = W
            overlay_msg.encoding = 'bgr8'
            overlay_msg.step     = W * 3
            overlay_msg.data     = overlay_bgr.tobytes()
            self.pub_overlay.publish(overlay_msg)

            # Build and publish masked point cloud
            if combined_mask.any():
                pc_msg = self.build_masked_pointcloud(depth, combined_mask, header)
                self.pub_points.publish(pc_msg)
                self.get_logger().info(f'Published {pc_msg.width} masked points')
            else:
                self.get_logger().warn('No valid mask pixels — not publishing points')

            response.success     = True
            response.message     = f'Detected {len(detections)} object(s), masked {combined_mask.sum()} pixels'
            response.num_objects = len(detections)

        except Exception as e:
            import traceback
            traceback.print_exc()
            response.success = False
            response.message = str(e)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
