#!/usr/bin/env python3
"""
Subscribes to realsense2_camera driver topics and provides:
  - /ur_suctionbot/camera/color/image_raw  
  - /ur_suctionbot/camera/depth/image_raw  
  - /ur_suctionbot/camera/depth/image_raw 
  - /ur_suctionbot/camera/save_frame       (service — saves .png + .npy pair)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs import msg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_srvs.srv import Trigger
import numpy as np
import cv2
import os


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Parameters
        self.declare_parameter('save_dir', '/data/captures')
        self.save_dir = self.get_parameter('save_dir').value
        self._frame_idx   = 0
        self._last_color  = None
        self._last_depth  = None

        # Subscribe to realsense2_camera driver topics
        self.sub_color = self.create_subscription(Image, '/camera/camera/color/image_raw', self._color_cb, 10)
        self.sub_depth = self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10)
        self.sub_cloud = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self._cloud_cb, 5)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self._info_cb, 1)

        # Republish 
        self.pub_color = self.create_publisher(Image, '/ur_suctionbot/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/ur_suctionbot/camera/depth/image_raw', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/ur_suctionbot/camera/points', 5)
        self.pub_info = self.create_publisher(CameraInfo, '/ur_suctionbot/camera/camera_info', 1)

        # Service to save a frame pair for offline suction viewer
        self.srv_save = self.create_service(Trigger, '/ur_suctionbot/camera/save_frame', self._save_frame_cb)

        self.get_logger().info('Camera node started')
        self.get_logger().info(f'Saving frames to: {self.save_dir}')

    def _color_cb(self, msg: Image):
        self.pub_color.publish(msg)
        # Convert to numpy for saving
        self._last_color = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

    def _depth_cb(self, msg: Image):
        self.pub_depth.publish(msg)
        # Convert to numpy for saving (uint16)
        self._last_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def _cloud_cb(self, msg: PointCloud2):
        self.pub_cloud.publish(msg)

    def _info_cb(self, msg: CameraInfo):
        self.pub_info.publish(msg)

    def _save_frame_cb(self, request, response):
        if self._last_color is None or self._last_depth is None:
            response.success = False
            response.message = 'No frames received yet'
            return response

        os.makedirs(self.save_dir, exist_ok=True)
        stem = f'frame_{self._frame_idx:04d}'

        # Save RGB as PNG, depth as npy
        cv2.imwrite(os.path.join(self.save_dir, f'{stem}.png'),self._last_color)
        np.save(os.path.join(self.save_dir, f'{stem}.npy'), self._last_depth)

        self._frame_idx += 1
        response.success = True
        response.message = f'Saved {stem} to {self.save_dir}'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
