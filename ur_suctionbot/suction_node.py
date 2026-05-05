#!/usr/bin/env python3
'''
Subscribes to the camera topics and computes suction scores using:
  - KNN-based normal estimation + local flatness (std) scoring
  - RANSAC-based plane fitting + inlier ratio scoring
  - Sobel filter-based normal estimation + local flatness scoring   

Subs:
  /ur_suctionbot/camera/depth/image_raw  
  /ur_suctionbot/camera/color/image_raw
  /ur_suctionbot/camera/color/camera_info

Pubs:
  /ur_suctionbot/suction/scores    (ur_suctionbot/SuctionScore)
  /ur_suctionbot/suction/best      (ur_suctionbot/GraspCandidate)

Services:
  /ur_suctionbot/suction/compute   (ur_suctionbot/ComputeSuction)
'''

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import struct
from geometry_msgs.msg import Point, PoseStamped, Quaternion
#from cv_bridge import CvBridge
import numpy as np

from ur_suctionbot.msg import SuctionScore, GraspCandidate
from ur_suctionbot.srv import ComputeSuction
from ur_suctionbot.knn import score_knn
from ur_suctionbot.sobel import score_sobel
from ur_suctionbot.ransac import score_ransac


class SuctionNode(Node):
    def __init__(self):
        super().__init__('suction_node')

        # Parameters
        self.declare_parameter('method',          'knn')
        self.declare_parameter('threshold',        0.5)
        self.declare_parameter('knn_k',            30)
        self.declare_parameter('std_win',          25)
        self.declare_parameter('cup_diameter_mm',  30.0)
        self.declare_parameter('ransac_iters',     50)
        self.declare_parameter('ransac_tol_mm',    3.0)
        self.declare_parameter('depth_scale',      0.001)
        self._load_params()
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # self._bridge      = CvBridge()
        self._last_depth  = None
        self._last_header = None
        self._last_color   = None

        # Subscriptions
        self.sub_depth = self.create_subscription(Image, '/ur_suctionbot/camera/depth/image_raw', self._depth_cb, 10)
        self.sub_color = self.create_subscription(Image, '/ur_suctionbot/camera/color/image_raw', self._color_cb, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/ur_suctionbot/camera/camera_info', self._info_cb, 1)

        # Publishers
        self.pub_scores = self.create_publisher(SuctionScore, '/ur_suctionbot/suction/scores', 5)
        self.pub_best = self.create_publisher(GraspCandidate, '/ur_suctionbot/suction/best', 5)
        self.pub_viz = self.create_publisher(PointCloud2, '/ur_suctionbot/suction/visualization', 5)

        # Service
        self.srv_compute = self.create_service(ComputeSuction, '/ur_suctionbot/suction/compute', self._compute_cb)

        self.get_logger().info(f'Suction node ready [method={self.method}]')

    
    def _load_params(self):
        self.method        = self.get_parameter('method').value
        self.threshold     = self.get_parameter('threshold').value
        self.knn_k         = self.get_parameter('knn_k').value
        self.std_win       = self.get_parameter('std_win').value
        self.cup_mm        = self.get_parameter('cup_diameter_mm').value
        self.ransac_iters  = self.get_parameter('ransac_iters').value
        self.ransac_tol_mm = self.get_parameter('ransac_tol_mm').value
        self.depth_scale   = self.get_parameter('depth_scale').value

    def _info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.cx = msg.k[2]
        self.fy = msg.k[4]
        self.cy = msg.k[5]
        self.get_logger().info(f'Camera intrinsics loaded: fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f}', once=True)

    def _depth_cb(self, msg: Image):
        self._last_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self._last_header = msg.header

    def _color_cb(self, msg: Image):
        #self._last_color = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return

    def backproject(self, depth: np.ndarray):
        """Convert depth image to 3D point cloud."""
        H, W  = depth.shape
        d     = depth.astype(np.float64) * self.depth_scale
        valid = (d > 0) & np.isfinite(d)

        v_idx, u_idx = np.where(valid)
        z = d[valid]
        x = (u_idx - self.cx) * z / self.fx
        y = (v_idx - self.cy) * z / self.fy

        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        return xyz, u_idx.astype(np.int32), v_idx.astype(np.int32), H, W
    
    def _compute_win(self, xyz: np.ndarray) -> int:
        """Calculate std filter window in pixels.
    
           If std_win > 0: use manual window size directly.
           If std_win == 0: calculate from cup size and median depth. """
        if self.std_win > 0:
            return self.std_win | 1  # manual window, odd 
    
        if self.cup_mm > 0 and xyz is not None and len(xyz) > 0:
            median_depth = float(np.median(xyz[:, 2]))
            if median_depth > 0:
                win_px = int(round((self.cup_mm / 1000.0) * self.fx / median_depth))
                win_px = max(3, win_px | 1)   # minimum 3, must be odd
                win_px = min(101, win_px)      # cap at 101
                self.get_logger().debug(
                    f'Cup {self.cup_mm}mm @ {median_depth:.2f}m = {win_px}px window')
                return win_px
        return 25  # fallback to default 25px window

    def compute_scores(self, method):
        xyz, u, v, H, W = self.backproject(self._last_depth)
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        win = self._compute_win(xyz)

        if method == 'knn':
            return score_knn(xyz, u, v, H, W, self.knn_k, win, approach), xyz, u, v

        elif method == 'sobel':
            return score_sobel(xyz, u, v, H, W, win, approach), xyz, u, v

        elif method == 'ransac':
            cup = self.cup_mm if self.cup_mm > 0 else None
            return score_ransac(xyz, u, v, H, W, win, cup, self.ransac_iters, self.ransac_tol_mm, approach, self.fx), xyz, u, v

        raise ValueError(f'Unknown method: {method}')



    def _compute_cb(self, request, response):
        # Compute suction scores based on the latest depth and color images
        # Publish scores and best grasp candidate
        if self._last_depth is None:
            response.success = False
            response.message = 'No depth frame received yet'
            return response

        if self.fx is None:
            response.success = False
            response.message = 'Waiting for camera intrinsics'
            return response

        method = request.method or self.method
        threshold = request.threshold or self.threshold

        try:
            scores, xyz, u, v = self.compute_scores(method)
            best_idx = int(np.argmax(scores))
            best_pt  = xyz[best_idx]

            # SuctionScore message
            score_msg = SuctionScore()
            score_msg.header = self._last_header
            score_msg.scores = scores.tolist()
            score_msg.method = method
            score_msg.threshold = threshold
            score_msg.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in xyz]

            # GraspCandidate message
            grasp = GraspCandidate()
            grasp.header = self._last_header
            grasp.score = float(scores[best_idx])
            grasp.method = method
            grasp.cup_diameter_mm = self.cup_mm
            grasp.pose.header = self._last_header
            grasp.pose.pose.position = Point(
                x=float(best_pt[0]),
                y=float(best_pt[1]),
                z=float(best_pt[2]))
            grasp.pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=0.0, w=1.0)

            self.pub_scores.publish(score_msg)
            self.pub_best.publish(grasp)
            self.publish_visualization(scores, xyz, self._last_header)

            response.success        = True
            response.message        = f'Best score: {grasp.score:.3f}'
            response.best_candidate = grasp
            #response.all_scores     = score_msg

        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f'Scoring failed: {e}')

        return response
    

    def _score_to_rgb(self, t: float):
        """Map score 0-1 to BGR color. Blue=low, red=high."""
        t = max(0.0, min(1.0, t))
        r = max(0.0, min(1.0, 1.5 - abs(t - 0.75) * 4))
        g = max(0.0, min(1.0, 1.5 - abs(t - 0.50) * 4))
        b = max(0.0, min(1.0, 1.5 - abs(t - 0.25) * 4))
        return int(r * 255), int(g * 255), int(b * 255)


    def publish_visualization(self, scores: np.ndarray, xyz: np.ndarray, header):
        """Publish scored point cloud as PointCloud2 for RViz2."""
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        point_step = 16
        data = []

        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = self._score_to_rgb(float(scores[i]))
            rgb_int = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
            data.append(struct.pack('ffff', x, y, z, rgb_float))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width  = len(xyz)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step   = point_step
        msg.row_step     = point_step * len(xyz)
        msg.data         = b''.join(data)
        msg.is_dense     = True

        self.pub_viz.publish(msg)


    


def main(args=None):
    rclpy.init(args=args)
    node = SuctionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()