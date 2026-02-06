#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class PFClusterAvoidNode(Node):
    """
    /scan -> (x,y) points -> clustering -> nearest centroid -> Potential Field -> /cmd_vel
    (외부 라이브러리 없이 동작)
    """

    def __init__(self):
        super().__init__('pf_cluster_avoid_node')

        # Topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_topic', '/cmd_vel')

        # Range filter
        self.declare_parameter('range_min', 0.12)
        self.declare_parameter('range_max', 6.0)

        # ROI for obstacle search
        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 3.0)
        self.declare_parameter('roi_y_min', -1.5)
        self.declare_parameter('roi_y_max', 1.5)

        # Clustering (PCL Euclidean 개념)
        self.declare_parameter('cluster_tolerance', 0.20)
        self.declare_parameter('min_points', 6)
        self.declare_parameter('max_points', 600)

        # Potential Field params (사용자 코드 기반)
        self.declare_parameter('Kp_att', 0.6)          # (원본 0.02는 위치 업데이트용이라, 속도용으로는 너무 작아서 키움)
        self.declare_parameter('Kp_rep', 3.0)
        self.declare_parameter('obstacle_bound', 1.2)  # m

        # Virtual goal (base_link 기준)
        self.declare_parameter('goal_forward', 2.0)    # m  (전방으로 이만큼 가고 싶다)

        # Command scaling / limits
        self.declare_parameter('v_max', 0.22)          # TB3 권장 범위 근처
        self.declare_parameter('w_max', 1.5)
        self.declare_parameter('v_min', 0.0)

        # Safety stop
        self.declare_parameter('stop_dist', 0.28)      # m (너무 가까우면 정지+회전)

        scan_topic = self.get_parameter('scan_topic').value
        cmd_topic = self.get_parameter('cmd_topic').value

        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.pub = self.create_publisher(Twist, cmd_topic, 10)

        self.get_logger().info(f"PFClusterAvoidNode: {scan_topic} -> {cmd_topic}")

    # ---------------- Potential Field (사용자 코드 기반) ----------------
    def calc_attractive_force(self, x, y, gx, gy, Kp_att):
        e_x, e_y = gx - x, gy - y
        dist = float(np.hypot(e_x, e_y))
        dist = max(dist, 1e-6)
        att_x = Kp_att * e_x / dist
        att_y = Kp_att * e_y / dist
        return att_x, att_y

    def calc_repulsive_force(self, x, y, obs_xy_list, Kp_rep, obstacle_bound):
        rep_x, rep_y = 0.0, 0.0
        for ox, oy in obs_xy_list:
            dx, dy = ox - x, oy - y
            d = float(np.hypot(dx, dy))
            d = max(d, 1e-6)

            if d < obstacle_bound:
                # 사용자 식 그대로 (안정화 위해 d 최소값 처리만 추가)
                rep_x += -Kp_rep * (1.0 / d - 1.0 / obstacle_bound) * (1.0 / (d * d)) * (dx / d)
                rep_y += -Kp_rep * (1.0 / d - 1.0 / obstacle_bound) * (1.0 / (d * d)) * (dy / d)

        return rep_x, rep_y

    # ---------------- ROS callback ----------------
    def on_scan(self, msg: LaserScan):
        # LaserScan -> ranges/angles
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.shape[0], dtype=np.float32) * msg.angle_increment

        rmin = float(self.get_parameter('range_min').value)
        rmax = float(self.get_parameter('range_max').value)

        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        ranges = ranges[valid]
        angles = angles[valid]

        # 포인트가 너무 적으면 정지
        if ranges.size < 10:
            self.publish_cmd(0.0, 0.0)
            return

        # polar -> cartesian (base_scan/base_link 기준)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        # ROI
        x_min = float(self.get_parameter('roi_x_min').value)
        x_max = float(self.get_parameter('roi_x_max').value)
        y_min = float(self.get_parameter('roi_y_min').value)
        y_max = float(self.get_parameter('roi_y_max').value)

        roi = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        xs = xs[roi]
        ys = ys[roi]

        pts = np.stack([xs, ys], axis=1) if xs.size else np.zeros((0, 2), dtype=np.float32)

        # 클러스터링
        tol = float(self.get_parameter('cluster_tolerance').value)
        min_pts = int(self.get_parameter('min_points').value)
        max_pts = int(self.get_parameter('max_points').value)

        clusters = self.euclidean_clustering(pts, tol, min_pts, max_pts) if pts.shape[0] else []

        # 가장 가까운 클러스터 중심(장애물)
        nearest_centroid = None
        nearest_dist = 1e9

        for idxs in clusters:
            c = pts[idxs].mean(axis=0)
            d = float(np.hypot(c[0], c[1]))
            if d < nearest_dist:
                nearest_dist = d
                nearest_centroid = c

        # Potential Field 계산 (로봇 위치를 (0,0)으로 두고 힘만 계산)
        x, y = 0.0, 0.0
        goal_forward = float(self.get_parameter('goal_forward').value)
        gx, gy = goal_forward, 0.0  # 전방으로 가고 싶다

        Kp_att = float(self.get_parameter('Kp_att').value)
        Kp_rep = float(self.get_parameter('Kp_rep').value)
        obstacle_bound = float(self.get_parameter('obstacle_bound').value)

        att_x, att_y = self.calc_attractive_force(x, y, gx, gy, Kp_att)

        obs_list = []
        if nearest_centroid is not None:
            obs_list = [(float(nearest_centroid[0]), float(nearest_centroid[1]))]

        rep_x, rep_y = self.calc_repulsive_force(x, y, obs_list, Kp_rep, obstacle_bound)

        pot_x = att_x + rep_x
        pot_y = att_y + rep_y

        # 안전: 너무 가까우면 일단 정지 + 회전
        stop_dist = float(self.get_parameter('stop_dist').value)
        if nearest_centroid is not None and nearest_dist <= stop_dist:
            w = -1.0 if nearest_centroid[1] > 0.0 else 1.0  # 장애물 왼쪽이면 오른쪽 회전(음수)
            self.publish_cmd(0.0, self.clamp(w, -float(self.get_parameter('w_max').value), float(self.get_parameter('w_max').value)))
            return

        # pot 벡터 -> cmd_vel 변환
        v_max = float(self.get_parameter('v_max').value)
        w_max = float(self.get_parameter('w_max').value)
        v_min = float(self.get_parameter('v_min').value)

        # 방향(각속도): 목표 방향으로 향하도록
        heading = float(np.arctan2(pot_y, pot_x))  # [-pi, pi]
        w = 1.2 * heading  # gain

        # 속도: 전방 성분이 클수록 빠르게 (뒤로 가는 건 막음)
        v = float(np.hypot(pot_x, pot_y))
        v = min(v, v_max)
        v = max(v, v_min)

        # 뒤로 가는 상황 방지: pot_x가 음수면 일단 회전 위주
        if pot_x < 0.05:
            v *= 0.2

        w = self.clamp(w, -w_max, w_max)

        self.get_logger().info(
            f"pot=({pot_x:.2f},{pot_y:.2f}) v={v:.2f} w={w:.2f} "
            f"nearest={nearest_dist:.2f} y={nearest_centroid[1]:.2f}" if nearest_centroid is not None
            else f"pot=({pot_x:.2f},{pot_y:.2f}) v={v:.2f} w={w:.2f} nearest=None",
            throttle_duration_sec=1.0
        )

        self.publish_cmd(v, w)

    def publish_cmd(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub.publish(msg)

    def clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def euclidean_clustering(self, pts: np.ndarray, tol: float, min_pts: int, max_pts: int):
        """
        거리 tol 이내의 점을 연결 그래프로 보고 BFS로 군집 찾기.
        2D LiDAR(수백 포인트) 수업/실습용으로 충분히 빠름.
        """
        if pts is None or pts.shape[0] == 0:
            return []

        n = pts.shape[0]
        visited = np.zeros(n, dtype=bool)
        clusters = []
        tol2 = tol * tol

        for i in range(n):
            if visited[i]:
                continue

            queue = [i]
            visited[i] = True
            cluster = [i]

            while queue:
                cur = queue.pop()
                diff = pts - pts[cur]
                dist2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
                nbrs = np.where((dist2 <= tol2) & (~visited))[0]

                if nbrs.size:
                    visited[nbrs] = True
                    queue.extend(nbrs.tolist())
                    cluster.extend(nbrs.tolist())

                if len(cluster) > max_pts:
                    break

            if min_pts <= len(cluster) <= max_pts:
                clusters.append(np.array(cluster, dtype=np.int32))

        return clusters


def main():
    rclpy.init()
    node = PFClusterAvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
