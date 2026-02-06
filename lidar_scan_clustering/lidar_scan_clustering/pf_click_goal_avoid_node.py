#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist


def yaw_from_quat(qx, qy, qz, qw):
    # yaw only
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class PFClickGoalAvoid(Node):
    """
    Inputs:
      - /scan (LaserScan)
      - /odom (Odometry)
      - /goal_pose (PoseStamped)  <-- RViz2 "2D Nav Goal"
    Output:
      - /cmd_vel (Twist)

    Potential Field:
      Attractive force to goal (global)
      Repulsive force from nearest LiDAR cluster centroid (local obstacle)
    """

    def __init__(self):
        super().__init__('pf_click_goal_avoid')

        # topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_topic', '/cmd_vel')

        # scan filter + ROI (robot frame)
        self.declare_parameter('range_min', 0.12)
        self.declare_parameter('range_max', 6.0)
        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 3.0)
        self.declare_parameter('roi_y_min', -1.5)
        self.declare_parameter('roi_y_max', 1.5)

        # clustering (PCL Euclidean concept)
        self.declare_parameter('cluster_tolerance', 0.20)
        self.declare_parameter('min_points', 6)
        self.declare_parameter('max_points', 700)

        # potential field gains (속도 제어용 스케일)
        self.declare_parameter('Kp_att', 1.0)
        self.declare_parameter('Kp_rep', 1.0)
        self.declare_parameter('Kp_rep_limit', 0.8)
        self.declare_parameter('obstacle_bound', 2.0)

        # control
        self.declare_parameter('v_max', 0.22)
        self.declare_parameter('w_max', 1.5)
        self.declare_parameter('heading_gain', 1.5)   # w = gain * heading_error
        self.declare_parameter('stop_dist', 0.28)      # m (너무 가까우면 정지+회전)
        self.declare_parameter('goal_tolerance', 0.25) # m (목표 도착 판정)

        # internal state
        self.odom_ok = False
        self.goal_ok = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.gx = 0.0
        self.gy = 0.0

        # latest nearest obstacle centroid in robot frame (base_link-ish)
        self.nearest_obs_robot = None
        self.nearest_obs_dist = 1e9

        # subs/pubs
        self.sub_scan = self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self.on_scan, 10)
        self.sub_odom = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.on_odom, 10)
        self.sub_goal = self.create_subscription(
            PoseStamped, self.get_parameter('goal_topic').value, self.on_goal, 10)

        self.pub_cmd = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)

        # control timer (scan/odom/goal 비동기라 주기적으로 계산)
        self.timer = self.create_timer(0.05, self.control_step)  # 20Hz

        self.get_logger().info("PFClickGoalAvoid ready. Use RViz2 '2D Nav Goal' to set /goal_pose.")

    # ---------------- callbacks ----------------
    def on_goal(self, msg: PoseStamped):
        self.gx = float(msg.pose.position.x)
        self.gy = float(msg.pose.position.y)
        self.goal_ok = True
        self.get_logger().info(f"New goal: ({self.gx:.2f}, {self.gy:.2f})")

    def on_odom(self, msg: Odometry):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.odom_ok = True

    def on_scan(self, msg: LaserScan):
        # scan -> points in robot frame
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.shape[0], dtype=np.float32) * msg.angle_increment

        rmin = float(self.get_parameter('range_min').value)
        rmax = float(self.get_parameter('range_max').value)

        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        ranges = ranges[valid]
        angles = angles[valid]

        if ranges.size < 10:
            self.nearest_obs_robot = None
            self.nearest_obs_dist = 1e9
            return

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
        if xs.size < 10:
            self.nearest_obs_robot = None
            self.nearest_obs_dist = 1e9
            return

        pts = np.stack([xs, ys], axis=1)

        clusters = self.euclidean_clustering(
            pts,
            float(self.get_parameter('cluster_tolerance').value),
            int(self.get_parameter('min_points').value),
            int(self.get_parameter('max_points').value)
        )

        nearest_c = None
        nearest_d = 1e9
        for idxs in clusters:
            c = pts[idxs].mean(axis=0)
            d = float(np.hypot(c[0], c[1]))
            if d < nearest_d:
                nearest_d = d
                nearest_c = c

        self.nearest_obs_robot = None if nearest_c is None else (float(nearest_c[0]), float(nearest_c[1]))
        self.nearest_obs_dist = nearest_d

    # ---------------- control ----------------
    def control_step(self):
        # need odom + goal
        if not (self.odom_ok and self.goal_ok):
            return

        # goal reached?
        goal_err = float(np.hypot(self.gx - self.x, self.gy - self.y))
        if goal_err <= float(self.get_parameter('goal_tolerance').value):
            self.publish_cmd(0.0, 0.0)
            return

        # Attractive force in GLOBAL frame (toward goal)
        Kp_att = float(self.get_parameter('Kp_att').value)
        att_x, att_y = self.calc_attractive_force(self.x, self.y, self.gx, self.gy, Kp_att)

        # Repulsive force from nearest obstacle centroid
        # obstacle centroid is in ROBOT frame -> convert to GLOBAL using odom pose
        rep_x, rep_y = 0.0, 0.0
        if self.nearest_obs_robot is not None:
            ox_r, oy_r = self.nearest_obs_robot

            # robot frame -> global frame
            ox_g = self.x + np.cos(self.yaw) * ox_r - np.sin(self.yaw) * oy_r
            oy_g = self.y + np.sin(self.yaw) * ox_r + np.cos(self.yaw) * oy_r

            Kp_rep = float(self.get_parameter('Kp_rep').value)
            obstacle_bound = float(self.get_parameter('obstacle_bound').value)
            rep_x, rep_y = self.calc_repulsive_force(self.x, self.y, [(ox_g, oy_g)], Kp_rep, obstacle_bound)

            # safety: too close -> stop + turn away
            if self.nearest_obs_dist <= float(self.get_parameter('stop_dist').value):
                w = -1.0 if oy_r > 0.0 else 1.0
                self.publish_cmd(0.0, self.clamp(w, -float(self.get_parameter('w_max').value), float(self.get_parameter('w_max').value)))
                return

        # Potential vector in GLOBAL frame
        pot_x = att_x + rep_x
        pot_y = att_y + rep_y

        # Convert to ROBOT frame for heading control
        # v uses forward component; w uses heading error
        pot_rx = np.cos(self.yaw) * pot_x + np.sin(self.yaw) * pot_y
        pot_ry = -np.sin(self.yaw) * pot_x + np.cos(self.yaw) * pot_y

        heading = float(np.arctan2(pot_ry, pot_rx))  # desired heading in robot frame

        # speed commands
        v_max = float(self.get_parameter('v_max').value)
        w_max = float(self.get_parameter('w_max').value)
        heading_gain = float(self.get_parameter('heading_gain').value)

        # forward speed: pot 벡터 크기 기반, 뒤로는 제한
        v = float(np.hypot(pot_rx, pot_ry))
        v = min(v, v_max)
        if pot_rx < 0.05:
            v *= 0.2  # 거의 옆/뒤 방향이면 느리게

        w = heading_gain * heading
        w = self.clamp(w, -w_max, w_max)

        self.publish_cmd(v, w)

    def publish_cmd(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.pub_cmd.publish(msg)

    # ---------------- potential field math (user code 기반) ----------------
    def calc_attractive_force(self, x, y, gx, gy, Kp_att):
        ex, ey = gx - x, gy - y
        d = float(np.hypot(ex, ey))
        d = max(d, 1e-6)
        return (Kp_att * ex / d, Kp_att * ey / d)

    def calc_repulsive_force(self, x, y, obs_list, Kp_rep, obstacle_bound):
        rep_x, rep_y = 0.0, 0.0
        for ox, oy in obs_list:
            dx, dy = ox - x, oy - y
            d = float(np.hypot(dx, dy))
            d = max(d, 1e-6)
            if d < obstacle_bound:
                rep_x += -Kp_rep * (1.0 / d - 1.0 / obstacle_bound) * (1.0 / (d * d)) * (dx / d)
                rep_y += -Kp_rep * (1.0 / d - 1.0 / obstacle_bound) * (1.0 / (d * d)) * (dy / d)

            Kp_rep_limit = float(self.get_parameter('Kp_rep_limit').value)

            if rep_x > Kp_rep_limit:
                rep_x = Kp_rep_limit
            elif rep_x < -Kp_rep_limit:
                rep_x = -Kp_rep_limit
            else:
                rep_x = rep_x     

            if rep_y > Kp_rep_limit:
                rep_y = Kp_rep_limit
            elif rep_y < -Kp_rep_limit:
                rep_y = -Kp_rep_limit
            else:
                rep_y = rep_y

        return rep_x, rep_y

    # ---------------- clustering (Euclidean, BFS) ----------------
    def euclidean_clustering(self, pts: np.ndarray, tol: float, min_pts: int, max_pts: int):
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

    def clamp(self, x, lo, hi):
        return max(lo, min(hi, x))


def main():
    rclpy.init()
    node = PFClickGoalAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_cmd(0.0, 0.0)
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
