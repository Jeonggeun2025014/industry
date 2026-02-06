#!/usr/bin/env python3
import heapq
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist


def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class PlannerPFAStar(Node):
    """
    RViz2 2D Nav Goal(/goal_pose) + /odom + /scan 기반
    mode = "pf" or "astar"
      - pf    : Potential Field(Attractive + Repulsive)
      - astar : /scan로 만든 로컬 점유그리드에서 A* + 간단 경로추종
    """

    def __init__(self):
        super().__init__('planner_pf_astar_node')

        # Topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_topic', '/cmd_vel')

        # Mode
        self.declare_parameter('mode', 'pf')  # 'pf' or 'astar'

        # Scan filters
        self.declare_parameter('range_min', 0.12)
        self.declare_parameter('range_max', 6.0)
        self.declare_parameter('roi_x_min', -0.5)  # astar는 로봇 주변도 필요해서 약간 뒤도 포함
        self.declare_parameter('roi_x_max', 4.0)
        self.declare_parameter('roi_y_min', -2.5)
        self.declare_parameter('roi_y_max', 2.5)

        # Clustering (nearest centroid obstacle for PF)
        self.declare_parameter('cluster_tolerance', 0.20)
        self.declare_parameter('min_points', 6)
        self.declare_parameter('max_points', 800)

        # Potential Field params (속도용 스케일)
        self.declare_parameter('Kp_att', 1.0)
        self.declare_parameter('Kp_rep', 3.0)
        self.declare_parameter('obstacle_bound', 1.2)
        self.declare_parameter('stop_dist', 0.28)

        # A* local grid params
        self.declare_parameter('grid_size', 6.0)      # meters (6x6)
        self.declare_parameter('grid_res', 0.05)      # meters/cell
        self.declare_parameter('inflate_radius', 0.20)  # meters (로봇 반경+여유)
        self.declare_parameter('astar_lookahead', 0.60) # meters (경로 추종 목표점)

        # Control params
        self.declare_parameter('v_max', 0.22)
        self.declare_parameter('w_max', 1.5)
        self.declare_parameter('heading_gain', 1.5)
        self.declare_parameter('goal_tolerance', 0.25)

        # State
        self.odom_ok = False
        self.goal_ok = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.gx = 0.0
        self.gy = 0.0

        # Latest scan points (robot frame)
        self.pts_robot = np.zeros((0, 2), dtype=np.float32)

        # PF nearest obstacle (robot frame)
        self.nearest_centroid_robot = None
        self.nearest_dist = 1e9

        # ROS I/O
        self.sub_scan = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.on_scan, 10)
        self.sub_odom = self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.on_odom, 10)
        self.sub_goal = self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.on_goal, 10)
        self.pub_cmd = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 10)

        self.timer = self.create_timer(0.05, self.control_step)  # 20Hz

        self.get_logger().info("PlannerPFAStar ready. Set goal with RViz2 '2D Nav Goal'.")

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
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.shape[0], dtype=np.float32) * msg.angle_increment

        rmin = float(self.get_parameter('range_min').value)
        rmax = float(self.get_parameter('range_max').value)
        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        ranges = ranges[valid]
        angles = angles[valid]

        if ranges.size < 10:
            self.pts_robot = np.zeros((0, 2), dtype=np.float32)
            self.nearest_centroid_robot = None
            self.nearest_dist = 1e9
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
        self.pts_robot = np.stack([xs, ys], axis=1).astype(np.float32) if xs.size else np.zeros((0, 2), dtype=np.float32)

        # PF용: nearest centroid (클러스터링)
        self.nearest_centroid_robot, self.nearest_dist = self.compute_nearest_centroid(self.pts_robot)

    # ---------------- main control ----------------
    def control_step(self):
        if not (self.odom_ok and self.goal_ok):
            return

        # goal reached?
        if float(np.hypot(self.gx - self.x, self.gy - self.y)) <= float(self.get_parameter('goal_tolerance').value):
            self.publish_cmd(0.0, 0.0)
            return

        mode = str(self.get_parameter('mode').value).lower().strip()
        if mode == 'astar':
            self.step_astar()
        else:
            self.step_pf()

    # ---------------- Potential Field step ----------------
    def step_pf(self):
        # Attractive (global)
        Kp_att = float(self.get_parameter('Kp_att').value)
        att_x, att_y = self.calc_attractive_force(self.x, self.y, self.gx, self.gy, Kp_att)

        # Repulsive from nearest centroid
        rep_x, rep_y = 0.0, 0.0
        stop_dist = float(self.get_parameter('stop_dist').value)

        if self.nearest_centroid_robot is not None:
            ox_r, oy_r = self.nearest_centroid_robot
            if self.nearest_dist <= stop_dist:
                w = -1.0 if oy_r > 0.0 else 1.0
                self.publish_cmd(0.0, self.clamp(w, -self.get_parameter('w_max').value, self.get_parameter('w_max').value))
                return

            # robot->global
            ox_g = self.x + np.cos(self.yaw) * ox_r - np.sin(self.yaw) * oy_r
            oy_g = self.y + np.sin(self.yaw) * ox_r + np.cos(self.yaw) * oy_r

            Kp_rep = float(self.get_parameter('Kp_rep').value)
            obstacle_bound = float(self.get_parameter('obstacle_bound').value)
            rep_x, rep_y = self.calc_repulsive_force(self.x, self.y, [(ox_g, oy_g)], Kp_rep, obstacle_bound)

        # Potential vector (global)
        pot_x = att_x + rep_x
        pot_y = att_y + rep_y

        # Convert pot to robot frame for cmd
        pot_rx = np.cos(self.yaw) * pot_x + np.sin(self.yaw) * pot_y
        pot_ry = -np.sin(self.yaw) * pot_x + np.cos(self.yaw) * pot_y

        heading = float(np.arctan2(pot_ry, pot_rx))
        v_max = float(self.get_parameter('v_max').value)
        w_max = float(self.get_parameter('w_max').value)
        heading_gain = float(self.get_parameter('heading_gain').value)

        v = float(np.hypot(pot_rx, pot_ry))
        v = min(v, v_max)
        if pot_rx < 0.05:
            v *= 0.2

        w = self.clamp(heading_gain * heading, -w_max, w_max)
        self.publish_cmd(v, w)

    # ---------------- A* step ----------------
    def step_astar(self):
        # goal (global) -> robot frame
        gx_r, gy_r = self.global_to_robot(self.gx, self.gy)

        # build local occupancy grid from scan points
        grid, origin, res = self.build_local_grid(self.pts_robot)
        if grid is None:
            # scan이 없으면 PF로라도
            self.step_pf()
            return

        # start/goal cell
        start = self.world_to_cell(0.0, 0.0, origin, res, grid.shape)  # robot at center
        goal = self.world_to_cell(gx_r, gy_r, origin, res, grid.shape)

        if start is None or goal is None:
            # goal이 너무 멀면 clamp해서 로컬 계획
            gx_r = float(np.clip(gx_r, -float(self.get_parameter('grid_size').value)/2, float(self.get_parameter('grid_size').value)/2))
            gy_r = float(np.clip(gy_r, -float(self.get_parameter('grid_size').value)/2, float(self.get_parameter('grid_size').value)/2))
            goal = self.world_to_cell(gx_r, gy_r, origin, res, grid.shape)
            if goal is None:
                self.step_pf()
                return

        # A* path
        path = self.astar(grid, start, goal)
        if not path:
            # 길이 없으면 회전하거나 PF fallback
            self.step_pf()
            return

        # path -> waypoint (lookahead)
        lookahead = float(self.get_parameter('astar_lookahead').value)
        wx, wy = self.pick_waypoint_from_path(path, origin, res, lookahead)

        # waypoint (robot frame) -> cmd_vel
        heading = float(np.arctan2(wy, wx))
        dist = float(np.hypot(wx, wy))

        v_max = float(self.get_parameter('v_max').value)
        w_max = float(self.get_parameter('w_max').value)
        heading_gain = float(self.get_parameter('heading_gain').value)

        v = min(v_max, 0.6 * dist)  # 가까우면 느리게
        if wx < 0.05:
            v *= 0.2

        w = self.clamp(heading_gain * heading, -w_max, w_max)
        self.publish_cmd(v, w)

    # ---------------- helpers ----------------
    def compute_nearest_centroid(self, pts: np.ndarray):
        if pts is None or pts.shape[0] < 10:
            return None, 1e9

        clusters = self.euclidean_clustering(
            pts,
            float(self.get_parameter('cluster_tolerance').value),
            int(self.get_parameter('min_points').value),
            int(self.get_parameter('max_points').value)
        )
        if not clusters:
            return None, 1e9

        nearest_c = None
        nearest_d = 1e9
        for idxs in clusters:
            c = pts[idxs].mean(axis=0)
            d = float(np.hypot(c[0], c[1]))
            if d < nearest_d:
                nearest_d = d
                nearest_c = (float(c[0]), float(c[1]))
        return nearest_c, nearest_d

    def global_to_robot(self, gx, gy):
        # global point -> robot frame (x forward, y left)
        dx = gx - self.x
        dy = gy - self.y
        rx = np.cos(self.yaw) * dx + np.sin(self.yaw) * dy
        ry = -np.sin(self.yaw) * dx + np.cos(self.yaw) * dy
        return float(rx), float(ry)

    def build_local_grid(self, pts_robot: np.ndarray):
        grid_size = float(self.get_parameter('grid_size').value)
        res = float(self.get_parameter('grid_res').value)
        inflate_r = float(self.get_parameter('inflate_radius').value)

        n = int(round(grid_size / res))
        n = max(n, 40)
        grid = np.zeros((n, n), dtype=np.uint8)  # 0 free, 1 occ

        # origin in robot frame: bottom-left corner
        half = grid_size / 2.0
        origin = (-half, -half)

        if pts_robot is None or pts_robot.shape[0] == 0:
            return grid, origin, res

        # mark occupied cells from points (and inflate)
        infl = int(np.ceil(inflate_r / res))
        for p in pts_robot:
            cx = self.world_to_cell(p[0], p[1], origin, res, grid.shape)
            if cx is None:
                continue
            i, j = cx
            for di in range(-infl, infl + 1):
                for dj in range(-infl, infl + 1):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        grid[ii, jj] = 1

        # start 주변은 free로 살짝 보장(센서 노이즈 완화)
        s = self.world_to_cell(0.0, 0.0, origin, res, grid.shape)
        if s is not None:
            si, sj = s
            grid[max(0, si-1):min(n, si+2), max(0, sj-1):min(n, sj+2)] = 0

        return grid, origin, res

    def world_to_cell(self, x, y, origin, res, shape):
        ox, oy = origin
        n_i, n_j = shape
        j = int(np.floor((x - ox) / res))  # x -> col
        i = int(np.floor((y - oy) / res))  # y -> row
        if 0 <= i < n_i and 0 <= j < n_j:
            return (i, j)
        return None

    def cell_to_world(self, i, j, origin, res):
        ox, oy = origin
        x = ox + (j + 0.5) * res
        y = oy + (i + 0.5) * res
        return float(x), float(y)

    def pick_waypoint_from_path(self, path, origin, res, lookahead):
        # path: list of (i,j) from start->goal
        # accumulate distance in robot-frame world coordinates
        if len(path) == 1:
            return self.cell_to_world(path[0][0], path[0][1], origin, res)

        last_x, last_y = self.cell_to_world(path[0][0], path[0][1], origin, res)
        acc = 0.0
        for k in range(1, len(path)):
            x, y = self.cell_to_world(path[k][0], path[k][1], origin, res)
            acc += float(np.hypot(x - last_x, y - last_y))
            if acc >= lookahead:
                return x, y
            last_x, last_y = x, y
        return last_x, last_y

    # ---------------- A* ----------------
    def astar(self, grid, start, goal):
        # 8-neighbor A*
        if grid[start[0], start[1]] == 1:
            return []
        if grid[goal[0], goal[1]] == 1:
            # goal이 막혀있으면 주변 free 찾기(간단)
            goal = self.find_nearest_free(grid, goal, max_r=5)
            if goal is None:
                return []

        def h(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                     (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

        openpq = []
        heapq.heappush(openpq, (0.0, start))
        came = {start: None}
        g = {start: 0.0}

        while openpq:
            _, cur = heapq.heappop(openpq)
            if cur == goal:
                return self.reconstruct_path(came, cur)

            for di, dj, c in neighbors:
                ni, nj = cur[0] + di, cur[1] + dj
                if ni < 0 or nj < 0 or ni >= grid.shape[0] or nj >= grid.shape[1]:
                    continue
                if grid[ni, nj] == 1:
                    continue

                nxt = (ni, nj)
                ng = g[cur] + c
                if nxt not in g or ng < g[nxt]:
                    g[nxt] = ng
                    f = ng + h(nxt, goal)
                    heapq.heappush(openpq, (f, nxt))
                    came[nxt] = cur

        return []

    def find_nearest_free(self, grid, cell, max_r=5):
        ci, cj = cell
        for r in range(1, max_r + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        if grid[ni, nj] == 0:
                            return (ni, nj)
        return None

    def reconstruct_path(self, came, cur):
        path = []
        while cur is not None:
            path.append(cur)
            cur = came[cur]
        path.reverse()
        return path

    # ---------------- PF math (user code 기반) ----------------
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
        return rep_x, rep_y

    # ---------------- clustering (BFS) ----------------
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

    def publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_cmd.publish(msg)

    def clamp(self, x, lo, hi):
        return max(lo, min(hi, x))


def main():
    rclpy.init()
    node = PlannerPFAStar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
