#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class ScanClusterNode(Node):
    """
    /scan(LaserScan) -> (x,y) points -> Euclidean clustering(거리 기반, PCL 방식) -> MarkerArray publish
    외부 패키지(sklearn/scipy) 없이 동작
    """

    def __init__(self):
        super().__init__('scan_cluster_node')

        # 토픽/프레임
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('frame_id', '')  # 비우면 msg.header.frame_id 사용

        # 거리 필터
        self.declare_parameter('range_min', 0.12)
        self.declare_parameter('range_max', 6.0)

        # ROI (전방 x, 좌우 y)
        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 3.5)
        self.declare_parameter('roi_y_min', -2.0)
        self.declare_parameter('roi_y_max', 2.0)

        # 클러스터링 파라미터(PCL 대응)
        self.declare_parameter('cluster_tolerance', 0.20)  # [m]
        self.declare_parameter('min_points', 6)
        self.declare_parameter('max_points', 800)

        # RViz 마커 크기
        self.declare_parameter('point_scale', 0.05)
        self.declare_parameter('centroid_scale', 0.12)

        scan_topic = self.get_parameter('scan_topic').value
        self.sub = self.create_subscription(LaserScan, scan_topic, self.on_scan, 10)
        self.pub = self.create_publisher(MarkerArray, '/clusters_markers', 10)

        self.get_logger().info(f"Subscribe {scan_topic} -> Publish /clusters_markers (MarkerArray)")

    def on_scan(self, msg: LaserScan):
        # 1) LaserScan -> ranges/angles
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.shape[0], dtype=np.float32) * msg.angle_increment

        rmin = float(self.get_parameter('range_min').value)
        rmax = float(self.get_parameter('range_max').value)

        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        ranges = ranges[valid]
        angles = angles[valid]

        if ranges.size < 10:
            self._publish_deleteall()
            return

        # 2) polar -> cartesian
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        # 3) ROI filter
        x_min = float(self.get_parameter('roi_x_min').value)
        x_max = float(self.get_parameter('roi_x_max').value)
        y_min = float(self.get_parameter('roi_y_min').value)
        y_max = float(self.get_parameter('roi_y_max').value)

        roi = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        xs = xs[roi]
        ys = ys[roi]

        if xs.size < 10:
            self._publish_deleteall()
            return

        pts = np.stack([xs, ys], axis=1)  # (N,2)

        # 4) Euclidean clustering (PCL 방식: 거리 기반 region growing)
        tol = float(self.get_parameter('cluster_tolerance').value)
        min_pts = int(self.get_parameter('min_points').value)
        max_pts = int(self.get_parameter('max_points').value)

        clusters = self.euclidean_clustering(pts, tol, min_pts, max_pts)

        # 5) MarkerArray publish
        frame_override = self.get_parameter('frame_id').value
        frame_id = frame_override if frame_override else msg.header.frame_id

        point_scale = float(self.get_parameter('point_scale').value)
        centroid_scale = float(self.get_parameter('centroid_scale').value)

        marray = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marray.markers.append(delete_all)

        for k, cluster_idx in enumerate(clusters):
            cluster_pts = pts[cluster_idx]

            color = self._pseudo_color(k)

            # (A) 군집 점 표시
            mk = Marker()
            mk.header = msg.header
            mk.header.frame_id = frame_id
            mk.ns = "cluster_points"
            mk.id = k
            mk.type = Marker.POINTS
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = point_scale
            mk.scale.y = point_scale
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = color
            mk.points = [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in cluster_pts]
            marray.markers.append(mk)

            # (B) 중심점 표시
            c = cluster_pts.mean(axis=0)
            mc = Marker()
            mc.header = msg.header
            mc.header.frame_id = frame_id
            mc.ns = "cluster_centroids"
            mc.id = 10000 + k
            mc.type = Marker.SPHERE
            mc.action = Marker.ADD
            mc.pose.position.x = float(c[0])
            mc.pose.position.y = float(c[1])
            mc.pose.position.z = 0.0
            mc.pose.orientation.w = 1.0
            mc.scale.x = centroid_scale
            mc.scale.y = centroid_scale
            mc.scale.z = centroid_scale
            mc.color.r, mc.color.g, mc.color.b, mc.color.a = color
            marray.markers.append(mc)

        self.pub.publish(marray)

        self.get_logger().info(
            f"N={pts.shape[0]} clusters={len(clusters)} tol={tol} min={min_pts}",
            throttle_duration_sec=1.0
        )

    def euclidean_clustering(self, pts: np.ndarray, tol: float, min_pts: int, max_pts: int):
        """
        거리 tol 이내인 점들을 연결(그래프)했다고 보고 BFS로 군집을 찾는 방식.
        N이 360~720 수준이면 O(N^2)도 충분히 빠름(수업용 안정).
        """
        n = pts.shape[0]
        visited = np.zeros(n, dtype=bool)
        clusters = []

        tol2 = tol * tol  # 거리 제곱으로 비교(빠름)

        for i in range(n):
            if visited[i]:
                continue

            # seed 시작
            queue = [i]
            visited[i] = True
            cluster = [i]

            while queue:
                cur = queue.pop()

                # cur과 모든 점의 거리^2 계산 (벡터화)
                diff = pts - pts[cur]
                dist2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]

                # tol 이내이고 아직 방문 안 한 점들
                nbrs = np.where((dist2 <= tol2) & (~visited))[0]
                if nbrs.size > 0:
                    visited[nbrs] = True
                    queue.extend(nbrs.tolist())
                    cluster.extend(nbrs.tolist())

                # 너무 커지면(벽 전체 등) 중단 처리 가능
                if len(cluster) > max_pts:
                    break

            # 크기 필터
            if min_pts <= len(cluster) <= max_pts:
                clusters.append(np.array(cluster, dtype=np.int32))

        return clusters

    def _publish_deleteall(self):
        marray = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        marray.markers.append(m)
        self.pub.publish(marray)

    def _pseudo_color(self, k: int):
        palette = [
            (1.0, 0.2, 0.2, 1.0),
            (0.2, 1.0, 0.2, 1.0),
            (0.2, 0.2, 1.0, 1.0),
            (1.0, 1.0, 0.2, 1.0),
            (1.0, 0.2, 1.0, 1.0),
            (0.2, 1.0, 1.0, 1.0),
        ]
        return palette[k % len(palette)]


def main():
    rclpy.init()
    node = ScanClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
