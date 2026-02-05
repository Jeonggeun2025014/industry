import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import cv2

from ultralytics import YOLO


class YoloStopNode(Node):
    def __init__(self):
        super().__init__('yolo_stop_node')

        # ---- parameters (수업에서 쉽게 바꾸도록) ----
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('model', 'yolov8n.pt')  # nano: 가장 가벼움
        self.declare_parameter('conf_th', 0.5)         # confidence threshold
        self.declare_parameter('min_box_area', 6000)   # 너무 작은 검출은 무시(픽셀^2)
        self.declare_parameter('stop_hold_sec', 0.3)   # 잠깐이라도 보이면 이 시간만큼 정지 유지

        image_topic = self.get_parameter('image_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        model_name = self.get_parameter('model').value

        self.conf_th = float(self.get_parameter('conf_th').value)
        self.min_box_area = int(self.get_parameter('min_box_area').value)
        self.stop_hold_sec = float(self.get_parameter('stop_hold_sec').value)

        self.bridge = CvBridge()
        self.model = YOLO(model_name)

        self.sub = self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, '/yolo/image', 10)

        self.last_person_time = None

        self.get_logger().info(f"Subscribed: {image_topic}")
        self.get_logger().info(f"Publishing cmd_vel: {cmd_vel_topic}")
        self.get_logger().info(f"Debug image: /yolo/image")
        self.get_logger().info(f"YOLO model: {model_name}")

    def publish_stop(self):
        twist = Twist()  # all zeros -> stop
        self.cmd_pub.publish(twist)

    def image_cb(self, msg: Image):
        # 1) ROS Image -> OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        # 2) YOLO inference
        results = self.model.predict(frame, verbose=False)[0]

        person_detected = False

        # 3) parse detections
        if results.boxes is not None and results.names is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = results.names.get(cls_id, str(cls_id))

                if conf < self.conf_th:
                    continue

                # bbox
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                area = max(0, x2 - x1) * max(0, y2 - y1)

                # 작은 박스는 노이즈로 보고 무시
                if area < self.min_box_area:
                    continue

                # 디버그 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if name == 'person':
                    person_detected = True

        # 4) stop logic (hold)
        now = self.get_clock().now()

        if person_detected:
            self.last_person_time = now
            self.publish_stop()
            self.get_logger().info("Person detected -> STOP", throttle_duration_sec=0.5)

        # hold stop for a short time even if momentarily lost
        if self.last_person_time is not None:
            dt = (now - self.last_person_time).nanoseconds / 1e9
            if dt < self.stop_hold_sec:
                self.publish_stop()

        # 5) publish debug image
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = msg.header
            self.debug_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f"debug publish error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
