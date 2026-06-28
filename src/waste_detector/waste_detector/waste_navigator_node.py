#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math

STOP_DISTANCE  = 0.40
OBSTACLE_DIST  = 0.45
KP_LINEAR      = 0.4
KP_ANGULAR     = 0.8
MAX_LINEAR     = 0.18
MAX_ANGULAR    = 0.60
EXPLORE_LINEAR = 0.12
TIMEOUT_SEC    = 1.5

class WasteNavigatorNode(Node):

    def __init__(self):
        super().__init__('waste_navigator')

        self.pub_cmd     = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.pub_trigger = self.create_publisher(Bool, '/waste_collector/trigger', 10)

        self.sub_pos = self.create_subscription(
            PointStamped, '/waste_detector/position', self.position_callback, 10)
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.last_detection_time = None
        self.collecting  = False
        self.target_x    = 0.0
        self.target_z    = 999.0
        self.front_dist  = 999.0
        self.left_dist   = 999.0
        self.right_dist  = 999.0

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info('WasteNavigator iniciado!')

    def scan_callback(self, msg):
        r = msg.ranges
        n = len(r)

        def safe(v):
            return v if (not math.isnan(v) and not math.isinf(v) and v > 0.01) else 999.0

        front = list(range(0, 20)) + list(range(n - 20, n))
        self.front_dist = min(safe(r[i]) for i in front)
        self.left_dist  = min(safe(r[i]) for i in range(60, 120))
        self.right_dist = min(safe(r[i]) for i in range(240, 300))

    def position_callback(self, msg):
        self.last_detection_time = self.get_clock().now()
        self.target_x = msg.point.x
        self.target_z = msg.point.z

    def _pub(self, linear, angular):
        msg = TwistStamped()
        msg.twist.linear.x  = float(linear)
        msg.twist.angular.z = float(angular)
        self.pub_cmd.publish(msg)

    def control_loop(self):
        has_target = False
        if self.last_detection_time is not None:
            elapsed = (self.get_clock().now() - self.last_detection_time).nanoseconds * 1e-9
            has_target = elapsed < TIMEOUT_SEC

        if has_target:
            if self.target_z <= STOP_DISTANCE:
                self._pub(0, 0)
                if not self.collecting:
                    self.collecting = True
                    t = Bool(); t.data = True
                    self.pub_trigger.publish(t)
                    self.get_logger().info(f'Resíduo alcançado @ Z={self.target_z:.2f}m!')
                return

            self.collecting = False
            angle_error = math.atan2(self.target_x, self.target_z)
            lin = min(MAX_LINEAR,  KP_LINEAR  * (self.target_z - STOP_DISTANCE))
            ang = max(-MAX_ANGULAR, min(MAX_ANGULAR, -KP_ANGULAR * angle_error))

            if self.front_dist < OBSTACLE_DIST:
                lin = 0.0
                ang = MAX_ANGULAR

            self._pub(lin, ang)
            self.get_logger().info(
                f'APPROACH Z={self.target_z:.2f}m ang={ang:.2f}',
                throttle_duration_sec=0.5)

        else:
            self.collecting = False
            if self.front_dist < OBSTACLE_DIST:
                direction = MAX_ANGULAR if self.left_dist > self.right_dist else -MAX_ANGULAR
                self._pub(0, direction)
                self.get_logger().info(
                    f'EXPLORE obstáculo! front={self.front_dist:.2f}m',
                    throttle_duration_sec=1.0)
            else:
                ang = 0.0
                if self.left_dist < 0.6:
                    ang = -0.3
                elif self.right_dist < 0.6:
                    ang = 0.3
                self._pub(EXPLORE_LINEAR, ang)
                self.get_logger().info(
                    f'EXPLORE front={self.front_dist:.2f}m esq={self.left_dist:.2f}m dir={self.right_dist:.2f}m',
                    throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = WasteNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
