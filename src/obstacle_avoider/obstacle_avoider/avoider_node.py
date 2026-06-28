import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import math

class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        self.DISTANCIA_SEGURA = 0.5
        self.VELOCIDADE_LINEAR = 0.15
        self.VELOCIDADE_ANGULAR = 0.8
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.get_logger().info('Nó de desvio de obstáculos iniciado!')

    def scan_callback(self, msg):
        ranges = msg.ranges
        total = len(ranges)
        def valida(r):
            return not math.isinf(r) and not math.isnan(r) and r > 0.0
        cone = 30
        frente_indices = list(range(0, cone)) + list(range(total - cone, total))
        esquerda_indices = list(range(60, 120))
        direita_indices = list(range(240, 300))
        frente   = [ranges[i] for i in frente_indices   if valida(ranges[i])]
        esquerda = [ranges[i] for i in esquerda_indices if valida(ranges[i])]
        direita  = [ranges[i] for i in direita_indices  if valida(ranges[i])]
        dist_frente   = min(frente)   if frente   else float('inf')
        dist_esquerda = min(esquerda) if esquerda else float('inf')
        dist_direita  = min(direita)  if direita  else float('inf')
        twist = TwistStamped()
        if dist_frente < self.DISTANCIA_SEGURA:
            if dist_esquerda > dist_direita:
                twist.twist.linear.x = 0.0
                twist.twist.angular.z = self.VELOCIDADE_ANGULAR
                self.get_logger().info(f'Obstáculo ({dist_frente:.2f}m) → Girando ESQUERDA')
            else:
                twist.twist.linear.x = 0.0
                twist.twist.angular.z = -self.VELOCIDADE_ANGULAR
                self.get_logger().info(f'Obstáculo ({dist_frente:.2f}m) → Girando DIREITA')
        else:
            twist.twist.linear.x = self.VELOCIDADE_LINEAR
            twist.twist.angular.z = 0.0
            self.get_logger().info(f'Livre ({dist_frente:.2f}m) → Avançando')
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
