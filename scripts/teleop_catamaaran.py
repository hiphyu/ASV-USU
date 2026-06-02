#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import sys, termios, tty

# Key mapping
# Key mapping yang diperbaiki
# Format: (action, left_thrust, right_thrust, left_steer, right_steer)
MOVE_BINDINGS = {
    'w': ('forward', 100.0, 100.0, 0.0, 0.0),    # Maju lurus
    'x': ('fast forward', 150.0, 150.0, 0.0, 0.0),
    's': ('backward', -10.0, -10.0, 0.0, 0.0), # Mundur lurus
    'a': ('left turn', -100.0, 250.0, -1.0, 1.0),  # Belok kanan
    'b': ('half left turn', -50.0, 100.0, -0.8, 0.8),  # Belok kanan tengah
    'd': ('right turn', 250.0, -100.0, 1.0, -1.0), # Belok kiri
    'e': ('half right turn', 100.0, -50.0, 0.8, -0.8), # Belok kiri
    'q': ('stop', 0.0, 0.0, 0.0, 0.0),
}

def getKey():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class TeleopWAMV(Node):
    def __init__(self):
        super().__init__('teleop_wamv')

        # Publisher untuk thrust
        self.left_thrust_pub = self.create_publisher(Float64, '/model/wamv/left/joint/cmd_thrust', 10)
        self.right_thrust_pub = self.create_publisher(Float64, '/model/wamv/right/joint/cmd_thrust', 10)

        # Publisher untuk steer/posisi thruster
        self.left_steer_pub = self.create_publisher(Float64, '/wamv/left/thruster/joint/cmd_pos', 10)
        self.right_steer_pub = self.create_publisher(Float64, '/wamv/right/thruster/joint/cmd_pos', 10)

        self.get_logger().info("Teleop WAMV started! Use keys: w=forward, s=backward, a=left, b=half left, d=right, e=half right, q=stop")

    # Di dalam class TeleopWAMV
    def publish_cmd(self, left_thrust, right_thrust, left_steer, right_steer):
        # thrust
        l_t_msg = Float64()
        l_t_msg.data = left_thrust
        self.left_thrust_pub.publish(l_t_msg)

        r_t_msg = Float64()
        r_t_msg.data = right_thrust
        self.right_thrust_pub.publish(r_t_msg)

        # steer
        l_s_msg = Float64()
        l_s_msg.data = left_steer
        self.left_steer_pub.publish(l_s_msg)

        r_s_msg = Float64()
        r_s_msg.data = right_steer
        self.right_steer_pub.publish(r_s_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopWAMV()

    try:
        while True:
            key = getKey()
            if key in MOVE_BINDINGS.keys():
                action, lt, rt, ls, rs = MOVE_BINDINGS[key]
                node.get_logger().info(f"Command: {action}")
                node.publish_cmd(lt, rt, ls, rs)
            elif key == '\x03':  # Ctrl-C
                break
    except Exception as e:
        print(e)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
