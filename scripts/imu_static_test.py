#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math, numpy as np

class StaticTest(Node):
    def __init__(self):
        super().__init__('static_test')
        self.yaw_data = []
        self.create_subscription(Imu, '/boat/imu/data', self.cb, 10)
        # Stop otomatis setelah 60 detik
        self.create_timer(60.0, self.report)

    def cb(self, msg):
        z, w = msg.orientation.z, msg.orientation.w
        x, y = msg.orientation.x, msg.orientation.y
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.yaw_data.append(yaw)

    def report(self):
        arr = np.array(self.yaw_data)
        print(f"Jumlah sample : {len(arr)}")
        print(f"Yaw mean      : {arr.mean():.6f} rad")
        print(f"Yaw std       : {arr.std():.6f} rad")
        print(f"R_yaw (std²)  : {arr.std()**2:.8f}  ← masukkan ke EKF node")
        rclpy.shutdown()

rclpy.init()
rclpy.spin(StaticTest())