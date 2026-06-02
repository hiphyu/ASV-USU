#!/usr/bin/env python3
import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry

def quat_to_yaw(q):
    qx,qy,qz,qw = q.x,q.y,q.z,q.w
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_quat(yaw):
    half = 0.5*yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

class GPSOnly(Node):
    def __init__(self):
        super().__init__('gps_only_odom')
        self.declare_parameter('earth_radius', 6371000.0)

        self.Re = float(self.get_parameter('earth_radius').value)

        self.gps0 = None
        self.gt0 = None
        self.imu_yaw0 = None
        self.imu_yaw = 0.0

        self.sub_gt = self.create_subscription(Odometry, '/odom', self.gt_cb, 50)
        self.sub_gps = self.create_subscription(NavSatFix, '/boat/gps/fix', self.gps_cb, 50)
        self.sub_imu = self.create_subscription(Imu, '/boat/imu/data', self.imu_cb, 50)
        self.pub = self.create_publisher(Odometry, '/gps_only/odom', 10)

    def gt_cb(self, msg: Odometry):
        if self.gt0 is None:
            self.gt0 = (msg.pose.pose.position.x, msg.pose.pose.position.y, quat_to_yaw(msg.pose.pose.orientation))

    def imu_cb(self, msg: Imu):
        self.imu_yaw = quat_to_yaw(msg.orientation)
        if self.imu_yaw0 is None:
            self.imu_yaw0 = self.imu_yaw

    def gps_cb(self, msg: NavSatFix):
        if self.gt0 is None or self.imu_yaw0 is None:
            return
        if self.gps0 is None:
            self.gps0 = (msg.latitude, msg.longitude)

        lat0, lon0 = self.gps0
        dlon = math.radians(msg.longitude - lon0)
        dlat = math.radians(msg.latitude - lat0)

        x_rel = self.Re * dlon * math.cos(math.radians(lat0))
        y_rel = self.Re * dlat

        gx0, gy0, gyaw0 = self.gt0
        yaw = wrap(self.imu_yaw - self.imu_yaw0 + gyaw0)

        out = Odometry()
        out.header = msg.header
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_link'
        # out.child_frame_id = 'gps_only'
        out.pose.pose.position.x = float(gx0 + x_rel)
        out.pose.pose.position.y = float(gy0 + y_rel)
        out.pose.pose.position.z = 0.0
        qx,qy,qz,qw = yaw_to_quat(yaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # [x, y, z, roll, pitch, yaw] -> urutan diagonal: 0, 7, 14, 21, 28, 35
        cov = [0.0] * 36

        cov[0]  = 0.25   # Variansi Posisi X (dalam meter kuadrat)
        cov[7]  = 0.25   # Variansi Posisi Y

        out.pose.covariance = cov
        self.pub.publish(out)

def main():
    rclpy.init()
    node = GPSOnly()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
