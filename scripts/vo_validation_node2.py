#!/usr/bin/env python3
"""
Cara pakai:
  ros2 run <pkg> <nama file> --ros-args -p lintasan:=A -p run_ke:=1
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
import math

class PureEKFNode(Node):

    def __init__(self):
        super().__init__('pure_ekf_node')

        self.declare_parameter('lintasan', 'A')
        self.declare_parameter('run_ke', 1)
        self.lintasan = self.get_parameter('lintasan').value.upper()
        self.run_ke   = self.get_parameter('run_ke').value

        # State vector [x, y, vx, vy, yaw, yaw_rate]
        self.state = np.zeros(6, dtype=float)
        self.P = np.identity(6)

        # Parameter Q
        self.Q = np.diag([5e-4, 5e-4, 1e-04, 1e-04, 1e-05, 1e-05]) #Lintasan A
        # self.Q = np.diag([1e-3, 1e-3, 1e-4, 1e-4, 5e-6, 5e-6]) #Lintasan B

        # Parameter R
        self.R_gps = np.diag([0.18, 0.18])
        self.R_imu_yaw  = 0.005
        self.R_imu_rate = 0.000019

        self.initialized      = False
        self.last_predict_time = None

        # Hasil EKF di publish ke topic /pure_ekf_vxvy/odom
        self.odom_pub = self.create_publisher(
            Odometry, '/pure_ekf_vxvy/odom', 10
        )

        # Subscribers
        self.create_subscription(Odometry, '/gps_only/odom', self._gps_cb, 10)
        self.create_subscription(Imu,      '/boat/imu/data', self._imu_cb, 50)
        self.create_subscription(Odometry, '/odom',          self._gt_cb,  50)

        # Timer 50Hz
        self.create_timer(0.02, self._timer_cb)

        # Ground truth cache
        self._gt_x = None; self._gt_y = None

        self.get_logger().info(
            f'\n{"="*50}\n'
            f'  PURE EKF — Lintasan {self.lintasan} Run {self.run_ke}\n'
            f'  Topic output: /pure_ekf_vxvy/odom\n'
            f'{"="*50}'
        )

    def _norm(self, a):
        while a >  math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a

    def _quat_to_yaw(self, q):
        siny = 2.0*(q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0*(q.y**2 + q.z**2)
        return math.atan2(siny, cosy)

    def _yaw_to_quat(self, yaw):
        h = yaw*0.5
        return (0.0, 0.0, math.sin(h), math.cos(h))

    # PREDICT
    def _predict(self, dt):
        if dt <= 0 or dt > 1.0:
            return

        x, y, vx, vy, yaw, yr = self.state
        cy, sy = math.cos(yaw), math.sin(yaw)

        x_new   = x + (vx*cy - vy*sy)*dt
        y_new   = y + (vx*sy + vy*cy)*dt
        yaw_new = self._norm(yaw + yr*dt)

        vx_new = vx * 0.98
        vy_new = vy * 0.98

        self.state = np.array([x_new, y_new, vx_new, vy_new, yaw_new, yr])

        F = np.eye(6)
        F[0,2]=cy*dt;  F[0,3]=-sy*dt; F[0,4]=(-vx*sy-vy*cy)*dt
        F[1,2]=sy*dt;  F[1,3]= cy*dt; F[1,4]=( vx*cy-vy*sy)*dt
        F[4,5]=dt

        self.P = F @ self.P @ F.T + self.Q

    # UPDATE
    def _update(self, z, H, R):
        y = z - H @ self.state
        if H.shape[0] == 1 and abs(H[0,4]) > 0.5:
            y[0] = self._norm(y[0])

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.state += K @ y
        self.state[4] = self._norm(self.state[4])

        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P

    # PUBLISH
    def _publish(self):
        x, y, vx, vy, yaw, yr = self.state

        odom = Odometry()
        odom.header.stamp    = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        qx, qy, qz, qw = self._yaw_to_quat(yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x  = float(vx)
        odom.twist.twist.linear.y  = float(vy)
        odom.twist.twist.angular.z = float(yr)
        odom.pose.covariance[0]  = float(self.P[0,0])
        odom.pose.covariance[7]  = float(self.P[1,1])
        odom.pose.covariance[35] = float(self.P[4,4])
        self.odom_pub.publish(odom)

    # TIMER 50Hz
    def _timer_cb(self):
        if not self.initialized:
            return
        now = self.get_clock().now()
        if self.last_predict_time is None:
            self.last_predict_time = now
            return
        dt = (now - self.last_predict_time).nanoseconds/1e9
        self.last_predict_time = now
        self._predict(dt)
        self._publish()

    # GPS CALLBACK
    def _gps_cb(self, msg):
        gx = msg.pose.pose.position.x
        gy = msg.pose.pose.position.y

        if not self.initialized:
            self.state[0] = gx
            self.state[1] = gy
            self.state[4] = self._quat_to_yaw(msg.pose.pose.orientation)
            self.initialized      = True
            self.last_predict_time = self.get_clock().now()
            self.get_logger().info(
                f'GPS_INIT x={gx:.3f} y={gy:.3f}'
            )
            self._publish()
            return

        z = np.array([gx, gy])
        H = np.zeros((2,6)); H[0,0]=1.0; H[1,1]=1.0
        self._update(z, H, self.R_gps)
        self._publish()

    # IMU CALLBACK
    def _imu_cb(self, msg):
        if not self.initialized:
            return
        yaw = self._quat_to_yaw(msg.orientation)
        yr  = msg.angular_velocity.z

        H_yaw  = np.zeros((1,6)); H_yaw[0,4]  = 1.0
        H_rate = np.zeros((1,6)); H_rate[0,5] = 1.0
        self._update(np.array([yaw]),  H_yaw,  np.array([[self.R_imu_yaw]]))
        self._update(np.array([yr]),   H_rate, np.array([[self.R_imu_rate]]))

    def _gt_cb(self, msg):
        self._gt_x = msg.pose.pose.position.x
        self._gt_y = msg.pose.pose.position.y


def main(args=None):
    rclpy.init(args=args)
    node = PureEKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()