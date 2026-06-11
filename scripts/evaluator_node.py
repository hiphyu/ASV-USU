#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import message_filters
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — save dulu, show pakai viewer lain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import math
import subprocess
import signal
from datetime import datetime


# ============================================================
# POSISI BUOY
# ============================================================
BUOYS_RED = [
    (18.96, 10.24),   # mb_round_red_x
    (29.46, 13.74),   # mb_round_red_1
    (39.96,  8.74),   # mb_round_red_3
    (64.46, 24.24),   # mb_round_red_5
    (64.46, 31.24),   # mb_round_red_7
    (64.46, 38.24),   # mb_round_red_9
    (50.46, 65.49),   # mb_round_red_11
    (34.71, 68.99),   # mb_round_red_12
    (24.21, 68.99),   # mb_round_red_13
]

BUOYS_GREEN = [
    (18.96,  2.24),   # mb_round_green_x
    (29.46,  5.74),   # mb_round_green_2
    (39.96,  0.74),   # mb_round_green_4
    (72.46, 24.24),   # mb_round_green_6
    (72.46, 31.24),   # mb_round_green_8
    (72.46, 38.24),   # mb_round_green_10
    (50.46, 57.49),   # mb_round_green_12
    (34.71, 60.99),   # mb_round_green_13
    (24.21, 60.99),   # mb_round_green_14
]

# lintasanB
# BUOYS_RED = [
#     (17.21, -32.62),  # mb_round_red_0
#     (27.71, -34.12),  # mb_round_red_2
#     (38.21, -29.12),  # mb_round_red_4
#     (59.21, -49.12),  # mb_round_red_6
#     (59.21, -56.12),  # mb_round_red_8
#     (59.21, -63.12),  # mb_round_red_10
#     (45.21, -92.12),  # mb_round_red_14
#     (31.21, -95.62),  # mb_round_red_15
#     (20.71, -95.62),  # mb_round_red_16
# ]

# BUOYS_GREEN = [
#     (17.21, -24.62),  # mb_round_green_1
#     (27.71, -28.12),  # mb_round_green_3
#     (38.21, -21.12),  # mb_round_green_5
#     (67.21, -49.12),  # mb_round_green_7
#     (67.21, -56.12),  # mb_round_green_9
#     (67.21, -63.12),  # mb_round_green_11
#     (45.21, -84.12),  # mb_round_green_15
#     (31.21, -87.62),  # mb_round_green_16
#     (20.71, -87.62),  # mb_round_green_17
# ]

DOCKS = [
    (-3.80,  6.12),
    (-3.81,  7.12),
    # (-3.79, -24.62), #LintasanB
    # (-3.79, -25.62), #LintasanB
]


class EKFEvaluatorWithCSV(Node):

    def __init__(self):
        super().__init__('ekf_evaluator_with_csv')

        self.declare_parameter('lintasan', 'A')
        self.declare_parameter('run_ke', 1)
        self.declare_parameter('rekam_rosbag', True)

        self.lintasan     = self.get_parameter('lintasan').value.upper()
        self.run_ke       = self.get_parameter('run_ke').value
        self.rekam_rosbag = self.get_parameter('rekam_rosbag').value

        self.data = {
            'gt':  {'x': [], 'y': []},
            'ekf': {'x': [], 'y': []},
            'gps': {'x': [], 'y': []}
        }
        self.errors_ekf = []
        self.errors_gps = []

        self._latest_imu_yaw      = None
        self._latest_imu_yaw_rate = 0.0

        self.sample_count = 0
        self.start_time   = None

        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        label = f'Lintasan{self.lintasan}_Run{self.run_ke}'
        self.csv_filename       = f'ekf_trajectory_{label}_{ts}.csv'
        self.plot_traj_filename = f'ekf_trajectory_{label}_{ts}.png'
        self.plot_err_filename  = f'ekf_error_{label}_{ts}.png'

        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.writer   = csv.writer(self.csv_file)
        self.writer.writerow([
            'sample', 'timestamp_s', 'lintasan', 'run_ke',
            'gps_x', 'gps_y',
            'ekf_x', 'ekf_y', 'ekf_vx', 'ekf_vy',
            'ekf_yaw_deg', 'ekf_yaw_rate',
            'imu_yaw_deg', 'imu_yaw_rate_rads',
            'gt_x', 'gt_y', 'gt_yaw_deg',
            'error_ekf_m', 'error_gps_m',
            'error_ekf_x', 'error_ekf_y',
            'error_gps_x', 'error_gps_y',
        ])

        self.imu_sub = self.create_subscription(
            Imu, '/boat/imu/data', self._imu_cb, 50
        )

        self.sub_gt  = message_filters.Subscriber(self, Odometry, '/odom')
        self.sub_ekf = message_filters.Subscriber(self, Odometry, '/pure_ekf_vxvy/odom')
        self.sub_gps = message_filters.Subscriber(self, Odometry, '/gps_only/odom')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_gt, self.sub_ekf, self.sub_gps],
            queue_size=20, slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self._rosbag_proc = None
        if self.rekam_rosbag:
            self._start_rosbag(label, ts)

        self.get_logger().info(
            f'\n{"="*55}\n'
            f'  EKF Evaluator\n'
            f'  Lintasan : {self.lintasan}   Run ke : {self.run_ke}\n'
            f'  CSV      : {self.csv_filename}\n'
            f'  Rosbag   : {"AKTIF" if self.rekam_rosbag else "NONAKTIF"}\n'
            f'  Ctrl+C untuk stop dan generate hasil\n'
            f'{"="*55}'
        )

    def _start_rosbag(self, label, ts):
        bag_dir = f'rosbag_{label}_{ts}'
        topics  = ['/odom', '/gps_only/odom', '/pure_ekf_vxvy/odom',
                   '/boat/imu/data', '/boat/gps/fix','/tf']
        cmd = ['ros2', 'bag', 'record', '-o', bag_dir] + topics
        try:
            self._rosbag_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.get_logger().info(f'  Rosbag merekam ke: {bag_dir}/')
        except Exception as e:
            self.get_logger().warn(f'Rosbag gagal dimulai: {e}')

    def _stop_rosbag(self):
        if self._rosbag_proc is not None:
            try:
                self._rosbag_proc.send_signal(signal.SIGINT)
                self._rosbag_proc.wait(timeout=5)
            except Exception:
                pass

    def _imu_cb(self, msg: Imu):
        self._latest_imu_yaw      = self.quat_to_yaw(msg.orientation)
        self._latest_imu_yaw_rate = msg.angular_velocity.z

    def synced_callback(self, msg_gt, msg_ekf, msg_gps):
        self.get_logger().info("Data diterima!")
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
        t = (now - self.start_time).nanoseconds / 1e9
        self.sample_count += 1

        gt  = msg_gt.pose.pose.position
        ekf = msg_ekf.pose.pose.position
        gps = msg_gps.pose.pose.position

        ekf_vx       = msg_ekf.twist.twist.linear.x
        ekf_vy       = msg_ekf.twist.twist.linear.y
        ekf_yaw_rate = msg_ekf.twist.twist.angular.z
        ekf_yaw_deg  = math.degrees(self.quat_to_yaw(msg_ekf.pose.pose.orientation))
        gt_yaw_deg   = math.degrees(self.quat_to_yaw(msg_gt.pose.pose.orientation))
        imu_yaw_deg  = math.degrees(self._latest_imu_yaw) \
                       if self._latest_imu_yaw is not None else ekf_yaw_deg

        error_ekf = math.hypot(gt.x - ekf.x, gt.y - ekf.y)
        error_gps = math.hypot(gt.x - gps.x, gt.y - gps.y)

        self.data['gt']['x'].append(gt.x);   self.data['gt']['y'].append(gt.y)
        self.data['ekf']['x'].append(ekf.x); self.data['ekf']['y'].append(ekf.y)
        self.data['gps']['x'].append(gps.x); self.data['gps']['y'].append(gps.y)
        self.errors_ekf.append(error_ekf)
        self.errors_gps.append(error_gps)

        self.writer.writerow([
            self.sample_count, f'{t:.3f}',
            self.lintasan, self.run_ke,
            f'{gps.x:.6f}', f'{gps.y:.6f}',
            f'{ekf.x:.6f}', f'{ekf.y:.6f}',
            f'{ekf_vx:.6f}', f'{ekf_vy:.6f}',
            f'{ekf_yaw_deg:.4f}', f'{ekf_yaw_rate:.6f}',
            f'{imu_yaw_deg:.4f}', f'{self._latest_imu_yaw_rate:.6f}',
            f'{gt.x:.6f}', f'{gt.y:.6f}', f'{gt_yaw_deg:.4f}',
            f'{error_ekf:.6f}', f'{error_gps:.6f}',
            f'{ekf.x - gt.x:.6f}', f'{ekf.y - gt.y:.6f}',
            f'{gps.x - gt.x:.6f}', f'{gps.y - gt.y:.6f}',
        ])
        self.csv_file.flush()

        if self.sample_count % 10 == 0:
            print(
                f'#{self.sample_count:4d} | t={t:.1f}s | '
                f'err_EKF={error_ekf:.3f}m | '
            )

    def calculate_rmse(self, errors):
        if not errors:
            return 0.0
        return float(np.sqrt(np.mean(np.square(errors))))

    def plot_results(self):
        self.csv_file.close()
        self._stop_rosbag()

        rmse_ekf = self.calculate_rmse(self.errors_ekf)
        rmse_gps = self.calculate_rmse(self.errors_gps)
        n        = len(self.errors_ekf)
        improv   = (rmse_gps - rmse_ekf) / rmse_gps * 100 if rmse_gps > 0 else 0.0

        print('\n' + '='*50)
        print(f'  HASIL EVALUASI — Lintasan {self.lintasan}  Run {self.run_ke}')
        print('='*50)
        print(f'  Jumlah sample : {n}')
        print(f'  RMSE GPS      : {rmse_gps:.4f} m')
        print(f'  RMSE EKF      : {rmse_ekf:.4f} m  ({improv:+.1f}%)')
        print(f'  CSV           : {self.csv_filename}')
        print('='*50)

        if n == 0:
            print('Tidak ada data.')
            return

        # ============================================================
        # Simpan KEDUA plot dulu — baru buka di file manager / viewer
        # Tidak pakai plt.show() supaya tidak ada window yang bisa
        # di-close dan memotong proses penyimpanan plot kedua.
        # ============================================================

        # Plot 1: Trajectory + Buoy
        fig1, ax = plt.subplots(figsize=(9, 8))
        fig1.suptitle(
            f'Trajectory — Lintasan {self.lintasan}  Run {self.run_ke}',
            fontsize=13, fontweight='bold'
        )

        buoy_r = 0.8
        for (bx, by) in BUOYS_RED:
            ax.add_patch(plt.Circle((bx, by), buoy_r, color='red', alpha=0.85, zorder=2))
            ax.add_patch(plt.Circle((bx, by), buoy_r, fill=False,
                         edgecolor='darkred', linewidth=0.8, zorder=2))
        for (bx, by) in BUOYS_GREEN:
            ax.add_patch(plt.Circle((bx, by), buoy_r, color='green', alpha=0.85, zorder=2))
            ax.add_patch(plt.Circle((bx, by), buoy_r, fill=False,
                         edgecolor='darkgreen', linewidth=0.8, zorder=2))
        for (dx, dy) in DOCKS:
            ax.add_patch(plt.Rectangle(
                (dx - 0.4, dy - 0.3), 0.8, 0.6,
                color='saddlebrown', alpha=0.9, zorder=2
            ))

        ax.plot(self.data['gps']['x'], self.data['gps']['y'],
                'r-', lw=0.8, alpha=0.55, zorder=3,
                label=f'GPS  (RMSE={rmse_gps:.3f} m)')
        ax.plot(self.data['ekf']['x'], self.data['ekf']['y'],
                'b-', lw=1.8, alpha=0.90, zorder=4,
                label=f'EKF  (RMSE={rmse_ekf:.3f} m)')
        ax.plot(self.data['gt']['x'], self.data['gt']['y'],
                'k--', lw=1.8, alpha=0.85, zorder=5,
                label='Ground Truth')

        ax.scatter(self.data['gt']['x'][0],  self.data['gt']['y'][0],
                   c='black', s=80, zorder=6, marker='o')
        ax.scatter(self.data['gt']['x'][-1], self.data['gt']['y'][-1],
                   c='black', s=80, zorder=6, marker='s')
        ax.annotate('Start', xy=(self.data['gt']['x'][0],  self.data['gt']['y'][0]),
                    xytext=(6, 6), textcoords='offset points', fontsize=8)
        ax.annotate('End',   xy=(self.data['gt']['x'][-1], self.data['gt']['y'][-1]),
                    xytext=(6, 6), textcoords='offset points', fontsize=8)

        patch_red   = mpatches.Patch(color='red',         label='Buoy merah')
        patch_green = mpatches.Patch(color='green',       label='Buoy hijau')
        patch_dock  = mpatches.Patch(color='saddlebrown', label='Dock')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [patch_red, patch_green, patch_dock],
                  labels  + ['Buoy merah', 'Buoy hijau', 'Dock'],
                  fontsize=8, loc='best')
        ax.set_xlabel('X (meter)')
        ax.set_ylabel('Y (meter)')
        ax.set_title('Trajectory 2D + Posisi Buoy')
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal')

        plt.tight_layout()
        fig1.savefig(self.plot_traj_filename, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f'  Plot 1 disimpan : {self.plot_traj_filename}')

        # Plot 2: Error per Sample
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.suptitle(
            f'Error per Sample — Lintasan {self.lintasan}  Run {self.run_ke}',
            fontsize=13, fontweight='bold'
        )

        samples = list(range(1, n + 1))
        ax2.plot(samples, self.errors_gps, 'r-', lw=1.0, alpha=0.6,
                 label=f'GPS error  (RMSE={rmse_gps:.3f} m)')
        ax2.plot(samples, self.errors_ekf, 'b-', lw=1.3, alpha=0.85,
                 label=f'EKF error  (RMSE={rmse_ekf:.3f} m)')
        ax2.axhline(rmse_gps, color='r', ls='--', lw=0.8, alpha=0.5)
        ax2.axhline(rmse_ekf, color='b', ls='--', lw=0.8, alpha=0.5)
        ax2.set_xlabel('Sample #')
        ax2.set_ylabel('Error (meter)')
        ax2.set_title('Error Deviasi Posisi per Sample')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2.savefig(self.plot_err_filename, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'  Plot 2 disimpan : {self.plot_err_filename}')

        try:
            subprocess.Popen(['xdg-open', self.plot_traj_filename])
            subprocess.Popen(['xdg-open', self.plot_err_filename])
        except Exception:
            pass

        print('\nKedua file PNG sudah tersimpan di direktori saat ini.')

    def quat_to_yaw(self, q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        return math.atan2(siny, cosy)


def main(args=None):
    rclpy.init(args=args)
    node = EKFEvaluatorWithCSV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.plot_results()
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()