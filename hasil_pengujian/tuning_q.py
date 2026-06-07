#!/usr/bin/env python3
"""
Cara pakai:
  python3 tuning_q.py <folder_rosbag>
"""

import sys, math
import numpy as np
from baca_bag import read_bag

# Grid nilai Q
Q_pos_vals = [5e-3, 1e-3, 5e-4]   # noise posisi  (x, y)
Q_vel_vals = [5e-4, 1e-4, 5e-5]   # noise kecepatan (vx, vy)
Q_yaw_vals = [1e-5, 5e-6, 1e-6]   # noise yaw

# Parameter R
R_GPS  = np.diag([0.18, 0.18])
R_YAW  = 0.005
R_RATE = 0.000019

# Helper
def norm_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

# PURE EKF — satu run offline
def run_pure_ekf(Q_diag,
                 t_imu, yaw_imu, rate_imu, ax_imu, ay_imu,
                 t_gps, x_gps, y_gps,
                 t_gt,  x_gt,  y_gt):

    state = np.zeros(6, dtype=float)
    P     = np.identity(6, dtype=float)
    Q     = np.diag(Q_diag)

    initialized = False
    gps_ptr     = 0
    prev_t      = None
    res_x, res_y, res_t = [], [], []

    def _norm(a): return norm_angle(a)

    def _update(z_arr, H, R_mat):
        nonlocal state, P
        inn = z_arr - H @ state
        if H.shape[0] == 1 and abs(H[0, 4]) > 0.5:
            inn[0] = _norm(inn[0])
        S = H @ P @ H.T + R_mat
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        state += K @ inn
        state[4] = _norm(state[4])
        IKH = np.eye(6) - K @ H
        P[:] = IKH @ P
        P[:] = (P + P.T) / 2.0

    def _predict(dt):
        nonlocal state, P
        if dt <= 0 or dt > 1.0:
            return
        x, y, vx, vy, yaw, yr = state
        cy, sy = math.cos(yaw), math.sin(yaw)
        x_new   = x + (vx*cy - vy*sy)*dt
        y_new   = y + (vx*sy + vy*cy)*dt
        yaw_new = _norm(yaw + yr*dt)
        vx_new  = vx * 0.98
        vy_new  = vy * 0.98
        state[:] = [x_new, y_new, vx_new, vy_new, yaw_new, yr]
        F = np.eye(6)
        F[0,2] =  cy*dt;  F[0,3] = -sy*dt
        F[0,4] = (-vx*sy - vy*cy)*dt
        F[1,2] =  sy*dt;  F[1,3] =  cy*dt
        F[1,4] = ( vx*cy - vy*sy)*dt
        F[2,2] = 0.98
        F[3,3] = 0.98
        F[4,5] = dt
        P[:] = F @ P @ F.T + Q

    H_yaw  = np.zeros((1, 6)); H_yaw[0, 4]  = 1.0
    H_rate = np.zeros((1, 6)); H_rate[0, 5] = 1.0

    for k in range(len(t_imu)):
        tk = t_imu[k]
        if prev_t is None:
            prev_t = tk
            continue
        dt     = tk - prev_t
        prev_t = tk
        if dt <= 0 or dt > 0.5:
            continue

        while gps_ptr < len(t_gps) and t_gps[gps_ptr] < tk - 0.06:
            gps_ptr += 1

        if initialized:
            _predict(dt)

        _update(np.array([yaw_imu[k]]),  H_yaw,  np.array([[R_YAW]]))
        _update(np.array([rate_imu[k]]), H_rate, np.array([[R_RATE]]))

        if gps_ptr < len(t_gps) and abs(tk - t_gps[gps_ptr]) < 0.06:
            gx = x_gps[gps_ptr]
            gy = y_gps[gps_ptr]
            if not initialized:
                state[0] = gx; state[1] = gy; state[4] = yaw_imu[k]
                initialized = True
            else:
                z = np.array([gx, gy])
                H = np.zeros((2, 6)); H[0,0]=1.0; H[1,1]=1.0
                _update(z, H, R_GPS)
            gps_ptr += 1

        if initialized:
            res_x.append(state[0])
            res_y.append(state[1])
            res_t.append(tk)

    if len(res_x) < 10:
        return 999.0, [], [], []

    x_gt_i = np.interp(res_t, t_gt, x_gt)
    y_gt_i = np.interp(res_t, t_gt, y_gt)
    rmse   = float(np.sqrt(np.mean(
        (np.array(res_x) - x_gt_i)**2 +
        (np.array(res_y) - y_gt_i)**2
    )))
    return rmse, res_x, res_y, res_t

# GPS-only RMSE baseline
def rmse_gps_only(t_gps, x_gps, y_gps, t_gt, x_gt, y_gt):
    errors = []
    for i in range(len(t_gps)):
        idx = int(np.argmin(np.abs(t_gt - t_gps[i])))
        if abs(t_gt[idx] - t_gps[i]) > 0.06:
            continue
        errors.append(math.hypot(x_gps[i]-x_gt[idx], y_gps[i]-y_gt[idx]))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

# Load data
def load_data(bag_path):
    print(f'Membaca rosbag: {bag_path}')
    gps_xy, imu_data, gt_data = read_bag(bag_path)
    t0 = imu_data[0]['time']
    t_imu    = np.array([d['time']-t0 for d in imu_data])
    yaw_imu  = np.array([d['yaw']     for d in imu_data])
    rate_imu = np.array([d['rate']    for d in imu_data])
    ax_imu   = np.array([d['ax']      for d in imu_data])
    ay_imu   = np.array([d['ay']      for d in imu_data])
    t_gps    = np.array([d['time']-t0 for d in gps_xy])
    x_gps    = np.array([d['x']       for d in gps_xy])
    y_gps    = np.array([d['y']       for d in gps_xy])
    t_gt     = np.array([d['time']-t0 for d in gt_data])
    x_gt     = np.array([d['x']       for d in gt_data])
    y_gt     = np.array([d['y']       for d in gt_data])
    print(f'  IMU : {len(t_imu):,} sampel  ({t_imu[-1]-t_imu[0]:.1f} detik)')
    print(f'  GPS : {len(t_gps):,} sampel')
    print(f'  GT  : {len(t_gt):,} sampel')
    return (t_imu, yaw_imu, rate_imu, ax_imu, ay_imu,
            t_gps, x_gps, y_gps, t_gt, x_gt, y_gt)

# MAIN
def main():
    if len(sys.argv) < 2:
        print('Cara pakai: python3 tuning_q.py <folder_rosbag>')
        sys.exit(1)

    bag_path = sys.argv[1].rstrip('/')
    bag_name = bag_path.split('/')[-1]

    data = load_data(bag_path)
    (t_imu, yaw_imu, rate_imu, ax_imu, ay_imu,
     t_gps, x_gps, y_gps, t_gt, x_gt, y_gt) = data

    r_gps = rmse_gps_only(t_gps, x_gps, y_gps, t_gt, x_gt, y_gt)
    print(f'\nGPS-only RMSE : {r_gps:.4f} m\n')

    sep = '=' * 55
    print(sep)
    print(f'{"Q_pos":>8} {"Q_vel":>8} {"Q_yaw":>8} {"RMSE":>12}')
    print(sep)

    best = {'rmse': 999.0, 'rx': [], 'ry': [], 'Q': None}
    all_results = []

    for qp in Q_pos_vals:
        for qv in Q_vel_vals:
            for qy in Q_yaw_vals:
                Q_diag = [qp, qp, qv, qv, qy, qy]
                rmse, rx, ry, _ = run_pure_ekf(Q_diag, *data)

                tag = ''
                if rmse < best['rmse']:
                    best.update({'rmse': rmse, 'rx': rx, 'ry': ry, 'Q': Q_diag})
                    tag = ' ← BEST'

                print(f'{qp:>8.0e} {qv:>8.0e} {qy:>8.0e} {rmse:>12.4f}{tag}')
                all_results.append({'qp': qp, 'qv': qv, 'qy': qy, 'rmse': rmse})

    print(sep)

    pct = (r_gps - best['rmse']) / r_gps * 100
    print(f'\n{"─"*55}')
    print(f'  GPS-only RMSE : {r_gps:.4f} m')
    print(f'  Best EKF RMSE : {best["rmse"]:.4f} m  ({pct:+.1f}%)  Q={best["Q"]}')
    print(f'{"─"*55}')

    with open(f'tuning_q_pure_{bag_name}.txt', 'w') as f:
        f.write(f'Bag      : {bag_name}\n')
        f.write(f'GPS RMSE : {r_gps:.4f} m\n\n')
        f.write(f'{"Q_pos":>8} {"Q_vel":>8} {"Q_yaw":>8} {"RMSE":>12}\n')
        f.write('='*40 + '\n')
        for r in sorted(all_results, key=lambda x: x['rmse']):
            f.write(f'{r["qp"]:>8.0e} {r["qv"]:>8.0e} {r["qy"]:>8.0e} {r["rmse"]:>12.4f}\n')
        f.write(f'\nBest:\n  Q    = {best["Q"]}\n  RMSE = {best["rmse"]:.4f} m\n')
    print(f'Tabel disimpan: tuning_q_pure_{bag_name}.txt')

if __name__ == '__main__':
    main()