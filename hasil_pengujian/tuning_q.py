#!/usr/bin/env python3
"""
tuning_q_pure.py
================
Tuning Q untuk PURE EKF.

Cara pakai:
  python3 tuning_q_pure.py <folder_rosbag>
"""

import sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from baca_bag import read_bag

# ══════════════════════════════════════════════════════════════
# Grid nilai Q
# ══════════════════════════════════════════════════════════════
Q_pos_vals = [5e-3, 1e-3, 5e-4]          # noise posisi  (x, y)
Q_vel_vals = [5e-4, 1e-4, 5e-5]   # noise kecepatan (vx, vy)
Q_yaw_vals = [1e-5, 5e-6, 1e-6]   # noise yaw

# ── Parameter R ───────────────────────────────────────────────
R_GPS  = np.diag([0.18, 0.18])
R_YAW  = 0.005
R_RATE = 0.000019

# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════
def norm_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

# ══════════════════════════════════════════════════════════════
# PURE EKF — satu run offline
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# GPS-only RMSE baseline
# ══════════════════════════════════════════════════════════════
def rmse_gps_only(t_gps, x_gps, y_gps, t_gt, x_gt, y_gt):
    errors = []
    for i in range(len(t_gps)):
        idx = int(np.argmin(np.abs(t_gt - t_gps[i])))
        if abs(t_gt[idx] - t_gps[i]) > 0.06:
            continue
        errors.append(math.hypot(x_gps[i]-x_gt[idx], y_gps[i]-y_gt[idx]))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0

# ══════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════
def plot_results(x_gt, y_gt, x_gps, y_gps, best, r_gps, all_results, bag_name):
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle(
        f'Tuning Q — Pure EKF  |  {bag_name}\n'
        f'GPS-only RMSE={r_gps:.4f}m   Best EKF={best["rmse"]:.4f}m'
        f'  ({(r_gps-best["rmse"])/r_gps*100:+.1f}%)',
        color='white', fontsize=12, fontweight='bold', y=0.99
    )
    DARK='#1a1a2e'; GRID='#2a2a4a'; TXT='#e0e0e0'

    def sa(ax, title, xl='', yl=''):
        ax.set_facecolor(DARK)
        ax.set_title(title, color=TXT, fontsize=10)
        ax.set_xlabel(xl, color=TXT, fontsize=8)
        ax.set_ylabel(yl, color=TXT, fontsize=8)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, lw=0.5, ls='--', alpha=0.5)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.90, bottom=0.08, wspace=0.32)
    ax_traj = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    sa(ax_traj, 'Trajectory: GT vs Pure EKF (Q Terbaik)', 'X (m)', 'Y (m)')
    ax_traj.plot(x_gt,  y_gt,  color='#00e5ff', lw=2.0, label='Ground Truth', zorder=4)
    ax_traj.plot(x_gps, y_gps, color='#ff4d6d', lw=0.8, alpha=0.5,
                 label=f'GPS-only  {r_gps:.4f}m', zorder=2)
    ax_traj.plot(best['rx'], best['ry'], color='#a259ff', lw=1.5, ls='--', alpha=0.85,
                 label=f'Best EKF  {best["rmse"]:.4f}m', zorder=3)
    ax_traj.scatter(x_gt[0],  y_gt[0],  c='white',  s=80, zorder=6, marker='o')
    ax_traj.scatter(x_gt[-1], y_gt[-1], c='#ffdd57', s=80, zorder=6, marker='X')
    ax_traj.legend(facecolor='#252545', edgecolor=GRID, labelcolor=TXT, fontsize=8)
    ax_traj.set_aspect('equal', adjustable='datalim')

    sa(ax_bar, 'Top-10 Kombinasi Q (RMSE terkecil)', 'Kombinasi Q', 'RMSE (m)')
    top10  = sorted(all_results, key=lambda r: r['rmse'])[:10]
    labels = [f"pos={r['qp']:.0e}\nvel={r['qv']:.0e}\nyaw={r['qy']:.0e}" for r in top10]
    x_idx  = np.arange(len(top10))
    ax_bar.bar(x_idx, [r['rmse'] for r in top10], 0.5,
               color='#a259ff', alpha=0.85, label='EKF')
    ax_bar.axhline(r_gps, color='#ff4d6d', lw=1.2, ls='--',
                   label=f'GPS-only {r_gps:.4f}m')
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels(labels, fontsize=6, color=TXT)
    ax_bar.legend(facecolor='#252545', edgecolor=GRID, labelcolor=TXT, fontsize=8)

    plt.tight_layout()
    out = f'tuning_q_pure_{bag_name}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'\nPlot disimpan: {out}')

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print('Cara pakai: python3 tuning_q_pure.py <folder_rosbag>')
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

    plot_results(x_gt, y_gt, x_gps, y_gps, best, r_gps, all_results, bag_name)

if __name__ == '__main__':
    main()