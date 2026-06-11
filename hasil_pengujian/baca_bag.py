#!/usr/bin/env python3
# baca_bag.py
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import math

def quat_to_yaw(x, y, z, w):
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

def read_bag(bag_path):
    typestore = get_typestore(Stores.ROS2_JAZZY)
    gps_data = []
    imu_data = []
    gt_data  = []

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            t = timestamp * 1e-9

            if connection.topic == '/gps_only/odom':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                gps_data.append({
                    'time': t,
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                })

            elif connection.topic == '/boat/imu/data':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                q = msg.orientation
                imu_data.append({
                    'time': t,
                    'yaw':  quat_to_yaw(q.x, q.y, q.z, q.w),
                    'rate': msg.angular_velocity.z,
                    'ax':   msg.linear_acceleration.x,
                    'ay':   msg.linear_acceleration.y,
                })

            elif connection.topic == '/odom':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                gt_data.append({
                    'time': t,
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                })

    # TIDAK dinormalisasi — GPS dan GT sudah dalam frame world Gazebo
    # karena gps_only_node.py menggunakan GT sebagai referensi awal

    print(f"  GPS : {len(gps_data)} data")
    print(f"  IMU : {len(imu_data)} data")
    print(f"  GT  : {len(gt_data)} data")

    return gps_data, imu_data, gt_data