#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os, xacro

def generate_launch_description():

    pkg_share  = get_package_share_directory('asv_usu')
    urdf_file  = os.path.join(pkg_share, 'urdf', 'model.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'my_world.sdf')

    robot_description_config = xacro.process_file(urdf_file).toxml()

    # === Gazebo ===
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': f'-r -v4 {world_file}'}.items()
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'catamaaran1',
            '-topic', 'robot_description',
            '-x', '-0.24099596970477125',
            '-y', '6.8790175316033544',
            '-z', '0.2',
            '-Y', '0',
        ],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_config,
            'use_sim_time': True
        }]
    )

    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    bridge_params = os.path.join(pkg_share, 'parameters', 'bridge_parameters.yaml')
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen'
    )

    rviz_config = os.path.join(pkg_share, 'rviz', 'robot.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )

    # === Sensor Fusion Nodes ===
    gps_only_node = Node(
        package='asv_usu',
        executable='gps_only_odom.py',
        name='gps_only_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    ekf_node = Node(
        package='asv_usu',
        executable='vo_validation_node2.py',
        name='ekf_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    evaluator_node = Node(
        package='asv_usu',
        executable='evaluator_node.py',
        name='evaluator_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([

        LogInfo(msg='[T=0s] Menjalankan Gazebo...'),
        gazebo_launch,
        robot_state_publisher,
        joint_state_publisher,
        ros_gz_bridge,
        rviz,

        TimerAction(period=5.0, actions=[
            LogInfo(msg='[T=5s] Spawn robot...'),
            spawn_robot,
        ]),

        TimerAction(period=30.0, actions=[
            LogInfo(msg='[T=30s] Menjalankan sensor fusion nodes...'),
            gps_only_node,
            ekf_node,
            evaluator_node,
        ]),
    ])