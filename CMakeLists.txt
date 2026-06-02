cmake_minimum_required(VERSION 3.8)
project(asv_usu)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(
  PROGRAMS
    scripts/gps_only_odom.py
    scripts/vo_validation_node2.py
    scripts/evaluator_node.py
    scripts/teleop_catamaaran.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY urdf launch worlds meshes models parameters rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()