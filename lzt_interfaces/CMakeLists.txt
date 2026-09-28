cmake_minimum_required(VERSION 3.8)
project(lzt_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectPoseHypothesis.msg"
  "msg/GraspCandidate.msg"
  "msg/JointPath.msg"
  "srv/PlanJointPath.srv"
  "action/ExecuteJointTrajectory.action"
  DEPENDENCIES geometry_msgs std_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

