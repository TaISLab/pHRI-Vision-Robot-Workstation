cmake_minimum_required(VERSION 3.0.2)

project(keypoints_3d)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
   scripts/keypoint_3d.py
   scripts/keypoint_3d_kalman.py
   scripts/keypoint_3d_plotter.py
   scripts/keypoint_skeleton.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
      launch 
      config 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
