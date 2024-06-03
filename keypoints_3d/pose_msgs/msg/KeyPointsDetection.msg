# This msg describes a single person detection keypoints.
#
# Points are given as vectors from the camera frame (u,v,1) or (0,0,0) 
#  if that keypoint was not detected
#
# Keypoints are listed according to human_pose.json at ros_trt_pose/models
#  0 nose
#  1 left_eye
#  2 right_eye
#  3 left_ear
#  4 right_ear
#  5 left_shoulder
#  6 right_shoulder
#  7 left_elbow
#  8 right_elbow
#  9 left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle
# 17 neck

geometry_msgs/Point[18] keypoints
