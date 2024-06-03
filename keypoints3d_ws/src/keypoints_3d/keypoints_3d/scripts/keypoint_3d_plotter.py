#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from pose_msgs.msg import Keypoint, Keypoint3D

def callback(keypoint_msg):
    cloud = PointCloud()
    cloud.header = keypoint_msg.header
    cloud.points = [kp.all_cameras_point for kp in keypoint_msg.keypoints]
    
    pub.publish(cloud)
    #rospy.loginfo("Published PointCloud with %d points", len(cloud.points))

def listener():
    rospy.init_node('keypoint_plotter', anonymous=True)
    rospy.Subscriber("/keypoints_3d", Keypoint3D, callback)
    rospy.spin()

if __name__ == '__main__':
    pub = rospy.Publisher('keypoints_cloud', PointCloud, queue_size=10)
    listener()
