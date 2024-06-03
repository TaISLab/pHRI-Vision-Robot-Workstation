#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def callback(keypoints_cloud):
    pub = rospy.Publisher('skeleton_markers', Marker, queue_size=10)
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.type = marker.LINE_LIST
    marker.action = marker.ADD
    marker.scale.x = 0.02
    marker.pose.orientation.w = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    """
    connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
        (5, 17), (6, 8), (6, 17), (7, 9), (8, 10), (11, 12), (11, 13),
        (11, 17), (12, 14), (12, 17), (13, 15), (14, 16), (0, 17)
    ]
    """
    
    connections = [
        (5, 7),
        (5, 17), (6, 8), (6, 17), (7, 9), (8, 10), (11, 12), (11, 13),
        (11, 17), (12, 14), (12, 17), (13, 15), (14, 16), (0, 17)    
    ]
    
    for start_idx, end_idx in connections:
        start_point = keypoints_cloud.points[start_idx]
        end_point = keypoints_cloud.points[end_idx]
        if not (is_zero(start_point) or is_zero(end_point)):
            marker.points.append(Point(start_point.x, start_point.y, start_point.z))
            marker.points.append(Point(end_point.x, end_point.y, end_point.z))
    
    pub.publish(marker)

def is_zero(point):
    return point.x == 0 and point.y == 0 and point.z == 0

def listener():
    rospy.init_node('keypoints_skeleton_publisher', anonymous=True)
    rospy.Subscriber('/keypoints_cloud', PointCloud, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
