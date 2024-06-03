#!/usr/bin/env python3

# ROS related
import rospy
from pose_msgs.msg import FrameDetection, KeyPointsDetection
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

import sys




import tf2_ros
from tf2_geometry_msgs import PointStamped
from tf2_geometry_msgs import do_transform_point




class KeyPointsPlotter():
        
    def __init__(self):
        self.my_log_info("Creating node")        

        # ROS parameters .......................................................
        self.frame_detections_sub_topic_name = rospy.get_param('~frame_detections_pub_topic_name', '/camera2/frame_detections')        
        self.marker_pub_topic_name = rospy.get_param('~marker_pub_topic_name', '/camera2/frame_detection_markers')        

        # ROS publishers .......................................................
        self.marker_pub = rospy.Publisher(self.marker_pub_topic_name, MarkerArray, queue_size=10)
        
        # define a tf2 transform buffer and pass it to a listener
        self.tf2_buffer = tf2_ros.Buffer() #  rospy.Duration(5) ??
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        
        # ROS subscribers ......................................................
        self.frame_detections_sub = rospy.Subscriber(self.frame_detections_sub_topic_name, FrameDetection, self.frame_callback)
        
        
        self.my_log_info("Node created\n")
        
    def frame_callback(self, frame_msg):
        num_persons = len(frame_msg.persons)
        keypointMarkers = MarkerArray()
        if (num_persons >=1):
            keypoints_i = frame_msg.persons[0].keypoints
            point_k = keypoints_i[9]
            
            # create a point along the given vector
            arrow_len = 3
            point_k.x = point_k.x * arrow_len
            point_k.y = point_k.y * arrow_len
            point_k.z = point_k.z * arrow_len        
        
            if point_k.z>0:
                #self.my_log_info("Left hand X:{:.2f}, Y:{:.2f}, Z:{:.2f} ".format(point_k.x,point_k.y,point_k.z))
                #self.my_log_info("")                      
                #self.my_log_info("")     
                
                # This creates a line marker between optical frame origin and point along the vector
                #marker_i = self.make_origin_marker(point_k,frame_msg.header)
                
                # As above, but first transforms both points into base_link frame
                marker_i = self.make_origin_marker_base_link(point_k,frame_msg.header)
                
                keypointMarkers.markers.append(marker_i)
                self.marker_pub.publish(keypointMarkers)
            
            
    def transform_point(self, a_point, a_header, to_frame):
        # Put together the input data
        ps = PointStamped()        
        ps.header = a_header
        ps.point = a_point
        
        transform_ok = False
        while not transform_ok and not rospy.is_shutdown():        
            t = rospy.Time()
            trans = self.tf2_buffer.lookup_transform(to_frame, a_header.frame_id, t)                
            target_ps = do_transform_point(ps, trans)                            
            transform_ok = True

        # We don't need resulting header, we know its value
        o_point = target_ps.point
        
        #self.my_log_info("{:.2f}, {:.2f}, {:.2f} ".format(a_point.x,a_point.y,a_point.z))
        #self.my_log_info("in frame_id: "+ str(a_header.frame_id))
        #self.my_log_info("is: ")
        #self.my_log_info("{:.2f}, {:.2f}, {:.2f} ".format(o_point.x,o_point.y,o_point.z))
        #self.my_log_info("in frame_id: "+ str(to_frame))
        #self.my_log_info("...............................")
        
        return o_point
    
    def my_log_info(self, text):
        rospy.loginfo("[" + rospy.get_name() + "]: " + text)                

    def my_log_debug(self, text):
        rospy.logdebug("[" + rospy.get_name() + "]: " + text)    

    def my_log_err(self, text):
        rospy.logerr("[" + rospy.get_name() + "]: " + text)  
        
    def make_origin_marker_base_link(self, end_point, a_header):
        # transform start point 
        start_point = Point()
        start_point.x = start_point.y = start_point.z = 0
        start_point_base_link = self.transform_point(start_point, a_header, 'base_link')

        # transform end point
        end_point_base_link = self.transform_point(end_point, a_header, 'base_link')

        # resulting header
        header_base_link = a_header
        header_base_link.frame_id = 'base_link'
        
        
        m = self.make_marker(start_point_base_link, end_point_base_link, header_base_link)
        return m
    
    def make_origin_marker(self, end_point, a_header):
        start_point = Point()
        start_point.x = start_point.y = start_point.z = 0
        m = self.make_marker(start_point, end_point, a_header)
        return m
    
    def make_marker(self, start_point, end_point, a_header):
        m = Marker()

        #self.my_log_info("Plotting marker between:")
        #self.my_log_info("{:.2f}, {:.2f}, {:.2f} ".format(start_point.x,start_point.y,start_point.z))
        #self.my_log_info("and:")
        #self.my_log_info("{:.2f}, {:.2f}, {:.2f} ".format(end_point.x,end_point.y,end_point.z))
        #self.my_log_info("in frame_id: "+ str(a_header.frame_id))
        #self.my_log_info("")
        
        # Start/End Points        
        # index 0: start 
        m.points.append(start_point)
        # index 1:end        
        arrow_len = 3
        end_point.x = end_point.x * arrow_len
        end_point.y = end_point.y * arrow_len
        end_point.z = end_point.z * arrow_len
        m.points.append(end_point)

        # avoid warning
        m.pose.orientation.w = 1
        # shaft diameter
        m.scale.x = 0.02
        # head diameter
        m.scale.y = 0.02
        # head length
        m.scale.z = 0.06

        m.action = Marker.ADD
        
        # how long till removal
        m.lifetime = rospy.Duration(0.05)
        
        m.header = a_header
        m.ns = 'marker_test'
        m.id = 0
        m.type = Marker.ARROW
        m.color.r = 1.0
        m.color.g = 0.2
        m.color.b = 0.2
        m.color.a = 0.6        
        return m  
        
def main(args):
  rospy.init_node('keypoint_plotter_node', anonymous=True)
  node = KeyPointsPlotter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down trt pose node")

if __name__ == '__main__':
    main(sys.argv)
        
