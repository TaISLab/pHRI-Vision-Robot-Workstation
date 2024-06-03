#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import least_squares
import message_filters
from pose_msgs.msg import FrameDetection
from geometry_msgs.msg import Point32, PointStamped
from std_msgs.msg import Header
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from pose_msgs.msg import Keypoint, Keypoint3D, AnglesDetection, Angle

from sensor_msgs.msg import Image, CameraInfo
import cv2
# always import cv2 before cv_bridge 
# See https://answers.ros.org/question/362388/cv_bridge_boost-raised-unreported-exception-when-importing-cv_bridge/
from cv_bridge import CvBridge, CvBridgeError
import image_geometry
import math

class KeyPointsProcessor:
    def __init__(self):
        rospy.init_node('keypoint_processor_node', anonymous=True)
        self.publisher = rospy.Publisher('/keypoints_3d', Keypoint3D, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.subscribers = [
            message_filters.Subscriber(f'/camera{i}/frame_detections', FrameDetection)
            for i in range(1, 5)
        ]
        self.images_info_sub = [
            message_filters.Subscriber(f'/camera{i}/color/camera_info', CameraInfo)
            for i in range(1, 5)
        ]
        self.ats = message_filters.ApproximateTimeSynchronizer(self.subscribers + self.images_info_sub, queue_size=10, slop=0.1)
        self.ats.registerCallback(self.callback)
        self.reference_keypoints = None
        self.initialized = False
        self.selected_persons = [None] * 4

    # Gets camera model
    def imageInfoCallback(self, cameraInfo):
        try:
            if hasattr(self, "camera_model"):
                return

            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(cameraInfo)

        except CvBridgeError as e:
            print(e)
            self.my_log_err("Cant set model: " + e )
            return


    # Thanks to https://answers.ros.org/question/195737/how-to-get-coordinates-from-depthimage/
    # Given a model from CameraInfo message, 3D ray (x,y,1)
    # This returns pixel coordinates (u, v)
    def vect_2_pixel(self, kp):
        if not hasattr(self, "camera_model"):
           return
            
        kp2D = np.array( self.camera_model.project3dToPixel((kp.x,kp.y,kp.z)) )        
        return kp2D


    # Transforms point from local frame to base_link (global frame)
    def transform_point(self, point, header, to_frame='base_link'):
        point_stamped = PointStamped(header=header, point=point)
        try:
            transformed_point = self.tf_buffer.transform(point_stamped, to_frame)
            return np.array([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # rospy.logwarn(f"Error transforming point from {header.frame_id} to {to_frame}: {e}")
            return np.array([0, 0, 0])


    # Detects person who raises right arm based on the pixel positions of two keypoints (right wrist and right shoulder)
    def select_persons_initial(self, msgs, info_msgs):
        if not self.initialized:
            for i, msg in enumerate(msgs):
                self.imageInfoCallback(info_msgs[i])
                for person in msg.persons:
                    right_shoulder = person.keypoints[6]
                    right_wrist = person.keypoints[10]
                    right_shoulder_2D = self.vect_2_pixel(right_shoulder)
                    right_wrist_2D = self.vect_2_pixel(right_wrist)
                    if right_wrist_2D[1] < right_shoulder_2D[1]:
                        self.selected_persons[i] = person
                        break

            if all(self.selected_persons): # check if all cameras have detected a person with a raised hand
                self.reference_keypoints = [p.keypoints for p in self.selected_persons]
                self.initialized = True # initial person has been succesfully selected


    # Tracks selected person
    def select_persons_based_on_reference(self, msgs):
        selected_persons = []
        for msg in msgs:
            min_cost = float('inf')
            selected_person = None
            for person in msg.persons:
                cost = self.calculate_cost(self.reference_keypoints[msgs.index(msg)], person.keypoints)
                if cost < min_cost:
                    min_cost = cost
                    selected_person = person
            selected_persons.append(selected_person)
        return selected_persons

    # Compares two keypoint skeletons and computes cost
    def calculate_cost(self, ref_keypoints, new_keypoints):
        distances = []
        for ref_kp, new_kp in zip(ref_keypoints, new_keypoints):
            if ref_kp.z > 0: # only considers non-zero reference keypoints to compute cost
                ref = np.array([ref_kp.x, ref_kp.y, ref_kp.z])
                new = np.array([new_kp.x, new_kp.y, new_kp.z])
                distances.append(np.linalg.norm(ref - new))
        if not distances:
            return float('inf')
        return np.mean(distances)

    ## 3D point estimation functions
    # Computes optimal 3D point from 4 spatial rays, previously transformed from camera frame to a global reference system (base_link)
    def calculate_points(self, origins, directions):
        origins, directions = self.ray_filtering(origins, directions)
        left_point, left_cost, num_cameras_left = np.array([0,0,0]), -1, 0 # only uses left cameras to estimate 3D point
        right_point, right_cost, num_cameras_right = np.array([0,0,0]), -1, 0 # only uses right cameras to estimate 3D point
        #left_point, left_cost, num_cameras_left = self.find_optimal_point_rays(origins[[1, 2],:], directions[[1, 2],:])
        #right_point, right_cost, num_cameras_right = self.find_optimal_point_rays(origins[[0, 3],:], directions[[0, 3],:])
        if len(origins) < 3: # force at least three rays to compute 3D point (all_cameras_point case)
            all_cameras_point = np.array([0,0,0])
            all_cameras_cost = -1
        else:
            all_cameras_point, all_cameras_cost, num_cameras = self.find_optimal_point_rays(origins, directions) # uses all 4 cameras (if available)
        return left_point, left_cost, right_point, right_cost, all_cameras_point, all_cameras_cost


    # Filters valid rays, ignore [0,0,0] (undetected keypoints are filtered)
    def ray_filtering(self, origins, directions):
        valid_indices = np.any(origins != np.array([0,0,0]), axis=1)
        return origins[valid_indices], directions[valid_indices] # return non-zero directions


    # Finds the optimal point closer to multiple rays
    def find_optimal_point_rays(self, origins, directions):
        if len(origins) == 0:
            return np.array([0, 0, 0]), -1, 0  # no keypoint detected, not enough rays
        if len(origins) == 1:
            return np.array([0, 0, 0]), -1, 1  # only one keypoint detected, not enough rays 
        initial_point = np.mean(origins, axis=0) # seed, initial 3D point estimated as the mean of camera origins
        result = least_squares(self.residuals, initial_point, args=(origins, directions), loss = 'soft_l1', max_nfev = 10) # call least_squared method, max 10 iterations
        return result.x, result.cost, len(origins)
    

    # Residuals for least_squares optimization
    def residuals(self, point, origins, directions):
        return [self.distance_point_to_ray(point, origins[i], directions[i]) for i in range(len(origins))]


    # Calculates the distance from a point in space to a ray
    def distance_point_to_ray(self, point, ray_origin, ray_direction):
        ray_direction = ray_direction / np.linalg.norm(ray_direction) # ensure ray_direction is normalized
        v = point - ray_origin
        projection = np.dot(v, ray_direction) * ray_direction         # project v onto the ray_direction vector
        distance_vector = v - projection                              # distance between 3D point and ray
        return np.linalg.norm(distance_vector)                        # value to be minimized, spatial distance between the point and the ray


    def vect2esf(self,x,y,z):
        ro = np.sqrt(x**2+y**2+z**2)
        if ro == 0:
            return 0, 0 # avoid division by zero
        Cbeta = z/ro
        Sbeta = np.sqrt((x**2+y**2)/(ro**2))
        beta = np.arctan2(Sbeta,Cbeta)
        Calpha = x/(ro*Sbeta)
        Salpha = y/(ro*Sbeta)
        alpha = np.arctan2(Salpha,Calpha)
        return alpha, beta

    
    def callback(self, msg1, msg2, msg3, msg4, info_msg1, info_msg2, info_msg3, info_msg4):
        info_msgs = [info_msg1, info_msg2, info_msg3, info_msg4]
        msgs = [msg1, msg2, msg3, msg4]
        if not self.initialized:
            self.select_persons_initial(msgs,info_msgs)
            selected_persons = self.selected_persons
        else:
            selected_persons = self.select_persons_based_on_reference(msgs)
            self.reference_keypoints = [p.keypoints for p in selected_persons]

        if all(selected_persons):
            keypoints_msg = Keypoint3D(header=Header(stamp=rospy.Time.now(), frame_id="base_link"))
            for person in selected_persons: # for each camera
                angles_detection = AnglesDetection()
                angles_detection.rays = [Angle(*self.vect2esf(kp.x, kp.y, kp.z)) for kp in person.keypoints]
                keypoints_msg.cameras.append(angles_detection)

            for i in range(18):  # for each keypoint
                origins = []
                directions = []
                for person in selected_persons:
                    kp = person.keypoints[i]  # extract keypoints
                    if kp.z > 0:
                        origin = self.transform_point(Point32(0, 0, 0), msgs[selected_persons.index(person)].header)  # base_link as the new system of reference
                        vector_transformed = self.transform_point(Point32(kp.x, kp.y, kp.z),  msgs[selected_persons.index(person)].header)
                        direction_transformed = vector_transformed - origin
                    else:
                        origin = np.array([0,0,0]) # keypoint undetected
                        direction_transformed = np.array([0,0,0]) # keypoint undetected

                    origins.append(origin)
                    directions.append(direction_transformed)

                origins_np = np.array(origins)
                directions_np = np.array(directions)
                left_point, left_cost, right_point, right_cost, all_cameras_point, all_cameras_cost = self.calculate_points(origins_np,directions_np) # calculate 3d keypoints

                kp = Keypoint(
                    id=i,
                    left_point=Point32(*left_point),
                    right_point=Point32(*right_point),
                    all_cameras_point=Point32(*all_cameras_point),
                    left_cost=left_cost,
                    right_cost=right_cost,
                    all_cameras_cost=all_cameras_cost
                )
                keypoints_msg.keypoints.append(kp)
            self.publisher.publish(keypoints_msg)


if __name__ == '__main__':
    processor = KeyPointsProcessor()
    rospy.spin()

