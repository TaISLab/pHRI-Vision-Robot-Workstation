#!/usr/bin/env python3

# ROS related
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from pose_msgs.msg import FrameDetection, KeyPointsDetection
from geometry_msgs.msg import Point
import rospkg

# Image related
import cv2
# always import cv2 before cv_bridge 
# See https://answers.ros.org/question/362388/cv_bridge_boost-raised-unreported-exception-when-importing-cv_bridge/
from cv_bridge import CvBridge, CvBridgeError  
import PIL.Image

# TRT_pose related
import torch
import torchvision.transforms as transforms
import trt_pose.coco
import trt_pose.models
from trt_pose.parse_objects import ParseObjects
import torch2trt
from torch2trt import TRTModule

# Others
import numpy as np
import math
import os
import sys
import json
import image_geometry

#kelly_colors
PERSON_COLORS = [ (193, 0, 32),     # vivid_red
                  (255, 179, 0),    # vivid_yellow
                  (128, 62, 117),   # strong_purple
                  (255, 104, 0),    # vivid_orange
                  (166, 189, 215),  # very_light_blue                  
                  (206, 162, 98),   # grayish_yellow
                  (129, 112, 102),  # medium_gray
                                    # these aren't good for people with defective color vision:
                  (0, 125, 52),     # vivid_green
                  (246, 118, 142),  # strong_purplish_pink
                  (0, 83, 138),     # strong_blue
                  (255, 122, 92),   # strong_yellowish_pink
                  (83, 55, 122),    # strong_violet
                  (255, 142, 0),    # vivid_orange_yellow
                  (179, 40, 81),    # strong_purplish_red
                  (244, 200, 0),    # vivid_greenish_yellow
                  (127, 24, 13),    # strong_reddish_brown
                  (147, 170, 0),    # vivid_yellowish_green
                  (89, 51, 21),     # deep_yellowish_brown
                  (241, 58, 19),    # vivid_reddish_orange
                  (35, 44, 22)]     # dark_olive_green
                  
                  
class ROSTRTPose():
        
    def __init__(self):
        self.my_log_info("Creating node")        
        self.hp_json_file = None
        self.model_weights = None
        self.width = 224
        self.height = 224
        self.i = 0
        self.image = None
        self.model_trt = None
        self.annotated_image = None
        self.counts = None
        self.peaks = None
        self.objects = None
        self.topology = None
        self.xy_circles = []
        self.cv_bridge = CvBridge()
        self.device = torch.device('cuda')

        # ROS parameters .......................................................
        # Based Dir should contain: model_file resnet/densenet, human_pose json file
        self.my_log_info("Loading ROS params")        
        self.model_folder = rospy.get_param('~model_dir', os.path.join(rospkg.RosPack().get_path('ros_trt_pose'), 'models'))
        self.model_name = rospy.get_param('~model', 'resnet18') # default to Resnet18        
        self.image_pub_topic_name = rospy.get_param('~image_pub_topic_name', 'detections_image')
        self.frame_detections_pub_topic_name = rospy.get_param('~frame_detections_pub_topic_name', 'frame_detections')        
        self.color_image_sub_topic_name = rospy.get_param('~color_image_sub_topic_name', 'image')
        self.images_info_sub_topic_name = rospy.get_param('~images_info_sub_topic_name', 'camera_info')
        
        # Load model .......................................................
        # Convert to TRT and Load Params
        self.my_log_info("Loading model params\n")
        self.load_params()
        self.my_log_info("Loading model weights\n")
        self.load_model()
        self.my_log_info("Models loaded...\n")

        # ROS publishers .......................................................
        self.my_log_info("Creating ROS comms\n")
        
        # Image with overimpressed detection
        self.image_pub = rospy.Publisher(self.image_pub_topic_name, Image, queue_size=10)
        # Frame detections
        self.frame_detections_pub = rospy.Publisher(self.frame_detections_pub_topic_name, FrameDetection, queue_size=10)

        # ROS subscribers ......................................................
        self.color_image_sub = message_filters.Subscriber(self.color_image_sub_topic_name, Image)
        self.images_info_sub = message_filters.Subscriber(self.images_info_sub_topic_name, CameraInfo)

        self.ts_message_filter = message_filters.TimeSynchronizer([self.color_image_sub, self.images_info_sub], 10)
        self.ts_message_filter.registerCallback(self.topics_callback)
                
        
        
        self.my_log_info("Node created\n")
        
    def my_log_info(self, text):
        rospy.loginfo("[" + rospy.get_name() + "]: " + text)                

    def my_log_debug(self, text):
        rospy.logdebug("[" + rospy.get_name() + "]: " + text)    

    def my_log_err(self, text):
        rospy.logerr("[" + rospy.get_name() + "]: " + text)    

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
             
    def pixel_2_vect(self, u,v):
        # Thanks to https://answers.ros.org/question/195737/how-to-get-coordinates-from-depthimage/
        # Given a model from CameraInfo message, pixel coordinates (u, v)
        # This returns 3D ray (x,y,1)

        if not hasattr(self, "camera_model"):
           return
			
        ray = np.array( self.camera_model.projectPixelTo3dRay((u,v)) )        
        return ray
        
    # Subscribe and Publish to image topic
    def topics_callback(self, color_img_msg, camera_info_msg):
        self.last_img_msg = color_img_msg

        #set camera info data
        self.imageInfoCallback(camera_info_msg)
        
        # convert color image to np array
        cv_image = self.cv_bridge.imgmsg_to_cv2(self.last_img_msg, desired_encoding='passthrough')
        self.image = np.asarray(cv_image)
        
        # Pre-process image message received from cam2image
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        self.image = cv2.resize(self.image, (self.width, self.height))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_pil = PIL.Image.fromarray(self.image)
        self.tensor_data = transforms.functional.to_tensor(self.image_pil).to(self.device)
        self.tensor_data.sub_(mean[:, None, None]).div_(std[:, None, None])
        self.data = self.tensor_data[None, ...]

        # do the detection
        cmap, paf = self.model_trt(self.data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        self.counts, self.objects, self.peaks = self.parse_objects(cmap, paf)  # , cmap_threshold=0.15, link_threshold=0.15)
        self.has_detections = int(self.counts[0]) > 0
        # draw objects
        self.draw_image()

        # create 
        self.parse_k()
        
            
        # publish messages
        if self.has_detections:
            self.frame_detections_pub.publish(self.frame_detections_message)
            self.image_pub.publish(self.image_message)

        #else:
            #self.my_log_info(f"No detection, no publication")

            
        self.has_detections = False

    def draw_image(self):
        self.annotated_image=self.image
        height = self.annotated_image.shape[0]
        width = self.annotated_image.shape[1]
        count = int(self.counts[0])
        K = self.topology.shape[0]
        for i in range(count):
            # Each person will have a different color ...
            color = PERSON_COLORS[i]
            obj = self.objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = self.peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(self.annotated_image, (x, y), 3, color, 2)

            for k in range(K):
                c_a = self.topology[k][2]
                c_b = self.topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = self.peaks[0][c_a][obj[c_a]]
                    peak1 = self.peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(self.annotated_image, (x0, y0), (x1, y1), color, 2)            
        
        # Put annotated image into a ros msg        
        try:
            self.image_message = self.cv_bridge.cv2_to_imgmsg(self.annotated_image, encoding='bgr8')
            self.image_message.header = self.last_img_msg.header
            
        except CvBridgeError as e:
            self.my_log_err(f"Error casting cv image to ros msg: {e} | skypping..")
            self.has_detections = False    
            
    def parse_k(self):
            self.frame_detections_message = FrameDetection()
            self.frame_detections_message.header = self.last_img_msg.header
        
            image_idx = 0
            count = int(self.counts[image_idx])
            for i in range(count):
                person_i = KeyPointsDetection()
                
                for k in range(18):
                    point_k = Point()
                    point_k.x = point_k.y = point_k.z = 0
                    
                    idx = self.objects[image_idx, i, k]
                    if _idx >= 0:
                        location = self.peaks[image_idx, k, _idx, :]
                        scaled_pixel_x = round(float(_location[1]) * self.width)
                        scaled_pixel_y = round(float(_location[0]) * self.height)

                        image_height = 480
                        image_width = 640

                        pixel_x = round(float(_location[1]) * image_width)
                        pixel_y = round(float(_location[0]) * image_height)
                        
                        vect3d = self.pixel_2_vect(pixel_x,pixel_y)
                        point_k.x = vect3d[0]
                        point_k.y = vect3d[1] 
                        point_k.z = vect3d[2]
                                                     
                    # append point (detected or not) to keypoints
                    person_i.keypoints[k] = point_k
                # append person to frame detection
                self.frame_detections_message.persons.append(person_i) 

    def load_params(self):
        if self.model_name == 'resnet18':
            MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
        if self.model_name == 'densenet121':
            MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'

        hp_json_file = os.path.join(self.model_folder, 'human_pose.json')
        with open(hp_json_file,'r') as f:
            human_pose = json.load(f)

        # set parameters
        self.model_weights = os.path.join(self.model_folder, MODEL_WEIGHTS)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.num_parts = len(human_pose['keypoints']) # Name of the body part
        self.num_links = len(human_pose['skeleton']) # Need to know
        self.parse_objects = ParseObjects(self.topology)

    def load_model(self):
        #self.get_logger().info("Model Weights are loading \n")
        if self.model_name == 'resnet18':
            model = trt_pose.models.resnet18_baseline_att(self.num_parts, 2*self.num_links).cuda().eval()
            model.load_state_dict(torch.load(self.model_weights))
            MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
            OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            self.height, self.width = 224,224
        if self.model_name == 'densenet121':
            model = trt_pose.models.densenet121_baseline_att(self.num_parts, 2*self.num_links).cuda().eval()
            model.load_state_dict(torch.load(self.model_weights))
            MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
            OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
            self.height, self.width = 256,256

        model_file_path = os.path.join(self.model_folder, OPTIMIZED_MODEL)
        if not os.path.isfile(model_file_path):
            data = torch.zeros((1,3, elf.height, self.width)).cuda()
            self.model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(self.model_trt.state_dict(), model_file_path)
        
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(model_file_path))


def main(args):
  rospy.init_node('trt_pose_node', anonymous=True)
  node = ROSTRTPose()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down trt pose node")

if __name__ == '__main__':
    main(sys.argv)
