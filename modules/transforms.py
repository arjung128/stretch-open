#!/usr/bin/env python3

import tf
import rospy
from geometry_msgs.msg import PointStamped
import numpy as np
import time
import os 
import subprocess
import signal

class Transforms_Module():
    def __init__(self, data_input=None):
        self.data_input = data_input

    def launch_ros(self):
        self.stretch_ros_core = subprocess.Popen(["roslaunch", "stretch_core", "stretch_driver.launch"], shell=False, preexec_fn=os.setsid)
        # sleep to give time for rosmaster to start
        time.sleep(7)
    
    def stop_ros(self):
        # os.killpg(os.getpgid(self.stretch_ros_core.pid), signal.SIGTERM)
        subprocess.run(["rosnode", "kill", "-a"])
        subprocess.run(["killall", "-9", "rosmaster"])
        while self.stretch_ros_core.poll() is None:
            time.sleep(1)

    def get_target_nav_location(self, classification):
        '''
        given classification, returns target nav location (relative to handle) according to no-learning baseline.
        output: (base_x, base_y)
        - base_x: [-1, 1] with 1 being to left of handle
        - base_y: [0,  1] with 1 being away from  handle
        '''
        if classification == 'Left-hinged':
            return (0.65, 0.45)
        elif classification == 'Right-hinged':
            return (-0.125, 0.625)
        elif classification == 'Pulls out':
            return (0.025, 0.725)
        else:
            raise NotImplemented

    def transform_data(self):
        if self.data_input is None:
            print("Missing Data Input")
            return None

        self.launch_ros()
        x_3d, y_3d, depth = self.data_input['handle_3d']
        x_3d_plus_surfnorm, y_3d_plus_surfnorm, depth_plus_surfnorm = self.data_input['handle_3d_plus_surfnorm']
        # image is mirrored so points will be ordered from left topmost point and continue counter clockwise
        polygon = self.data_input['polygon']
        class_name = self.data_input['classification']

        # initialize new ros node
        rospy.init_node('transform_debug', anonymous=True)
        # create transform listener
        tf_listener = tf.TransformListener()  
        tf_listener.waitForTransform("/camera_depth_frame", "/base_link", rospy.Time(0), rospy.Duration(10.0))

        #########################
        # HANDLE
        #########################
        # point in camera coordinates
        camera_point = PointStamped()
        camera_point.header.frame_id = "camera_depth_frame"
        camera_point.header.stamp = rospy.Time(0)

        # depth frame
        camera_point.point.x = float(depth)
        camera_point.point.y = -float(y_3d)
        camera_point.point.z = -float(x_3d)
        # apply transform
        base_point = tf_listener.transformPoint("base_link", camera_point)
        print('base handle point:', base_point)

        #########################
        # RADIUS
        #########################
        base_poly_points = []
        for point in polygon:
            # point in camera coordinates
            camera_point = PointStamped()
            camera_point.header.frame_id = "camera_depth_frame"
            camera_point.header.stamp = rospy.Time(0)

            # depth frame
            camera_point.point.x = float(point[2])
            camera_point.point.y = -float(point[1])
            camera_point.point.z = -float(point[0])
            # apply transform
            base_poly_points.append(tf_listener.transformPoint("base_link", camera_point))
        
        # ignore height (z) and average the left side coordinates (x,y) and the right side coordinates (x,y)
        right_side_poly = np.array([(base_poly_points[0].point.x + base_poly_points[1].point.x)/2, (base_poly_points[0].point.y + base_poly_points[1].point.y)/2])
        left_side_poly = np.array([(base_poly_points[2].point.x + base_poly_points[3].point.x)/2, (base_poly_points[2].point.y + base_poly_points[3].point.y)/2])
        
        handle_array = np.array([base_point.point.x, base_point.point.y])
        left_dist = np.linalg.norm(handle_array - left_side_poly)
        right_dist = np.linalg.norm(handle_array - right_side_poly)

        print("left dist: ", left_dist)
        print("right_dist: ", right_dist)

        if class_name == "Right-hinged":
            radius = right_dist
        elif class_name == "Left-hinged":
            radius = left_dist
        else:
            radius = 0
        #########################
        # SURFACE NORMAL
        #########################
        # point in camera coordinates
        camera_point = PointStamped()
        camera_point.header.frame_id = "camera_depth_frame"
        camera_point.header.stamp = rospy.Time(0)

        # depth frame
        camera_point.point.x = float(depth_plus_surfnorm)
        camera_point.point.y = -float(y_3d_plus_surfnorm)
        camera_point.point.z = -float(x_3d_plus_surfnorm)
        # apply transform
        base_point_plus_surfnorm = tf_listener.transformPoint("base_link", camera_point)
        print('base handle point plus surfnorm:', base_point_plus_surfnorm)

        # surface normal
        lateral_diff = base_point_plus_surfnorm.point.y - base_point.point.y
        depth_diff = base_point_plus_surfnorm.point.x - base_point.point.x
        surfnorm_angle = np.arctan2(lateral_diff, depth_diff)
        
        if (surfnorm_angle * 180 / np.pi) > 0:
            converted_surfnorm_angle = 180 - (surfnorm_angle * 180 / np.pi)
        else:
            converted_surfnorm_angle = -1 * (180 + (surfnorm_angle * 180 / np.pi))

        print('surface normal angle: ', converted_surfnorm_angle)
        #########################
        # NAVIGATION GOAL
        #########################
        handle_frame_goal = self.get_target_nav_location(class_name)

        # forward direction in stretch coordinates (x)
        rad_surf_norm = converted_surfnorm_angle * np.pi /180
        goal_x = base_point.point.x - (handle_frame_goal[1]*np.cos(rad_surf_norm) - handle_frame_goal[0]*np.sin(rad_surf_norm))
        # lateral direction in stretch coordinates (y)
        goal_y = base_point.point.y + (handle_frame_goal[0]*np.cos(rad_surf_norm) + handle_frame_goal[1]*np.sin(rad_surf_norm))

        # print(f"handle x: {base_point.point.x}, handle y: {base_point.point.y}")
        output_dict = {'goal_point': np.array([goal_x, goal_y, base_point.point.z]),
                        'surf_norm': converted_surfnorm_angle,
                        'class': class_name,
                        'radius': radius
                        }
        
        self.stop_ros()
            
        return output_dict