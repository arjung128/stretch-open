#!/usr/bin/env python3
import stretch_body.robot
import argparse
import numpy as np
import time

import subprocess
import time

class Navigation_Module():
    def __init__(self, data_input=None, v_r = 0.1, v_m = 0.08):
        self.data_input = data_input
        self.v_r = v_r
        self.v_m = v_m

    def run_navigation(self):

        if self.data_input is None:
            print("Missing Data Input")
            return 

        # load the outputs from transform_camera_point.py
        stretch_x = self.data_input['goal_point'][0]
        stretch_y = self.data_input['goal_point'][1]
        surf_norm = self.data_input['surf_norm']

        robot = stretch_body.robot.Robot()
        robot.startup()

        # the robot navigates to goal in 3 steps: rotation, translation, rotation
        # object avoidance has not yet been implemented
        heading_dir = np.arctan2(stretch_y, stretch_x) 
        forward_dist = np.sqrt(stretch_x**2 + stretch_y**2)

        print("heading_dir: ", heading_dir)
        print("forward_dist: ", forward_dist)

        # first rotation
        robot.base.rotate_by(heading_dir, v_r = self.v_r)
        robot.push_command()
        time.sleep(np.abs(heading_dir/self.v_r) + 2)

        # translation
        robot.base.translate_by(forward_dist, v_m = self.v_m)
        robot.push_command()
        time.sleep(np.abs(forward_dist/self.v_m) + 2)

        # second rotation
        second_rotation = -heading_dir - (surf_norm * np.pi / 180)
        robot.base.rotate_by(second_rotation, v_r = self.v_r)
        robot.push_command()
        time.sleep(np.abs(second_rotation/self.v_r) + 2)

        robot.stop()


