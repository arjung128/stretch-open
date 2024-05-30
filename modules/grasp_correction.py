#!/usr/bin/env python3
import stretch_body.robot
import argparse
import numpy as np
import time
import pinocchio as pin

class Grasp_Correction():
    def __init__(self,motion_plan=None, maskrcnn_data=None):
        self.qs = motion_plan
        self.maskrcnn_data = maskrcnn_data
        self.robot = None

    def init_robot(self):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        time.sleep(3)
        # poll status of sensors
        self.start_wrist_yaw_effort = self.robot.end_of_arm.get_joint('wrist_yaw').status['effort']
        self.start_guard_event = self.robot.arm.motor.status['guarded_event']

        print("yaw: ", self.start_wrist_yaw_effort)
        print("guard_event: ", self.start_guard_event)

    # rotate robot by 1 degree increments until contact is made
    def rotate_correction(self):
        while True:
            self.robot.arm.move_by(-0.05)
            self.robot.push_command()
            time.sleep(3)
            self.robot.base.rotate_by(1*np.pi/180)
            self.robot.push_command()
            time.sleep(3)
            self.robot.arm.move_by(0.05, contact_thresh_pos = 40)
            self.robot.push_command()
            time.sleep(3)

            curr_guarded_event = self.robot.arm.motor.status['guarded_event']
            print('guarded event:', curr_guarded_event - self.start_guard_event)

            curr_wrist_yaw_effort = self.robot.end_of_arm.get_joint('wrist_yaw').status['effort']

            yaw_diff = self.start_wrist_yaw_effort - curr_wrist_yaw_effort

            print("yaw: ", yaw_diff)

            if curr_guarded_event > self.start_guard_event or  abs(yaw_diff)>0.1: 
                break

    # extend arm by 1cm increments until contact is made
    def arm_correction(self):
        while True:
            self.robot.arm.move_by(0.01, contact_thresh_pos=40)
            self.robot.push_command()
            time.sleep(1)

            curr_wrist_yaw_effort = self.robot.end_of_arm.get_joint('wrist_yaw').status['effort']
            yaw_diff = self.start_wrist_yaw_effort - curr_wrist_yaw_effort
            print("yaw: ", yaw_diff)

            curr_guarded_event = self.robot.arm.motor.status['guarded_event']
            print('guarded event:', curr_guarded_event - self.start_guard_event)
            if curr_guarded_event > self.start_guard_event or abs(yaw_diff) > 0.1:
                break

    def compute_correction_offset(self):
        model_base, geom_model_base, visual_model_base = pin.buildModelsFromUrdf('./../catkin_ws/src/stretch_ros/stretch_description/urdf/exported_urdf/stretch.urdf', './../catkin_ws/src/stretch_ros/stretch_description/urdf/exported_urdf', pin.JointModelPlanar())
        data_base, geom_data_base, visual_data_base = pin.createDatas(model_base, geom_model_base, visual_model_base)

        q = self.qs[0]

        # forward kinematics
        pin.forwardKinematics(model_base, data_base, q)
        pin.framesForwardKinematics(model_base, data_base, q)
        frame_id = model_base.getFrameId('joint_gripper_fingertip_right')
        pos = data_base.oMf[frame_id]
        print(pos)

        # construct qpose
        status = self.robot.get_status()
        # --> rotation is always 2*pi when robot is started
        # consider translation due to rotational offset, and base rotation after contact debug (can count in contact debug) -- maybe later? These may be minimal.

        # q[8] = status['lift']['pos'] # lift_height # same as before
        arm_length = status['arm']['pos']
        q[9], q[10], q[11], q[12] = 0, 0, 0, 0 # reset arm
        if arm_length < 0.13:
            q[9] = arm_length
        elif arm_length >= 0.13 and arm_length < 0.26:
            q[9] = 0.13
            q[10] = arm_length - 0.13
        elif arm_length >= 0.26 and arm_length < 0.39:
            q[9] = 0.13
            q[10] = 0.13
            q[11] = arm_length - 0.26
        else:
            q[9] = 0.13
            q[10] = 0.13
            q[11] = 0.13
            q[12] = arm_length - 0.39
        q[13] = status['end_of_arm']['wrist_yaw']['pos'] # wrist_yaw
        q[14] = status['end_of_arm']['wrist_pitch']['pos'] # wrist_pitch

        # set yaw pitch roll to original

        # can visualize in meshcat as sanity check
        # may need to subtract a bit to account for gripper closing

        # alternatively, modify q[0] based on changes in contact_debug
        # but this may be more accurate, since with contact, the arm may be squished a bit

        # forward kinematics
        pin.forwardKinematics(model_base, data_base, q)
        pin.framesForwardKinematics(model_base, data_base, q)
        frame_id = model_base.getFrameId('joint_gripper_fingertip_right')
        pos = data_base.oMf[frame_id]
        print(pos)
        return pos.translation[:2]


    def run_correction(self):
        if self.qs is None or self.maskrcnn_data is None:
            print("Missing Data Input")
            return None

        self.init_robot()

        class_name = self.maskrcnn_data['classification']

        # use propriocetive feedback to adjust robot position until contact is made with cabinet
        if class_name == "Left-hinged":
            self.rotate_correction()
        else:
            self.arm_correction()

        self.compute_correction_offset()

        self.robot.stop()


