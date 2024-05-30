import stretch_body.robot
import numpy as np
import time
import sys 

class Execute():
    def __init__(self, motion_plan, maskrcnn_data, use_pdb = False):
        self.qs = motion_plan
        self.maskrcnn_data = maskrcnn_data
        self.robot = None
        self.use_pdb = use_pdb

    def start_robot(self):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()

    def stop_robot(self):
        # release gripper
        self.robot.end_of_arm.move_to('stretch_gripper',50)
        time.sleep(3)
        # stop robot
        self.robot.stop()

    def update_motion_plan(self, motion_plan):
        self.qs = motion_plan

    def get_absolute_rotation(self, qs):
        '''
        qs: (x, 20)
        '''
        thetas = []
        for q in qs:
            # two candidate thetas
            theta_1 = np.arccos(q[2])
            theta_2 = np.arcsin(q[3])

            # pick one based on which best explains data
            if np.abs(np.cos(theta_1) - q[2]) < 1e-15 and np.abs(np.sin(theta_1) - q[3]) < 1e-15:
                theta = theta_1
            elif np.abs(np.cos(theta_2) - q[2]) < 1e-15 and np.abs(np.sin(theta_2) - q[3]) < 1e-15:
                theta = theta_2
            else:
                assert False

            theta += np.pi/2 # coordinate transform between meshcat and real world
            thetas.append(theta)
        return thetas

    def absolute_to_relative_rotation(self, qs_base_thetas):
        '''
        keep the first absolute rotation, as then this way, we
        can orient the base to be parallel to the door, before
        the first waypoint adjusts the rotation.
        '''
        qs_base_thetas_relative = [qs_base_thetas[0]]
        for i in range(len(qs_base_thetas)-1):
            qs_base_thetas_relative.append(qs_base_thetas[i+1] - qs_base_thetas[i])
        return qs_base_thetas_relative

    def execute_trajectory(self, skip_first_waypoint):
        if self.maskrcnn_data is None or self.qs is None:
            print("Missing Data Input")
            return 0

        self.start_robot()

        class_name = self.maskrcnn_data['classification']
        handle_orientation = self.maskrcnn_data['handle_orientation']

        qs_base_thetas = self.get_absolute_rotation(self.qs)
        qs_base_thetas_relative = self.absolute_to_relative_rotation(qs_base_thetas)

        # duplicate first element
        self.qs = np.insert(self.qs, 0, self.qs[0], axis=0)
        qs_base_thetas_relative = np.insert(qs_base_thetas_relative, 1, 0, axis=0)

        if not skip_first_waypoint:
            self.robot.stow()


        if not skip_first_waypoint:
            # compute translational offset to account for observed translation during rotation
            trans_offset = 0.75 - (np.cos(qs_base_thetas[0]) * 0.75)
            self.robot.base.translate_by(trans_offset / 100)
            self.robot.push_command()
            time.sleep(5.0)

        start_i = 1 if skip_first_waypoint else 0
        for i in range(start_i, len(self.qs)):
            if self.use_pdb:
                import pdb; pdb.set_trace()
            print("Waypoint: ", i)
            # close gripper first if skip_first_waypoint
            if skip_first_waypoint and i == 1:
                # make sure gripper is parallel to ground 
                self.robot.end_of_arm.move_to('wrist_pitch', 0) # added for wrist camera
                self.robot.end_of_arm.move_to('stretch_gripper',-70)
                self.robot.push_command()
                time.sleep(3)
            
            # extract params
            lift_height = self.qs[i][-12] + 0.04 # hard-coding additional lift height
            arm_length = self.qs[i][-8] + self.qs[i][-9] + self.qs[i][-10] + self.qs[i][-11]
            wrist_yaw = self.qs[i][-7]
            rotate_by = qs_base_thetas_relative[i]

            # send commands
            # base.rotate_by(x_r, v_r, a_r, stiffness, deprecated, contact_thresh)
            self.robot.base.rotate_by(rotate_by, v_r = 0.2)
            self.robot.lift.move_to(lift_height, v_m = 0.05) 
            if i == 0:
                # separate first rotation from arm movemets to help prevent collisions
                self.robot.push_command()
                time.sleep(10)
                # query the current number of guarded contacts before extending the arm
                start_guard_event = self.robot.arm.motor.status['guarded_event']
            
            # end_of_arm.move_to(joint, c_r, v_r, a_r)
            self.robot.end_of_arm.move_to('wrist_yaw', wrist_yaw)
            # make sure gripper is parallel to ground 
            self.robot.end_of_arm.move_to('wrist_pitch', 0) # added for wrist camera
            if handle_orientation == 'horizontal':
                self.robot.end_of_arm.move_to('wrist_roll', 1.57)
            else:
                self.robot.end_of_arm.move_to('wrist_roll', 0)

            # prismatic_joint.move_to(x_m, v_m, a_m, stiffness, deprecated, deprecated, req_calibration, contact_thresh_pos, contact_thresh_neg)
            self.robot.arm.move_to(arm_length, v_m = 0.05, contact_thresh_pos = 40)

            if i == 0:
                if handle_orientation == 'vertical':
                    self.robot.end_of_arm.move_to('stretch_gripper',0)
                else:
                    self.robot.end_of_arm.move_to('stretch_gripper',50)
                self.robot.push_command()
                time.sleep(15)
                curr_guard_event = self.robot.arm.motor.status['guarded_event']
                print('guarded_event diff:', curr_guard_event - start_guard_event)

                # vertical primitive
                if handle_orientation == 'vertical':
                    # retract arm
                    self.robot.arm.move_by(-0.03, contact_thresh_pos=40)
                    self.robot.push_command()
                    time.sleep(3)
                    # tilt gripper down and open
                    self.robot.end_of_arm.move_to('wrist_pitch', -0.75)
                    time.sleep(3)
                    self.robot.end_of_arm.move_to('stretch_gripper', 50)
                    time.sleep(3)
                    # extend arm
                    extension_length = 0.03
                    if curr_guard_event > start_guard_event:
                        extension_length = 0.05
                    self.robot.arm.move_by(extension_length, contact_thresh_pos=40)
                    self.robot.push_command()
                    time.sleep(3)
                    # tilt gripper up
                    self.robot.end_of_arm.move_to('wrist_pitch', 0)
                    time.sleep(3)

                if curr_guard_event == start_guard_event or handle_orientation == 'vertical':
                    print("Correction needed")
                    self.stop_robot()
                    return 1
            else:
                self.robot.end_of_arm.move_to('stretch_gripper',-70)
                self.robot.push_command()
                time.sleep(5)

        self.stop_robot()
        return 0
