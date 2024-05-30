import numpy as np
from numpy.linalg import norm, solve
from open3d import *
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import math

class Motion_Plan_Module():
    def __init__(self, maskrcnn_data=None, transform_data=None):
        self.maskrcnn_data = maskrcnn_data
        self.transform_data = transform_data

    def get_target_nav_location(self, classification, left_vs_right=None):
        '''
        given classification, returns target nav location (relative to handle) according to no-learning baseline.
        output: (base_x, base_y)
        - base_x: [-1, 1] with 1 being to left of handle
        - base_y: [0,  1] with 1 being away from  handle
        '''
        if classification == 'drawer':
            return (0.025, 0.725)
        elif classification == 'door':
            if left_vs_right == 'left':
                return (0.65, 0.45)
            else:
                return (-0.125, 0.625)
        else:
            raise NotImplemented
            

    def inverse_kinematics_stretch_fullBody(self, pos, rot, base_x=None, base_y=None, ee_angle=0.0, num_tries=10, verbose=True, IT_MAX=5000, prev_qpos=None, cand_idx=-1, perturbation=0.01, rand_idx=1000, baseline=False, variable_position=False, variable_height=False, free_base_orientation=False, base_orientation_idx=0, variable_rotation=False, eps=1e-4):
        '''
        perform IK for the stretch.
        '''
        # set oMdes
        oMdes = pin.SE3(rot, pos)

        low = self.model_base.lowerPositionLimit + 1e-5
        low[-7] = -1.26 + 1e-5
        high = self.model_base.upperPositionLimit - 1e-5

        for k in range(num_tries):
            if prev_qpos is None:
                q = pin.neutral(self.model_base) # pin.randomConfiguration(model)
                q[4] = pos[2] - 0.10 # set lift
            else:
                q = prev_qpos

            if q is None: # no collision-free qpos found after perturbation
                return None
            DT     = 1e-1
            damp   = 1e-12

            i=0
            while True:
                pin.forwardKinematics(self.model_base, self.data_base, q)
                pin.framesForwardKinematics(self.model_base, self.data_base, q)

                frame_id = self.model_base.getFrameId('joint_gripper_fingertip_right')
                dMi = oMdes.actInv(self.data_base.oMf[frame_id])

                err = pin.log(dMi).vector
                if norm(err) < eps:
                    success = True
                    break
                if i >= IT_MAX:
                    success = False
                    break

                J = pin.computeFrameJacobian(self.model_base,self.data_base,q,frame_id)
                v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model_base,q,v*DT)

                # enforce joint limits
                q = np.clip(q, low, high)

                # fix other unused joints
                q[4:4+4] = np.array([1., 0., 0., 0.])

                # constrain rotation
                if base_x is not None and base_y is not None:
                    q[0] = base_x
                    q[1] = base_y
                    q[3] = np.clip(q[3], -1, 1)
                    q[2] = np.cos(np.arcsin(q[3]))
                
                # set finger joints
                q[-4] = 0.
                q[-3] = 0.

                i += 1

            return q, success    
                
                
    def get_motion_plan(self, base_x, base_y, classification, left_vs_right, radius, handle_height, correction_offset = None):
        '''
        perform SeqIK for the Stretch.
        '''
        # get trajectory
        if classification == 'door':
            if left_vs_right == 'left':
                angles = np.linspace((3/2)*np.pi, 2*np.pi, 10)[1:]
                x_offsets = radius * np.sin(angles) + radius
                y_offsets = radius * np.cos(angles)
                angles_rot = np.linspace(0, (1/2)*np.pi, 10)[1:]
            else:
                angles = np.linspace((3/2)*np.pi, 2*np.pi, 10)[1:]
                x_offsets = -(radius * np.sin(angles) + radius)
                y_offsets = radius * np.cos(angles)
                angles_rot = np.linspace(0, -(1/2)*np.pi, 10)[1:]
        elif classification == 'drawer':
            y_offsets = np.linspace(0., 0.35, 11)[1:]
            x_offsets = np.zeros_like(y_offsets)
        else:
            assert False

        eps = 1e-4
        init_qpos = pin.neutral(self.model_base)
        IT_MAX = 50000

        qs = []
        # initial waypoint
        # (+ is to the left, + is to the back, + is up)
        pos = np.array([0.0, 0.0, handle_height])
        if correction_offset is not None:
            pos[:2] += correction_offset
        pos_og = pos.copy()
        euler_rotation = R.from_euler('xyz', [np.pi/2, 0., 1.1]).as_matrix()

        success_idxs = []
        start_q, success = self.inverse_kinematics_stretch_fullBody(pos, euler_rotation, base_x=base_x, base_y=base_y, prev_qpos=init_qpos, num_tries=1, eps=eps, IT_MAX=IT_MAX)
        if success == True:
            qs.append(start_q)
            success_idxs.append(1.)
        else:
            success_idxs = np.zeros(10)
            return qs, success_idxs

        # subsequent waypoints are just y-offsets
        for i in range(9):
            pos[0] = pos_og[0] + x_offsets[i]
            pos[1] = pos_og[1] + y_offsets[i]
            if classification == 'door':
                euler_rotation = R.from_euler('xyz', [np.pi/2, 0., 1.1-angles_rot[i]]).as_matrix()
            elif classification == 'drawer':
                euler_rotation = R.from_euler('xyz', [np.pi/2, 0., 1.1]).as_matrix()
            else:
                assert False
            q, success = self.inverse_kinematics_stretch_fullBody(pos, euler_rotation, base_x=base_x, base_y=base_y, prev_qpos=qs[-1], num_tries=1, eps=eps, IT_MAX=IT_MAX)
            if success:
                qs.append(q)
                success_idxs.append(1.)
            else:
                success_idxs.append(0.)

        return qs, success_idxs

    def run_motion_planning(self, correction_offset=None):
        if self.maskrcnn_data is None or self.transform_data is None:
            print("Missing Data Input")
            return None

        print("Generating Motion Plan")
        class_name = self.maskrcnn_data['classification']
        # classes: ["Left-hinged", "Right-hinged", "Bottom-hinged", "Pulls out", "Top-hinged"]
        if class_name == 'Left-hinged':
            classification = 'door'
            left_vs_right = 'left'
        elif class_name == 'Right-hinged':
            classification = 'door'
            left_vs_right = 'right'
        elif class_name == 'Pulls out':
            classification = 'drawer'
            left_vs_right = None
        else:
            classification = class_name
            left_vs_right = None
        radius = self.transform_data['radius'] # meters
        handle_height = self.transform_data['goal_point'][2] # meters
        print("classification: ", classification, " left_vs_right: ", left_vs_right)
        # get target nav location
        base_x, base_y = self.get_target_nav_location(classification, left_vs_right)
        # base_x: lateral distance of base to handle ([-ve, +ve], with +ve being left)
        # base_y: forward distance of base to handle ([0, +ve] with 0 being handle)
        
        # load stretch model
        self.model_base, geom_model_base, visual_model_base = pin.buildModelsFromUrdf('./../catkin_ws/src/stretch_ros/stretch_description/urdf/exported_urdf/stretch.urdf', './../catkin_ws/src/stretch_ros/stretch_description/urdf/exported_urdf', pin.JointModelPlanar())
        self.data_base, geom_data_base, visual_data_base = pin.createDatas(self.model_base, geom_model_base, visual_model_base)
        
        # given parameters, use SeqIK to get motion plan
        motion_plan, _ = self.get_motion_plan(base_x, base_y, classification, left_vs_right, radius, handle_height, correction_offset)
        motion_plan = np.array(motion_plan)
        
        print("Waypoints: ", motion_plan.shape)

        return motion_plan
        
