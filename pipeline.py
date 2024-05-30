#!/usr/bin/env python
from maskrcnn_module import inference
from modules import transforms, navigation, motion_planning, execution, grasp_correction
import time

# perception and lift to 3d camera coordinates
maskrcnn = inference.Maskrcnn_Module('m1.pth', 'm2.pth', camera_id = '153122077062')
maskrcnn_output = maskrcnn.run_inference()

print("Maskrcnn Output: ", maskrcnn_output)

# transform to base coordinates
transform = transforms.Transforms_Module(maskrcnn_output)
transform_output = transform.transform_data()

print("Transform Output: ", transform_output)

# navigation
nav = navigation.Navigation_Module(transform_output)
nav.run_navigation() 

# initial motion planning
planner = motion_planning.Motion_Plan_Module(maskrcnn_output, transform_output)
motion_plan = planner.run_motion_planning()
# pre-grasp pose
deploy = execution.Execute(motion_plan, maskrcnn_output, use_pdb = True)
correction_needed = deploy.execute_trajectory(skip_first_waypoint=False)

# correction based on proprioceptive feedback
if correction_needed:
    grasp_corrector = grasp_correction.Grasp_Correction(motion_plan, maskrcnn_output)
    contact_offset = grasp_corrector.run_correction()
    # update motion plan
    updated_motion_plan = planner.run_motion_planning(contact_offset)
    deploy.update_motion_plan(updated_motion_plan)
    # continue execution with corrected pre-grasp pose
    deploy.execute_trajectory(skip_first_waypoint=True)

