# Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

[[Project page]](https://arjung128.github.io/opening-cabinets-and-drawers/)
[[Paper]](https://arxiv.org/abs/2402.17767)

<img width="100%" src="assets/teaser.gif">

[Arjun Gupta](https://arjung128.github.io/)\*,
[Michelle Zhang](https://www.linkedin.com/in/michelle-zhang-065037193/)\*,
[Rishik Sathua](https://www.linkedin.com/in/rishik-sathua/),
[Saurabh Gupta](http://saurabhg.web.illinois.edu/) <br />
\*denotes equal contribution.

## Installation
This codebase is developed and tested on Hello Robot's Stretch RE2 with ROS 1 Noetic.

Install dependencies:
```console
$ pip install -r requirements.txt
```

We recommend not using conda, as we found that conda does not play well with ROS.

Next, install detectron2 following the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Setup
1. Make sure that the robot URDF is calibrated. If not, follow the instructions [here](https://docs.hello-robot.com/0.2/stretch-ros/stretch_calibration/).
2. Download the Mask RCNN model weights from [here](https://drive.google.com/drive/folders/16d5AVrvcZF-rY-GZv1x7F9QKT02p7OEs?usp=share_link) and save these weights in the `weights/` directory.
3. Make sure the `catkin_ws` directory is in the parent directory, i.e. when you are inside the repo `stretch-open`, it can be accessed through `cd ./../catkin_ws`. If not, modify the paths in `modules/motion_planning.py` and `modules/grasp_correction.py`.
4. In `pipeline.py`, replace the `camera_id` in `maskrcnn = inference.Maskrcnn_Module('weights/m1.pth', 'weights/m2.pth', camera_id = '153122077062')` with the serial number of your Stretch's head camera. If you do not know your camera's serial number, use the following code snippet:
    ```console
    import pyrealsense2 as rs
    context = rs.context()
    devices = context.query_devices()
    # then, devices.back() should print out the serial number (S/N) of the D435i
    ``` 

## Running the Pipeline

Make sure to run `stretch_robot_home.py` and `stretch_robot_stow.py` first.

In our experiments, we positioned the Stretch RE2 ~1.5m away from a target cabinet or drawer. Once positioned, you may need to change the head camera viewpoint to have a cabinet or drawer in view.

To run the full pipeline use the following python script:
```console
$ python pipeline.py
```
This process starts by launching the perception module, which initiates a video stream and displays the Mask RCNN predictions on the current camera view. Each detected object will have a number displayed next to it. To select an object, press CTRL+C, type the corresponding number, and hit Enter. The robot will then proceed with the pipeline. For safety, we have added breakpoints after each waypoint during the motion plan execution (this feature can be turned off). Please see below for more details on each module.

## Perception
The perception module feeds the current RGB image through Mask RCNN model and displays the predictions. A user has to select which detected cabinet or drawer to open. This can be done by hitting CTRL+C and entering the number written next to the handle of the target object.

To run the Mask RCNN perception module use the following code:
```
maskrcnn = inference.Maskrcnn_Module('m1.pth', 'm2.pth', camera_id = '153122077062')
maskrcnn_output = maskrcnn.run_inference()
```

## 3D Transformations
<!-- transforming to base coordinates (include things to keep an eye out for such as coordinates) -->
The transforms module will take in the 3d camera coordinate maskrcnn module outputs and use ros to transform them into base coordinates.

To run the transforms module use the following code:
```
transform = transforms.Transforms_Module(maskrcnn_output)
transform_output = transform.transform_data()
```

## Navigation
The navigation module uses the stretch api to navigate to the target location output by the transform module. Currently, object avoidance is not implemented.

To run the navigation use the following code:
```
nav = navigation.Navigation_Module(transform_output)
nav.run_navigation() 
```

## Motion Planning and Deployment
Once Stretch has navigated to target location, generate a motion plan using:
```
planner = motion_planning.Motion_Plan_Module(maskrcnn_output, transform_output)
motion_plan = planner.run_motion_planning()
```
After a motion plan is generated, deploy the pregrasp pose using the following code:
```
deploy = execution.Execute(motion_plan, maskrcnn_output)
robot = deploy.start_robot()
correction_needed = deploy.execute_trajectory(skip_first_waypoint=False)
```
If `execute_trajectory()` returns 1 then grasp correction is needed. To run grasp correction use the following:
```
grasp_corrector = grasp_correction.Grasp_Correction(robot, motion_plan, maskrcnn_output)
contact_offset = grasp_corrector.run_correction()
```
To update the motion plan to take into account the movement made during grasp correction run:
```
updated_motion_plan = planner.run_motion_planning(contact_offset)
deploy.update_motion_plan(updated_motion_plan)
```
Continue executing the trajectory using the updated plan:
```
deploy.execute_trajectory(skip_first_waypoint=True)
```
Release object and stop the robot (run this regardless of if grasp correction was needed):
```
deploy.stop_robot()
```

## Trouble Shooting
### Calibration
We calibrated the Stretch's URDF using 4 additional Aruco markers placed on the floor in a 2x2 square 1-2m away from the robot's base. We have included the files we changed in order to customize the calibration process in the **calibration** folder. See below for further details:

#### `stretch_uncalibrated.udf`
Location: `catkin_ws/src/stretch_ros/stretch_description/urdf/`

Change: Add joints and links for custom aruco markers to end of file. In example provided, customization begins at line 907. Do not run `rosrun stretch_calibration update_uncalibrated_urdf.sh` after customization. 

#### `stretch_marker_dict.yaml`
Location: `catkin_ws/src/stretch_ros/stretch_core/config/`

Change: Specify information regarding aruco marker ids. `link_aruco_floor_*` are the custom floor markers we used. 

#### `collect_head_calibration_data.py`
Location: `catkin_ws/src/stretch_ros/stretch_calibration/nodes/`

Change: Add support for collecting data for custom aruco markers. Ctrl + F 'custom' to see locations of code customization. 

#### `process_head_calibration_data.py`
Location: `catkin_ws/src/stretch_ros/stretch_calibration/nodes/`

Change: Add support for processing and fitting to data for custom aruco markers. Ctrl + F 'custom' to see locations of code customization.

