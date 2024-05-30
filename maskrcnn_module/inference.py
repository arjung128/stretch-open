#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2

import time, datetime
import os
import pathlib
import argparse
import stretch_body.hello_utils as hu
# hu.print_stretch_re_use()

# Maskrcnn imports
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import detectron2
import os
import pycocotools
import cv2
import sys
from collections import OrderedDict

from detectron2 import config
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, DatasetMapper, build_detection_train_loader
from detectron2.evaluation import inference_on_dataset

from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import itertools
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import copy

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.structures.boxes import Boxes
from detectron2.structures.masks import BitMasks
from detectron2.structures.instances import Instances
from tqdm import tqdm

import struct
import operator
from gltflib import (
    GLTF, GLTFModel, Asset, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor, AccessorType,
    BufferTarget, ComponentType, GLBResource, FileResource)

import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision
import torchvision.transforms as T
import random


import importlib
import maskrcnn_module.custom_roi_heads
importlib.reload(maskrcnn_module.custom_roi_heads)
from maskrcnn_module.custom_roi_heads import ROI_HEADS_REGISTRY, CustomStandardROIHeads
ROI_HEADS_REGISTRY.register(CustomStandardROIHeads)

import maskrcnn_module.custom_dataset_mapper
importlib.reload(maskrcnn_module.custom_dataset_mapper)
from maskrcnn_module.custom_dataset_mapper import CustomDatasetMapper

from maskrcnn_module.utils import *

import maskrcnn_module.custom_predictor
importlib.reload(maskrcnn_module.custom_predictor)
from maskrcnn_module.custom_predictor import CustomPredictor

class Maskrcnn_Module():
    def __init__(self, model_weight_path_1, model_weight_path_2, score_thresh=0.2, fps=30, resolution=[640, 480], camera_id = None):
        # Maskrcnn settings
        self.model_weight_path_1 = model_weight_path_1
        self.model_weight_path_2 = model_weight_path_2
        self.path_to_val = 'place_holder.npy'
        self.score_thresh = score_thresh
        # Realsense settings (Video streaming configuration)
        self.fps_color = fps # FPS for color-only videos and for color+depth videos
        self.resolution_color = resolution  
        self.resolution_depth = resolution 
        self.camera_id = camera_id

    ######################################
    # Maskrcnn settings and initalize
    ######################################

    # register dataset
    def get_val_dataset(self):
        return np.load(self.path_to_val, allow_pickle=True)

    '''
    Description: Setup configurations for detectron2 model
    Inputs: first_init is True if init_model is being called for the first time
    Effects: Creates cfg object for model and registers dataset
    Returns: cfg object
    '''
    def init_model(self, first_init=True, model_weights=None):
        # random.seed(0)

        # register dataset
        if first_init:
            DatasetCatalog.register("val", self.get_val_dataset)
            # realsense images are flipped left and right from reality so also flip the class names
            MetadataCatalog.get("val").set(thing_classes=["Left-hinged", "Right-hinged", "Bottom-hinged", "Pulls out", "Top-hinged"])
            MetadataCatalog.get("val").set(keypoint_names=["handle_2d"])
            MetadataCatalog.get("val").set(keypoint_flip_map=[("handle_2d", "handle_2d")])

        # setup model
        cfg = config.get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"] 
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.KEYPOINT_ON = True # False
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 1
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.OUTPUT_DIR = ""
        cfg.DATASETS.TRAIN = ('val',) # set to val becuase it throws an error otherwise
        cfg.DATASETS.TEST = ('val',)

        # added cfg options to automate running experiments
        cfg.NUM_FC = 2 # 8 # 4 # 2
        cfg.HANDLE_LOSS = 1.0
        cfg.AXIS_LOSS = 1.0
        cfg.SURFNORM_LOSS = 1.0

        cfg.INPUT.MIN_SIZE_TEST = 800 # 0 # 800
        cfg.INPUT.RANDOM_FLIP = "none"

        cfg.MODEL.ROI_HEADS.NAME = "CustomStandardROIHeads"
        # threshold on prediction score
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
        # not evaluating the agnostic model
        cfg.MODEL.AGNOSTIC = False
        cfg.MODEL.CUSTOM_TRAINER = False # whether to use the custom_simply_trainer (gradient accumulation) or not
        cfg.MODEL.PROJ_HEAD = False # whether to use heads with projection or angle loss or not

        cfg.COLOR_JITTER_MIN = 0.9
        cfg.COLOR_JITTER_MAX = 1.
        cfg.COLOR_JITTER_SCALE = 0.1

        cfg.ERROR_HEAD_INPUT_TYPE = 3

        cfg.ERROR_HEAD_OUTPUT_TYPE = 1

        cfg.SURFNORM_OUTPUT_DIM = 1
        cfg.PREDICT_STD = 0
        cfg.QUANTILE_REG = 0

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        return cfg

    def create_predictor(self):
        self.cfg = self.init_model()
        self.predictor = CustomPredictor(self.cfg, self.model_weight_path_1, self.model_weight_path_2)

    ####################################
    # Realsense settings and initialize
    ####################################
    def init_camera(self):
        # Configure depth/color streams and specify which camera to use.
        # Note: width/heights are swapped since the images will be rotated 90 degrees from what the camera captures.
        #       The displayed images are also mirrored from reality
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.resolution_depth[0], self.resolution_depth[1], rs.format.z16, self.fps_color) 
        config.enable_stream(rs.stream.color, self.resolution_color[0], self.resolution_color[1], rs.format.bgr8, self.fps_color)
        if self.camera_id is not None:
            config.enable_device(self.camera_id)
        return config

    def create_camera_stream(self):
        config = self.init_camera()
        rs_cfg = self.pipeline.start(config)
        # get camera intrinsics
        profile = rs_cfg.get_stream(rs.stream.color)
        self.camera_intr = profile.as_video_stream_profile().get_intrinsics()
        # for depth to color alignment
        align_to = rs.stream.color
        self.align = rs.align(align_to)
            
    def run_inference(self):
        self.create_predictor()
        self.create_camera_stream()

        frame_count_color = 0
        print("Press Ctrl-C to start motion planning")
        # keep track of most recent prediction parameters
        maskrcnn_predictions = None
        try:
            # the camera takes a bit to adjust/start so the first few frames are significantly darker
            # throw away the first 30 frames
            for i in range(30):
                frames = self.pipeline.wait_for_frames()
            
            depth_frame_arr = []
            while True:
                # use the 10 most recent frames for depth
                '''
                num_depth_images = 10
                frame_arr_buf = []
                for _ in range(num_depth_images):
                    frames = self.pipeline.wait_for_frames()
                    frame_arr_buf.append(frames)
                
                # align color and depth frames
                for frame in frame_arr_buf:
                    aligned_frames = self.align.process(frame)
                    depth_frame = aligned_frames.get_depth_frame() 
                    depth_frame_arr.append(depth_frame)
                    if len(depth_frame_arr) > num_depth_images:
                        depth_frame_arr = depth_frame_arr[-num_depth_images:]
                '''
                depth_frame_arr = []
                num_depth_images = 10
                for _ in range(num_depth_images):
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame() 
                    depth_frame_arr.append(depth_frame)
                
                # use the most recent frame for color
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays.
                depth_image_arr = []
                for depth_idx in range(num_depth_images):
                    depth_image = np.asanyarray(depth_frame_arr[depth_idx].get_data())
                    depth_image_arr.append(depth_image)
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first).
                for depth_idx in range(num_depth_images):
                    depth_image = depth_image_arr[depth_idx]
                    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=-0.04, beta=255.0), cv2.COLORMAP_OCEAN)
                    depth_image = np.moveaxis(depth_image, 0, 1)
                    depth_image_arr[depth_idx] = depth_image

                # Camera is mounted sideways so rotate and flip color image for display purposes.
                color_image = np.moveaxis(color_image, 0, 1)
                color_image = np.fliplr(color_image)

                # Maskrcnn predictions
                frame = color_image.copy()
                output = self.predictor(frame)

                maskrcnn_predictions = output['instances'].to("cpu").get_fields()
                pred_scores = maskrcnn_predictions['scores']
                pred_classes = maskrcnn_predictions['pred_classes']
                pred_masks = maskrcnn_predictions['pred_masks']
                pred_handles = maskrcnn_predictions['pred_keypoints']
                pred_axis_points = maskrcnn_predictions['pred_axis_points']
                pred_surf_norm = maskrcnn_predictions['pred_surf_norm']
                pred_handle_orientations = maskrcnn_predictions['pred_handle_orientation']
                
                # filter out overlapping predictions based on score
                # the predictions from maskrcnn are already sorted by score largest to smallest
                unique_pred_ind = []
                for i in range(len(pred_scores)):
                    pred_handle = pred_handles[i][0].numpy()
                    x_2d = pred_handle[0]
                    y_2d = pred_handle[1]
                    print("i: ", i, " x_2d: ", x_2d, " y_2d: ", y_2d, "class: ", pred_classes[i].item(), "score: ", pred_scores[i])
                    # check that handle is in frame
                    row = round(pred_handle[1])
                    col = round(pred_handle[0])
                    if row < 0 or row >= 640 or col < 0 or col >= 480:
                        continue
                    # compare against predictions already in use
                    already_used = False
                    for j in unique_pred_ind:
                        already_used = already_used or pred_masks[j][round(pred_handle[1])][round(pred_handle[0])]
                    
                    if not already_used and pred_scores[i] >= self.score_thresh:
                        unique_pred_ind.append(i)


                print("unique masks: ", unique_pred_ind)

                num_pred = 0
                v = Visualizer(frame[:, :, ::-1], scale=1, instance_mode=ColorMode.IMAGE_BW)
                # iterate through each detection
                for i in unique_pred_ind:
                    if pred_scores[i] >= self.score_thresh:
                        num_pred +=1
                        # overlay mask and class label
                        vis = v.overlay_instances(masks=BitMasks(pred_masks[i].unsqueeze(0)), assigned_colors=[(0,0,1)], labels=[MetadataCatalog.get('val').get('thing_classes')[pred_classes[i]]])
                
                polygons = []
                # overlay predicted handle points and fitted polygons
                if num_pred > 0:
                    pred_img = Image.fromarray(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB))
                    draw = ImageDraw.Draw(pred_img)
                for i in unique_pred_ind:
                    pred_handle = pred_handles[i][0].numpy()
                    x_2d = pred_handle[0]
                    y_2d = pred_handle[1]
                    # check for in bound
                    if x_2d < 0 or y_2d < 0 or round(x_2d) >= self.resolution_color[1] or round(y_2d) >= self.resolution_color[0]:
                        continue
                    # unflip point and check for reasonable depth 
                    depth = depth_frame.get_distance(round(y_2d), round(self.resolution_color[1] - x_2d - 1))
                    if depth <= 0.0:
                        continue

                    if pred_scores[i] >= self.score_thresh:
                        # draw projections
                        width, height = pred_img.size
                        scale_factor = 1/200
                        circle_width = width*scale_factor
                        pred_handle = pred_handles[i][0]
                        # draw pred handle point (green)
                        draw.ellipse([pred_handle[0] - circle_width, pred_handle[1] - circle_width, pred_handle[0] + circle_width, pred_handle[1] + circle_width], fill=(0,255,0))
                        # number handle points
                        draw.text((pred_handle[0], pred_handle[1] - 10*circle_width), f"{i}", (0, 0, 255))

                        # find convex hull of masks and overlay approximated polygon
                        contours, heiarchy = cv2.findContours(pred_masks[i].numpy().astype(np.ubyte), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        hull = cv2.convexHull(contours[0])
                        perim = cv2.arcLength(hull, True)
                        poly = cv2.approxPolyDP(hull, 0.02*perim, True)
                        polygons.append(poly)
                        print("i: ", i,"polygon: ", poly.shape)

                        pred_img_arr = np.asarray(pred_img).copy()
                        cv2.drawContours(pred_img_arr, [poly], 0, (0, 0, 255), 1)

                        # convert back to Image for handle drawing
                        pred_img = Image.fromarray(pred_img_arr)
                        draw = ImageDraw.Draw(pred_img)

                        # handle orientation
                        # ["no handle annotation (ie no visible handle)", "circular handle (ie knob)", "vertical handle", "horizontal handle"]
                        print(f"Handle orientation of {i}'th object:", pred_handle_orientations[i])

                cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
                if num_pred > 0:
                    cv2.imshow('Realsense', np.array(pred_img))
                    # cv2.imshow('Depth', depth_image_arr[-1])
                else:
                    cv2.imshow('Realsense', frame)
                if cv2.waitKey(1) & 0xFF != 255:
                    raise KeyboardInterrupt()
                        
                print(f"Number of Predictions for Frame {frame_count_color}: {num_pred}")
                frame_count_color += 1

        except KeyboardInterrupt:
            ####################################
            # Ask user which prediction to use
            ####################################
            cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            if num_pred > 0:
                cv2.imshow('Realsense', np.array(pred_img))
                # cv2.imwrite('offline_eval/pred_img.jpg', np.array(pred_img))
            else:
                cv2.imshow('Realsense', frame)
            user_input = int(input("Select prediction: "))
            pred_handles = maskrcnn_predictions['pred_keypoints']
            if len(pred_handles)>0 and user_input < len(pred_handles):
                pred_class = maskrcnn_predictions['pred_classes'][user_input].numpy()
                pred_handle_orientation = maskrcnn_predictions['pred_handle_orientation'][user_input].numpy()
                pred_handle_orientation = 'vertical' if pred_handle_orientation[2] > pred_handle_orientation[3] else 'horizontal'
                
                class_names = ["Left-hinged", "Right-hinged", "Bottom-hinged", "Pulls out", "Top-hinged"]
                
                # input image was flipped so unflip predicted handle coordinates 
                pred_handle = pred_handles[user_input][0].numpy()
                x_2d = round(self.resolution_color[1] - pred_handle[0] - 1)
                y_2d = round(pred_handle[1])
                # convert to 3d coordinate
                depth = depth_frame.get_distance(y_2d, x_2d)
                result = rs.rs2_deproject_pixel_to_point(self.camera_intr, [y_2d, x_2d], depth)
                y_3d = result[0] # +x is right
                x_3d = result[1] # +y is down

                # points will be ordered from topmost point and continue clockwise
                poly = polygons[unique_pred_ind.index(user_input)]

                # ensure that first point is top right point
                lateral_diff = abs(poly[0][0][0] - poly[1][0][0])
                height_diff = abs(poly[0][0][1] - poly[1][0][1])
                if lateral_diff > height_diff:
                    poly =  np.append(poly, [poly[0]], axis=0)[1:] 
                
                # unflip points in polygon
                for i in range(len(poly)):
                    poly[i][0][0] = self.resolution_color[1] - poly[i][0][0] - 1

                # convert to 3d coordinate
                poly_3d = []
                for point in poly:
                    point_y = point[0][1]
                    point_x = point[0][0]
                    point_depth = depth_frame.get_distance(point_y, point_x)
                    point_3d = rs.rs2_deproject_pixel_to_point(self.camera_intr, [point_y, point_x], point_depth)
                    poly_3d.append(np.array([point_3d[1], point_3d[0], point_depth]))
                

                # plane-fitting
                # average depth image then do plane-fitting once
                '''if len(depth_frame_arr) > num_depth_images:
                    depth_frame_arr = depth_frame_arr[-num_depth_images:]
                depth_image_averaged = np.zeros((len(depth_frame_arr), 640, 480))
                for depth_idx in range(len(depth_frame_arr)):
                    depth_image = np.asanyarray(depth_frame_arr[depth_idx].get_data()) / 1000
                    depth_image = np.moveaxis(depth_image, 0, 1)
                    depth_image_averaged[depth_idx] = depth_image
                depth_image_averaged = np.average(depth_image_averaged, axis=0)
                pred_handle_3d_lookup, pred_handle_3d_plane, normal = deproject_pixel_to_point_plane(x_2d, y_2d, depth_image_averaged, maskrcnn_predictions['pred_masks'][user_input].numpy())
                '''
                print("DEPTH ARRAY LENGTH: ", len(depth_frame_arr))
                depth_image_averaged = np.zeros((len(depth_frame_arr), 640, 480))
                for depth_idx in range(len(depth_frame_arr)):
                    depth_image = np.asanyarray(depth_frame_arr[depth_idx].get_data()) / 1000
                    depth_image = np.moveaxis(depth_image, 0, 1)
                    depth_image_averaged[depth_idx] = depth_image
                depth_image_averaged = np.average(depth_image_averaged, axis=0)
                pred_mask = maskrcnn_predictions['pred_masks'][user_input].numpy().astype(np.uint8)
                # unflip mask
                pred_mask = np.fliplr(pred_mask)
                pred_handle_3d_lookup, pred_handle_3d_plane, normal = deproject_pixel_to_point_plane(x_2d, y_2d, depth_image_averaged, pred_mask)

                normal_corrected = np.array([normal[0], -normal[1], -normal[2]])

                # save output values
                handle_3d = np.array([x_3d, y_3d, depth])
                handle_3d_plus_surfnorm = handle_3d + normal_corrected 
                output_dict = {'handle_3d': np.array([x_3d, y_3d, depth]),
                                'handle_3d_plus_surfnorm': handle_3d_plus_surfnorm,
                                'classification': class_names[pred_class],
                                'polygon': np.array(poly_3d),
                                'handle_orientation': pred_handle_orientation}
                # np.save("maskrcnn_prediction_output.npy", output_dict)
                
            else:
                print("No prediction selected. Exiting")
                # np.save("maskrcnn_prediction_output.npy", np.array([]))
                output_dict = None            

            ####################################
            # Stop the streaming pipeline
            ####################################
            self.pipeline.stop()
            cv2.destroyAllWindows()
            return output_dict



