import argparse

import logging
import os
import sys
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import TORCH_VERSION, seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.engine import hooks
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase

import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import log_first_n

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler

from detectron2.config import configurable
from detectron2.engine.defaults import default_argument_parser, default_setup, default_writers, DefaultPredictor
from detectron2.data.build import build_batch_data_loader, build_detection_train_loader, build_detection_test_loader, get_detection_dataset_dicts, load_proposals_into_dataset, print_instances_class_histogram

import time
import quaternion
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.utils.file_io import PathManager

from detectron2.data.catalog import MetadataCatalog

import importlib
import maskrcnn_module.custom_flip
importlib.reload(maskrcnn_module.custom_flip)
from maskrcnn_module.custom_flip import custom_transform_instance_annotations

class CustomDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        #augmentations[1] = CustomHFlipTransform()    # i added, hacky way to hardcode random flips with surface normals
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.rand_point = False # True

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)

        # remove flipping
        augs = [augs[0]]
        print('augs:', augs)
        print('cfg:', cfg)
    
        recompute_boxes = False
        
        # '''
        # color_jitter_min, color_jitter_max = 0.9, 1.1
        # color_jitter_min, color_jitter_max = 0.5, 1.5
        # color_jitter_min, color_jitter_max = 0.1, 1.9
        # random_lighting_scale = 0.1
        color_jitter_min, color_jitter_max = cfg.COLOR_JITTER_MIN, cfg.COLOR_JITTER_MAX
        random_lighting_scale = cfg.COLOR_JITTER_SCALE
        print(color_jitter_min, color_jitter_max, random_lighting_scale)
        if is_train:
            # add other augmentations
            augs.append(T.RandomContrast(color_jitter_min, color_jitter_max))
            augs.append(T.RandomBrightness(color_jitter_min, color_jitter_max))
            augs.append(T.RandomSaturation(color_jitter_min, color_jitter_max))
            augs.append(T.RandomLighting(random_lighting_scale))
        # '''

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        # modify filename -- add period, remove s
        # COMMENT FOR INTEL DATASET:
        # dataset_dict["file_name"] = dataset_dict['file_name'][:-11] + dataset_dict['file_name'][-10:-3] + '.' + dataset_dict['file_name'][-3:]
        '''
        # for more data:
        if '.' in dataset_dict["file_name"]:
            dataset_dict["file_name"] ='/home/arjung2/drawers_clean/dataset_10x/train/compartment_faces/HeSYRw7eMtG/NC10_rgb_no_mask_img' + dataset_dict["file_name"].split('_')[-1][2:]
        else:
            dataset_dict["file_name"] = dataset_dict['file_name'][:-11] + dataset_dict['file_name'][-10:-3] + '.' + dataset_dict['file_name'][-3:]
        '''

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        # file_name = dataset_dict["file_name"]
        # format = self.image_format
        # with PathManager.open(file_name, "rb") as f:
        #     image = Image.open(f)
        #     down_sampled = image.resize((256, 192))
        #     up_sampled = down_sampled.resize((1920, 1440))
        #     image = utils._apply_exif_orientation(up_sampled)
        #     image = utils.convert_PIL_to_numpy(image, format)

        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)

        # confirming that detectron2 resize w/ bicubic interpolation and PIL Image's interpolation return same result
        # temp = Image.fromarray(image).resize((256,192))
        # print(np.array(temp).shape == aug_input.image.shape)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # print('image_shape:', image_shape)
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # rand point
        if self.rand_point:
            # when this is true, there is only one annotation per image
            assert len(dataset_dict["annotations"]) == 1

            # normalize rand_point
            normalized_randpoint = np.array(dataset_dict["annotations"][0]['rand_point']) / 1920
            normalized_randpoint *= 2
            normalized_randpoint -= 1

            blank_img = torch.ones_like(dataset_dict["image"], dtype=torch.double)
            blank_img[0, :, :] *= torch.tensor(normalized_randpoint[0])
            blank_img[1, :, :] *= torch.tensor(normalized_randpoint[1])
            blank_img[2, :, :] *= torch.tensor(0.)

            dataset_dict["image"] = blank_img.float()

        '''
        # store pc_camera_frame
        scene_name = dataset_dict["file_name"].split('/')[-2]
        image_id = dataset_dict["file_name"][-7:-4]
        pc_camera_frame = np.load(f'/home/arjung2/drawers_clean/maskrcnn_dataset/data_axis_thresh_posDepth_handleFix_v2_pcCamera/pc_camera_frames/{scene_name}_img{image_id}.npy')
        dataset_dict['pc_camera_frame'] = torch.from_numpy(pc_camera_frame)
        '''

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        #if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            #dataset_dict.pop("annotations", None)
            #dataset_dict.pop("sem_seg_file_name", None)
            #return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                # utils.transform_instance_annotations(
                custom_transform_instance_annotations(        # must also augment surface normals
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = custom_annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict



def custom_annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(np.array(boxes))

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    '''
    if(len(annos)) and "axis_points" in annos[0]:     
        target.gt_axis_points = torch.tensor(np.array([obj['axis_points']for obj in annos])) # added to get 3d axis points in the dataset
    '''

    # if(len(annos)) and "handle" in annos[0]:     
    #     target.gt_handle = torch.tensor(np.array([obj['handle']for obj in annos])) # added to get 3d handle point in the dataset
    if(len(annos)) and "handle_err" in annos[0]:     
        target.gt_handle_err = torch.tensor(np.array([obj['handle_err']for obj in annos])) # added to get 3d handle point in the dataset

    # if(len(annos)) and "surf_norm" in annos[0]:     
    #     target.gt_surf_norm = torch.tensor(np.array([obj['surf_norm']for obj in annos])) # added to get 3d surface normal vector in the dataset
    if(len(annos)) and "surf_norm_err" in annos[0]:     
        target.gt_surf_norm_err = torch.tensor(np.array([obj['surf_norm_err']for obj in annos])) # added to get 3d surface normal vector in the dataset

    # 1D planar surf norm err
    if(len(annos)) and "surf_norm_err_1d" in annos[0]:     
        target.gt_surf_norm_err_1d = torch.tensor(np.array([obj['surf_norm_err_1d']for obj in annos])) # added to get 3d surface normal vector in the dataset
        # target.gt_surf_norm_err_1d = torch.tensor(np.array([obj['surf_norm_err_1d']*(180/np.pi) for obj in annos])) # added to get 3d surface normal vector in the dataset # DEGREES

    if(len(annos)) and "2d_keypoints_err" in annos[0]:     
        # target.gt_2d_keypoints_err = torch.tensor(np.array([obj['2d_keypoints_err'][:2] for obj in annos])) # added to get 3d surface normal vector in the dataset
        # normalize
        target.gt_2d_keypoints_err = torch.tensor(np.array([obj['2d_keypoints_err'][:2] for obj in annos])/100) # added to get 3d surface normal vector in the dataset

        # categorize
        # median is 1.25
        bins = np.array([0.5, 1., 1.5, 2., 2.5, 3., 10.])
        twod_keypoints_err = np.array([np.abs(obj['2d_keypoints_err'][:2]) for obj in annos])
        inds = np.digitize(twod_keypoints_err, bins)
        target.gt_2d_keypoints_categorical_err = torch.tensor(inds)

    '''
    # if(len(annos)) and "handle_2d" in annos[0]:     
    if(len(annos)) and "keypoints" in annos[0]:     
        # target.gt_keypoints = torch.tensor(np.array([obj['handle_2d']for obj in annos])) # added to get 2d handle point in the dataset
        # target.gt_keypoints = Keypoints([obj['handle_2d']for obj in annos]) # added to get 2d handle point in the dataset
        keypoints = [obj['keypoints'] for obj in annos]
        # print('original keypoint:', keypoints[0])
        # keypoints = [[obj['keypoints'][0][1], obj['keypoints'][0][0], obj['keypoints'][0][2]] for obj in annos] # flipped
        keypoints = np.array(keypoints)
        # keypoints = np.expand_dims(keypoints, 1) # flipped
        # print('new keypoint:', keypoints[0])
        # print(keypoints.shape)
        target.gt_keypoints = Keypoints(keypoints)
    '''
    
    # if(len(annos)) and "rotation" in annos[0]:     
    #     target.rotation = torch.tensor(np.array([quaternion.as_float_array(obj['rotation']) for obj in annos])) # added to get 3d surface normal vector in the dataset
    
    # if(len(annos)) and "position" in annos[0]:     
    #     target.position = torch.tensor(np.array([obj['position']for obj in annos])) # added to get 3d surface normal vector in the dataset
    

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(np.copy(x))) for x in masks])
            )
        target.gt_masks = masks

    '''
    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)
    '''

    return target
