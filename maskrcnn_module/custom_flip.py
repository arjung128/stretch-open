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

from detectron2.data import transforms as T
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import transform_keypoint_annotations



def custom_transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints


    
    if "axis_points" in annotation:
        points = transform_axis_points_annotations(annotation['axis_points'], annotation['category_id'], transforms)
        annotation['axis_points'] = points

    if "handle" in annotation:
        points = transform_handle_annotations(annotation['handle'], transforms)
        annotation['handle'] = points

    if "surf_norm" in annotation:
        points = transform_surf_norm_annotations(annotation['surf_norm'], transforms)
        annotation['surf_norm'] = points

    
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    if do_hflip:
        if(annotation['category_id'] == 0):
            annotation['category_id'] = 1
        elif(annotation['category_id'] == 1):
            annotation['category_id'] = 0
    
    return annotation

def transform_axis_points_annotations(axis_points, category, transforms):
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    if do_hflip:
        axis_points[0] = -1 * axis_points[0]
        axis_points[3] = -1 * axis_points[3]
        # if top or bottom hinged flip order of points
        if category == 2 or category == 4:
            orig_left_point = axis_points[:3]
            orig_right_point = axis_points[3:]
            axis_points[:3] = orig_right_point
            axis_points[3:] = orig_left_point
    return axis_points

def transform_handle_annotations(handle, transforms):
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    if do_hflip:
        handle[0] = -1 * handle[0]
    return handle

def transform_surf_norm_annotations(surf_norm, transforms):
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
    if do_hflip:
        surf_norm[0] = -1 * surf_norm[0]
    return surf_norm