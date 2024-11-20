from typing import List, Tuple, Union, Any, Dict, Optional

import numpy as np
from common.miscellaneous import verbose_print

from common.data_manipulation.image_data_tools.bounding_box_tools import merge_bboxes, get_bboxes_overlap, \
    bbox_to_string, \
    get_bbox_dimensions, \
    get_bbox_centroids, standardise_xywh_bbox, expand_bbox
from common.data_manipulation.distributions import get_flattened_distribution
from common.visualizations.image_visualizations import draw_blob


ROI_BBOX_COLUMNS = ['RoI_x_min', 'RoI_y_min', 'RoI_x_max', 'RoI_y_max']


def generate_roi(bbox: np.ndarray, image_shape: Tuple[int, int], roi_config: Dict[str, Any], landmarks: Optional[np.ndarray] = None, mask_class: bool = False, verbose=0):
    original_bb_h, original_bb_w = get_bbox_dimensions(bbox)
    scale_config = roi_config.get('scale_config', None)
    shift_config = roi_config.get('shift_config', None)

    # Removed image_shape from interfering with roi bboxes
    random_scale = calculate_expansion_coef(scale_config, None, (original_bb_h, original_bb_w))

    scaled_bbox = expand_bbox(bbox, random_scale)
    scaled_height, scaled_width = get_bbox_dimensions(scaled_bbox)

    if landmarks is not None:
        lm_index = 1 + np.random.randint(len(landmarks) - 1) if mask_class else np.random.randint(len(landmarks))
        center_point = landmarks[lm_index]
    else:
        center_point = get_bbox_centroids(bbox)

    if shift_config is not None:
        shift_x_prec, shift_y_prec = calculate_shifts(shift_config, random_scale)

        shift_x = int(shift_x_prec * original_bb_w / 2)
        shift_y = int(shift_y_prec * original_bb_h / 2)
    else:
        shift_x_prec, shift_y_prec = 0, 0
        shift_x, shift_y = 0, 0

    RoI_xyhw_bbox = np.array([center_point[0] + shift_x, center_point[1] + shift_y, scaled_width, scaled_height])
    RoI_bbox = standardise_xywh_bbox(RoI_xyhw_bbox).astype(int)

    verbose_print("image_shape: {} | random_scale: {} | shift_x: {} | shift_y: {}".format(image_shape, random_scale, shift_x_prec, shift_y_prec), verbose, 1)
    verbose_print("original bbox:   {}".format(bbox_to_string(bbox)), verbose, 1)
    verbose_print("expanded_box:    {}".format(bbox_to_string(scaled_bbox)), verbose, 1)
    verbose_print("shifted_RoI_box: {}".format(bbox_to_string(RoI_bbox)), verbose, 1)

    return RoI_bbox, {"random_scale": random_scale, "shift_x": shift_x_prec, "shift_y": shift_y_prec}


def calculate_expansion_coef(scale_config, image_shape = None, original_bbox_shape = None):
    loc, scale = scale_config.get('loc', 0), scale_config['scale']
    min_scale, max_scale = scale_config.get('min', 0), scale_config.get('max', 0)

    re_loc, re_scale = scale_config.get('re_loc', 0), scale_config.get('re_scale', 0)
    re_sharpness, re_count = scale_config.get('re_sharpness', 0), scale_config.get('re_count', 0)

    if image_shape is not None and original_bbox_shape is not None:
        im_height, im_width = image_shape[:2]
        original_bb_h, original_bb_w = original_bbox_shape

        max_possible_scale = max([(im_height / original_bb_h), (im_width / original_bb_w)]) / 2
        max_scale = min([max_scale, max_possible_scale])
        min_scale = min([min_scale, max_scale])

    random_scale = get_flattened_distribution(loc, scale, min_scale, max_scale, re_loc, re_scale, re_sharpness, re_count)

    return random_scale


def calculate_shifts(shifts_config: Dict[str, float], roi_scale: float = 0.) -> Tuple[int, int]:
    roi_scaling = shifts_config.get('roi_scaling', 1)

    max_x_scale = shifts_config['max_x'] + roi_scale * roi_scaling
    max_y_scale = shifts_config['max_y'] + roi_scale * roi_scaling

    x_scale = shifts_config['x_scale']
    y_scale = shifts_config['y_scale']

    re_scale = shifts_config.get('re_scale', 0)
    re_sharpness = shifts_config.get('re_sharpness', 0)

    random_shift_x = get_flattened_distribution(0, x_scale, -max_x_scale, max_x_scale, 0, re_scale, re_sharpness)
    random_shift_y = get_flattened_distribution(0, y_scale, -max_y_scale, max_y_scale, 0, re_scale, re_sharpness)

    return random_shift_x, random_shift_y


def calculate_center_freedom_dims(RoI_bboxes: np.ndarray, original_bboxes: np.ndarray, max_shift_x: float = 1, max_shift_y: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get freedomes in xy based on centers. Logic is that the center of original bbox has to be inside the RoI bbox.
    :param RoI_bboxes:
    :param original_bboxes:
    :param max_shift_x:
    :param max_shift_y:
    :return: x_freedomes, y_freedomes
    """
    original_bboxes_centers_x = (original_bboxes[..., 0] + original_bboxes[..., 2]) / 2
    original_bboxes_centers_y = (original_bboxes[..., 1] + original_bboxes[..., 3]) / 2

    x_freedomes = np.minimum(original_bboxes_centers_x - RoI_bboxes[..., 0], RoI_bboxes[..., 2] - original_bboxes_centers_x) * max_shift_x
    y_freedomes = np.minimum(original_bboxes_centers_y - RoI_bboxes[..., 1], RoI_bboxes[..., 3] - original_bboxes_centers_y) * max_shift_y

    return x_freedomes, y_freedomes


def calculate_shift_freedom_dims(scaled_squares, original_bboxes, max_shift_x=1, max_shift_y=1):
    left_freedoms = (original_bboxes[..., 0] - scaled_squares[..., 0]) * max_shift_x
    top_freedoms = (original_bboxes[..., 1] - scaled_squares[..., 1]) * max_shift_y

    right_freedoms = (scaled_squares[..., 2] - original_bboxes[..., 2]) * max_shift_x
    bot_freedoms = (scaled_squares[..., 3] - original_bboxes[..., 3]) * max_shift_y

    freedomes = np.stack([left_freedoms, top_freedoms, right_freedoms, bot_freedoms], axis=1).astype(np.int)

    return freedomes


def randomise_shifts_uniform(freedomes, absolute=False):
    if absolute:
        freedomes = np.absolute(freedomes)

    random_shift_x = np.zeros(len(freedomes))
    valid_x_indice = np.where(-freedomes[:, 0] < freedomes[:, 2])
    random_shift_x[valid_x_indice] = np.random.randint(-freedomes[valid_x_indice, 0], freedomes[valid_x_indice, 2])

    random_shift_y = np.zeros(len(freedomes))
    valid_y_indice = np.where(-freedomes[:, 1] < freedomes[:, 3])
    random_shift_y[valid_y_indice] = np.random.randint(-freedomes[valid_y_indice, 1], freedomes[valid_y_indice, 3])

    return random_shift_x, random_shift_y


# def randomise_shifts_normal(x_freedomes: Union[np.ndarray, float], y_freedomes: Union[np.ndarray, float], roi_config: Dict[str, Any]) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
#     """
#     Randomises x and y shifts based on normal distribution with given parameters.
#     :param x_freedomes:
#     :param y_freedomes:
#     :param roi_config:
#     :return: x_shifts, y_shifts
#     """
#     shift_x_scale = roi_config.get('x_scale', 1)
#     shift_y_scale = roi_config.get('y_scale', 1)
#
#     x_shifts = get_normal_in_abs_range(x_freedomes, 0, shift_x_scale)
#     y_shifts = get_normal_in_abs_range(y_freedomes, 0, shift_y_scale)
#
#     random_scale = np.random.normal(0, scale)
#
#     if any(np.array([re_loc, re_scale, re_sharpness]) != 0):
#         random_scale = redistribute_top_proba(random_scale, loc, scale, re_loc, re_scale, re_sharpness)
#
#     random_scale = redistribute_offlier(random_scale, loc, scale, min_scale_thresh, max_scale_thresh)
#
#     return x_shifts, y_shifts


def merge_RoIs(rois: Union[List, np.ndarray], merge_rois_overlap_threshold: float) -> Tuple[bool, Union[List, np.ndarray]]:
    for i, roi_1 in enumerate(rois):
        for j, roi_2 in enumerate(rois):
            if j <= i:
                continue

            overlap = get_bboxes_overlap(roi_1, roi_2)

            if overlap >= merge_rois_overlap_threshold:
                merged_bb = merge_bboxes(roi_1, roi_2)

                rois[i] = merged_bb

                # RoIs_with_bbs[i][0] = merged_bb
                # RoIs_with_bbs[i][1].extend(bbs_2)

                del rois[j]

                return True, rois

    return False, rois


def hide_other_faces(image, original_bounding_box, other_bounding_boxes):
    hidden_image = image.copy()
    for other_bounding_box in other_bounding_boxes:
        if (other_bounding_box == original_bounding_box).all():
            continue

        hidden_image = draw_blob(hidden_image, other_bounding_box)

    return hidden_image


def get_bboxes_RoI(RoI_bbox, bbox, RoI_expansion=0, normalize_shifts=False):
    RoI_bbox_h = RoI_bbox[..., 3] - RoI_bbox[..., 1]
    RoI_bbox_w = RoI_bbox[..., 2] - RoI_bbox[..., 0]
    RoI_bbox_size = np.max([RoI_bbox_h, RoI_bbox_w], axis=0) * (1 + RoI_expansion)
    RoI_bbox_center_x = RoI_bbox[..., 0] + RoI_bbox_w / 2
    RoI_bbox_center_y = RoI_bbox[..., 1] + RoI_bbox_h / 2

    bbox_h = bbox[..., 3] - bbox[..., 1]
    bbox_w = bbox[..., 2] - bbox[..., 0]
    bbox_size = np.max([bbox_h, bbox_w], axis=0)
    bbox_center_x = bbox[..., 0] + bbox_w / 2
    bbox_center_y = bbox[..., 1] + bbox_h / 2

    RoI_scale = RoI_bbox_size / bbox_size
    RoI_shift_x = RoI_bbox_center_x - bbox_center_x
    RoI_shift_y = RoI_bbox_center_y - bbox_center_y

    if normalize_shifts:
        RoI_shift_x /= bbox_size
        RoI_shift_y /= bbox_size

    return RoI_scale, RoI_shift_x, RoI_shift_y