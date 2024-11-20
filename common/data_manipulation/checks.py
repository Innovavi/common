from typing import List, Union

import cv2
import numpy as np

from common.data_manipulation.image_data_tools.bounding_box_tools import get_bbox_points, parse_bbox


def check_if_image_is_too_small(image: np.ndarray, size_threshold: int) -> bool:
    """
    Checks if any of image dimensions are smaller than given threshold.
    :param image:
    :param size_threshold:
    :return:
    """
    return min(image.shape[:2]) <= size_threshold


def check_if_image_is_too_blurry(image: np.ndarray, blur_threshold: int) -> bool:
    """
    Checks if any of image dimensions are smaller than given threshold.
    :param image:
    :param blur_threshold:
    :return:
    """
    return cv2.Laplacian(image, cv2.CV_64F).var() <= blur_threshold


def check_if_bbox_is_inside_image(bounding_box: Union[np.ndarray, List], image_shape: np.ndarray) -> bool:
    """
    Checks if the middle of bb is out of the image.
    :param bounding_box:
    :param image_shape:
    :return: True if both if bounding_box points are inside the image. False otherwise.
    """
    image_bounding_box = np.array([0, 0, image_shape[1], image_shape[0]])
    top_left_point, bottom_right_point = get_bbox_points(bounding_box)

    if not check_if_point_is_inside_bbox(top_left_point, image_bounding_box):
        return False

    if not check_if_point_is_inside_bbox(bottom_right_point, image_bounding_box):
        return False

    return True


def check_if_point_is_inside_bbox(point: Union[np.ndarray, List], box: Union[np.ndarray, List]) -> bool:
    """
    Checks if the given point is within the bbox dimensions.
    :param point:
    :param box:
    :return:
    """
    pt_x, pt_y = point
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(box)

    if x_min <= pt_x <= x_max:
        if y_min <= pt_y <= y_max:
            return True

    return False


def check_if_points_are_inside_bbox(points: np.ndarray, box: np.ndarray, do_any: bool = False, do_all: bool = False) -> bool:
    """
    Checks if the given points are within the bbox dimensions.
    :param points:
    :param box:
    :param do_any:
    :param do_all:
    :return:
    """
    assert do_any or do_all, "One of do_any or do_all must be set to True"

    if do_any:
        return any([check_if_point_is_inside_bbox(point, box) for point in points])

    elif do_all:
        return all([check_if_point_is_inside_bbox(point, box) for point in points])

    return False
