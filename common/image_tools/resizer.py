import cv2
import numpy as np
from typing import Tuple, Optional
from enum import Enum

from common.data_manipulation.image_data_tools.landmark_tools import change_landmarks_origin_point
from common.data_manipulation.image_data_tools.bounding_box_tools import change_bbox_origin_point


class ResizingType(Enum):
    """
    Aspect ratio method to use.
    FIXED: Forces given dimensions.
    MAX: Resizes based on bigger dimension.
    MIN: Resizes based on smaller dimension.
    """
    FIXED = 0
    MAX = 1
    MIN = 2


def resize_image(image: np.ndarray, desired_shape: Tuple[int, int] = (35, 35), resizing_type: ResizingType = ResizingType.MAX,
                 add_padding: Optional[bool] = False, padding_value: Tuple[int, int, int] = (0, 0, 0), interpolation: Optional[int] = None) -> np.ndarray:
    """
    Resize given image while preserving its aspect ratio. Resizes in a way that dimensions are either the same size or smaller.
    :param image: Image to resize.
    :param desired_shape: New image shape.
    :param resizing_type: The dimension to match with desired size. If ResizingType.BIGGER_DIMENSION, then resizes in a way that output dimensions are
    either the same size or smaller. If resizing_type.SMALLER_DIMENSION, then resizes in a way that output dimensions are either the same size or bigger.
    :param add_padding:
    :param padding_value:
    :param interpolation: Interpolation to use. If None, uses the best one.
    :return: Resized image.
    """
    new_height, new_width = get_resized_shape(image.shape, desired_shape, resizing_type)

    if interpolation is None:
        interpolation = get_interpolation_method(image.shape[:2], desired_shape)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    if add_padding:
        resized_image, padding_bbox = pad_image(resized_image, desired_shape, border_value=padding_value)

    return resized_image


def resize_image_with_data(image: np.ndarray, desired_shape: Tuple[int, int] = (256, 256), bounding_box: np.ndarray = None, landmarks: np.ndarray = None,
                           resizing_type: ResizingType = ResizingType.MAX, add_padding: bool = False, interpolation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resizes the image and landmarks to the desired size.
    :param image:
    :param desired_shape:
    :param bounding_box:
    :param landmarks:
    :return: Resized image, landmarks and bounding box.
    """
    resized_bounding_box, resized_landmarks = None, None

    resized_image = resize_image(image, desired_shape, resizing_type, interpolation=interpolation)

    image_h, image_w = image.shape[:2]
    desired_h, desired_w = resized_image.shape[:2]

    x_scale = desired_w / image_w
    y_scale = desired_h / image_h

    if add_padding:
        resized_image, padding_bbox = pad_image(resized_image, desired_shape)

    if bounding_box is not None and len(bounding_box) > 0:
        resized_bounding_box = bounding_box.copy().astype(np.float32)
        resized_bounding_box[..., 0] *= x_scale
        resized_bounding_box[..., 1] *= y_scale
        resized_bounding_box[..., 2] *= x_scale
        resized_bounding_box[..., 3] *= y_scale

        if add_padding:
            resized_bounding_box = change_bbox_origin_point(resized_bounding_box, destination_origin_point=padding_bbox[:2])

    if landmarks is not None and len(landmarks) > 0:
        resized_landmarks = landmarks.copy()
        resized_landmarks[..., 0] = resized_landmarks[..., 0] * x_scale
        resized_landmarks[..., 1] = resized_landmarks[..., 1] * y_scale

        if add_padding:
            resized_landmarks = change_landmarks_origin_point(resized_landmarks, destination_origin_point=padding_bbox[:2])

    return resized_image, resized_bounding_box, resized_landmarks


def resize_image_to_fit_divisor(image: np.ndarray, divisor: int = 64, round_up: bool = None) -> np.ndarray:
    image_height, image_width = image.shape[:2]

    multiple_h = calculate_nearest_to_divisor(divisor, image_height, round_up)
    multiple_w = calculate_nearest_to_divisor(divisor, image_width, round_up)

    interp = get_interpolation_method((image_height, image_width), (multiple_h, multiple_w))
    resized_image = cv2.resize(image, (multiple_w, multiple_h), interpolation=interp)

    return resized_image


def pad_image(image: np.ndarray, desired_shape: Tuple[int, int], border_type: int = cv2.BORDER_CONSTANT, border_value: int = (0, 0, 0), center_image: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pads the image.
    :param image: Image to pad.
    :param desired_shape: New shape.
    :param border_type: Type of border filling to use. See cv2.BORDER for options.
    :param border_value: If using BORDER_CONSTANT, set the constant value.
    :return: Padded image, padding_bbox.
    """
    image_height, image_width = image.shape[:2]
    desired_height, desired_width = desired_shape[:2]

    pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if desired_height > image_height:
        vertical_padding = (desired_height - image_height) / 2
        pad_top, pad_bot = (np.floor(vertical_padding).astype(int), np.ceil(vertical_padding).astype(int)) if center_image else (0, int(vertical_padding * 2))

    if desired_width > image_width:
        horizontal_padding = (desired_width - image_width) / 2
        pad_left, pad_right = (np.floor(horizontal_padding).astype(int), np.ceil(horizontal_padding).astype(int)) if center_image else (0, int(horizontal_padding * 2))

    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right, borderType=border_type, value=border_value)

    padding_bbox = np.array([pad_left, pad_top, pad_right, pad_bot])

    return padded_image, padding_bbox


def get_resized_shape(image_shape, desired_shape, resizing_type):
    image_height, image_width = image_shape[:2]
    desired_height, desired_width = desired_shape[:2]

    im_ratio = image_width / image_height
    s_ratio = desired_width / desired_height

    # horizontal image
    if (im_ratio > s_ratio and resizing_type == resizing_type.MAX) or (im_ratio < s_ratio and resizing_type == resizing_type.MIN):
        new_width = desired_width
        new_height = np.round(new_width / im_ratio)

    # vertical image
    elif (im_ratio < s_ratio and resizing_type == resizing_type.MAX) or (im_ratio > s_ratio and resizing_type == resizing_type.MIN):
        new_height = desired_height
        new_width = np.round(new_height * im_ratio)

    # square image or resizing without preserving aspect ration
    else:
        new_height, new_width = desired_height, desired_width

    new_height, new_width = int(new_height), int(new_width)

    return new_height, new_width


def calculate_nearest_to_divisor(divisor, image_dimension, round_up=None):
    multiplier = image_dimension // divisor

    if round_up:
        multiplier += bool(image_dimension % divisor)

    elif round_up is None:
        multiplier += bool(image_dimension % divisor > divisor // 2)

    else:  # rounding down
        pass

    nearest_size = multiplier * divisor

    return nearest_size


def get_interpolation_method(image_shape: Tuple[int, int], desired_shape: Tuple[int, int]) -> int:
    """
    Finds the right interpolation method for resizing. INTER_AREA if downsampling; INTER_CUBIC if upsampling.
    :param image_shape:
    :param desired_shape:
    :return: Interpolation method.
    """
    image_height, image_width = image_shape[:2]
    desired_height, desired_width = desired_shape[:2]

    if image_height > desired_height or image_width > desired_width:
        # AREA is better for downsampling
        interp = cv2.INTER_AREA

    else:
        # CUBIC is better for upsampling
        interp = cv2.INTER_CUBIC

    return interp
