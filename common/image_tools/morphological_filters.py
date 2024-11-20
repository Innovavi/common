import cv2
import numpy as np
from typing import Tuple
from common.image_tools.image_loading import to_uint8

"""
Source for all following filters: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
"""


def erosion(thresholded_image, kernel_shape, iterations=1, invert=False):
    # type: (np.ndarray, Tuple[int, int], int, bool) -> np.ndarray
    """
    Erodes the image.
    """
    thresholded_image_uint8 = to_uint8(thresholded_image, invert)
    kernel = np.ones(kernel_shape, np.uint8)

    return cv2.erode(thresholded_image_uint8, kernel, iterations=iterations)


def dilation(thresholded_image, kernel_shape, iterations=1, invert=False):
    # type: (np.ndarray, Tuple[int, int], int, bool) -> np.ndarray
    """
    Dilates the image.
    """
    thresholded_image_uint8 = to_uint8(thresholded_image, invert)
    kernel = np.ones(kernel_shape, np.uint8)

    return cv2.dilate(thresholded_image_uint8, kernel, iterations=iterations)


def closing_filter(thresholded_image, kernel_shape, iterations=1, invert=False):
    # type: (np.ndarray, Tuple[int, int], int, bool) -> np.ndarray
    """
    Dilates then erodes the image.
    """
    thresholded_image_uint8 = to_uint8(thresholded_image, invert)
    kernel = np.ones(kernel_shape, np.uint8)

    return cv2.morphologyEx(thresholded_image_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def open_filter(thresholded_image, kernel_shape, iterations=1, invert=False):
    # type: (np.ndarray, Tuple[int, int], int, bool) -> np.ndarray
    """
    Erodes then dilates the image.
    """
    thresholded_image_uint8 = to_uint8(thresholded_image, invert)
    kernel = np.ones(kernel_shape, np.uint8)

    return cv2.morphologyEx(thresholded_image_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations)

