import sys
from enum import Enum
from typing import Union

import cv2
import numpy as np
from common.visualizations.figure_plotting import plot_histogram

from common.image_tools.image_loading import to_uint8


class PropertyType(Enum):
    """
    These constants are used for filtering images by object properties.
    """
    PROPERTY_TOP_X = 0
    PROPERTY_LEFT_Y = 1
    PROPERTY_WIDTH = 2
    PROPERTY_HEIGHT = 3
    PROPERTY_AREA = 4


def get_connected_object_properties(thresholded_image: np.ndarray, invert: bool = False):
    """
    Analyzes the image finding connected objects and each of their properties.
    :param thresholded_image: Binary image.
    :param property_indexes: Indices of properties which can be found in https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
    :param invert: Invert if the objects are black, while the background is white.
    :return: The specified properties of analyzed image connected objects.
    """
    thresholded_image_inv = to_uint8(thresholded_image, invert)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image_inv, connectivity=8)

    return nb_components - 1, output, stats[1:], centroids[1:]


def filter_by_object_property(thresholded_image: np.ndarray, property_index: PropertyType, min_value: int = 0, max_value: int = sys.maxsize, invert: bool = False):
    """
    Filters connected objects judging by the given property and min max parameters. The property is chosen from the 5 PROPERTY values above.
    :param thresholded_image: Image to filter.
    :param property_index: Index of property to filter by.
    :param min_values: Minimum values of objects that will be left on the image.
    :param max_values: Maximum values of objects that will be left on the image.
    :param invert: Invert if the objects are black, while the background is white.
    :return: Filtered image and object count.
    """
    object_count, objects_mask, properties, centroids = get_connected_object_properties(thresholded_image, invert)

    if object_count == 0:
        return thresholded_image, 0

    objects_property = properties[:, property_index.value]

    filtered_mask = thresholded_image.copy()
    filtered_object_count = 0

    for i in range(object_count):
        if objects_property[i] < min_value or objects_property[i] > max_value:
            filtered_mask[np.where(objects_mask == i + 1)] = 0
            filtered_object_count += 1

    return filtered_mask.astype(np.uint8), object_count - filtered_object_count


def get_component_properties(threshed_image, connectivity=8, do_plot=False):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(threshed_image, connectivity=connectivity)
    widths, heights, areas = stats[1:, 2], stats[1:, 3], stats[1:, 4]

    if do_plot:
        print("number of components", len(widths))
        print("widths")
        plot_histogram(widths, add_quatinles=True, fig_size=(22, 5))
        print("heights")
        plot_histogram(heights, add_quatinles=True, fig_size=(22, 5))
        print("areas")
        plot_histogram(areas, add_quatinles=True, fig_size=(22, 5))

    return widths, heights, areas