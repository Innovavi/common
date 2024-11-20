from typing import List

import cv2
import numpy as np


def irrectangle_to_bbox(irrectangle):
    left_x = np.min(irrectangle[:, 0])
    top_y = np.min(irrectangle[:, 1])
    right_x = np.max(irrectangle[:, 0])
    bot_y = np.max(irrectangle[:, 1])

    bbox = np.array([left_x, top_y, right_x, bot_y])

    return bbox


def scale_irrectangles(irrectangle, h_scale, w_scale):
    scaled_irrectangle = irrectangle.copy()
    scaled_irrectangle[..., 0] *= w_scale
    scaled_irrectangle[..., 1] *= h_scale

    return scaled_irrectangle


def get_irrectangle_dimensions(irrectangle):
    irrectangle_h1 = point_distance(irrectangle[0], irrectangle[2])
    irrectangle_h2 = point_distance(irrectangle[1], irrectangle[3])
    irrectangle_h = int(np.mean((irrectangle_h1, irrectangle_h2)))

    irrectangle_w1 = point_distance(irrectangle[0], irrectangle[1])
    irrectangle_w2 = point_distance(irrectangle[2], irrectangle[3])
    irrectangle_w = int(np.mean((irrectangle_w1, irrectangle_w2)))

    return irrectangle_h, irrectangle_w


def get_object_irrectangles(binary_image):
    contours = get_contours(binary_image)
    irrectangles = np.array([contour_to_irrectangle(cnt) for cnt in contours])

    bad_irrectangles_mask = get_bad_irrectangles_mask(irrectangles)
    irrectangles = irrectangles[bad_irrectangles_mask]

    return irrectangles


def get_bad_irrectangles_mask(irrectangles):
    bad_irrectangles_mask_x = irrectangles[:, 2, 0] > irrectangles[:, 0, 0]
    bad_irrectangles_mask_y = irrectangles[:, 2, 1] > irrectangles[:, 0, 1]

    bad_irrectangles_mask = np.bitwise_or(bad_irrectangles_mask_x, bad_irrectangles_mask_y)

    return bad_irrectangles_mask


def get_contours(binary_image):
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), 1, 2)

    return contours


def contour_to_irrectangle(cnt: List[np.ndarray]) -> np.ndarray:
    rect = cv2.minAreaRect(cnt)
    rect = cv2.boxPoints(rect)
    rect = np.int0(rect)

    return order_irrectangle_points(rect)


def order_irrectangle_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right (bot-left), and the fourth is the bottom-left (bot-right)
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def cv2_rect_to_irrectangle(cv2_rect):
    """
    Sorts points by their position in x and y axis separately, then gets 2 left points, checks which one is higher and lower and
    then the higher left point is ofcourse top_left and lower - bot_left. Does same to right side points.
    :param cv2_rect:
    :return: irrectangle
    """
    x_sort = np.argsort(cv2_rect[:, 0])
    y_sort = np.argsort(cv2_rect[:, 1])
    h_len = int(cv2_rect.shape[0] / 2)

    left_points_idx = x_sort[:h_len]
    right_points_idx = x_sort[h_len:]

    left_points = np.nonzero(np.in1d(y_sort, left_points_idx))[0]
    right_points = np.nonzero(np.in1d(y_sort, right_points_idx))[0]

    top_left_idx = min(left_points)
    bot_left_idx = max(left_points)
    top_right_idx = min(right_points)
    bot_right_idx = max(right_points)

    top_left_idx = y_sort[top_left_idx]
    bot_left_idx = y_sort[bot_left_idx]
    top_right_idx = y_sort[top_right_idx]
    bot_right_idx = y_sort[bot_right_idx]

    irrectangle = np.array([cv2_rect[top_left_idx], cv2_rect[top_right_idx],
                            cv2_rect[bot_left_idx], cv2_rect[bot_right_idx]])
    return irrectangle


def point_distance(point_1, point_2):
    return abs(np.linalg.norm(point_1 - point_2))