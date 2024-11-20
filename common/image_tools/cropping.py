from typing import Union, List, Tuple

import numpy as np
import cv2

from common.data_manipulation.image_data_tools.bounding_box_tools import get_4_point_bbox_dimensions, parse_bbox, clip_bbox_values


def crop_box(image: np.ndarray, bounding_box: Union[List, np.ndarray], do_clip: bool = False) -> np.ndarray:
    """
    Crops image based on the provided box.
    :param image: Image to crop.
    :param bounding_box: Box of format: [x_min, y_min, x_max, y_max].
    :param do_clip: .
    :return: Cropped image.
    """
    bounding_box = bounding_box.astype(int) if type(bounding_box) == np.ndarray else np.array([int(box_coor) for box_coor in bounding_box])
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bounding_box)
    image_shape = image.shape

    # Crop a clipped image, that will contain the needed pixels.
    clipped_box = clip_bbox_values(bounding_box, image.shape)
    clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max, clipped_bb_height, clipped_bb_width = parse_bbox(clipped_box)
    cropped_clipped_image = image[clipped_y_min:clipped_y_max, clipped_x_min:clipped_x_max]

    if do_clip:
        cropped_image = cropped_clipped_image

    else:  # If the bounding box was out of the image, this will make sure the image is put into the right coordinates of the mask.
        # First, calculate the top left coordinate.
        coors_to_put_image = [0, 0, clipped_bb_width, clipped_bb_height]
        if x_min < 0:
            coors_to_put_image[0] = abs(x_min)
            coors_to_put_image[2] += coors_to_put_image[0]
        if y_min < 0:
            coors_to_put_image[1] = abs(y_min)
            coors_to_put_image[3] += coors_to_put_image[1]

        # And then put the clipped image into the mask.
        # print("bb_h: {} | bb_w: {} | image_shape[2]: {}".format(bb_h, bb_w, image_shape[2]))
        cropped_image_shape = (bb_h, bb_w, image_shape[2]) if len(image_shape) > 2 else (bb_h, bb_w)
        cropped_image = np.zeros(cropped_image_shape, dtype=image.dtype)
        cropped_image[coors_to_put_image[1]:coors_to_put_image[3], coors_to_put_image[0]:coors_to_put_image[2]] = cropped_clipped_image

    return cropped_image


def crop_box_batch(image_list: Union[List[np.ndarray], np.ndarray], bbox_list: Union[List[np.ndarray], np.ndarray]):
    return [crop_box(image, bbox) for image, bbox in zip(image_list, bbox_list)]


def crop_irrectangle(image: np.ndarray, irrectangle: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops the image based on the four point box.
    :param image:
    :param box:
    :return: cropped_image, perspective_transform
    """
    box_h, box_w = get_4_point_bbox_dimensions(irrectangle)

    pts1 = np.float32(irrectangle)
    pts2 = np.float32([[0, 0], [box_w, 0], [0, box_h], [box_w, box_h]])

    perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)

    cropped_image = cv2.warpPerspective(image, perspective_transform, (box_w, box_h))

    return cropped_image, perspective_transform


def crop_image_center_square(image: np.ndarray) -> np.ndarray:
    """
    Crops a square from the image center with maximum possible dimensions within the image.
    :param image:
    :return: cropped_image
    """
    im_h, im_w = image.shape[:2]

    half_h = im_h / 2
    half_w = im_w / 2

    if im_h > im_w:
        top = int(half_h - half_w)
        bot = int(half_h + half_w)

        image = image[top:bot, :]

    elif im_h < im_w:
        left = int(half_w - half_h)
        right = int(half_w + half_h)

        image = image[:, left:right]

    return image
