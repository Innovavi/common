from typing import Union, List, Tuple, Optional, Iterable

import numpy as np
from common.miscellaneous import verbose_print
from math import floor, ceil

BOUNDING_BOX_PRINT_INTEGER_TEMPLATE = "x_min={:3d} | y_min={:3d} | x_max={:3d} | y_max={:3d} | height={:3d} | width={:3d}"
BOUNDING_BOX_PRINT_FLOAT_TEMPLATE = "x_min={:.2f} | y_min={:.2f} | x_max={:.2f} | y_max={:.2f} | height={:.2f} | width={:.2f}"


def expand_bbox(bbox: np.ndarray, expansion_ratio: Union[List[float], float], image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Expands given box by the expansion ratio. Positive ratio will expand, while a negative will shrink, relative to center point.
    If image_shape is specified, off coordinates will be fixed too.
    :param bbox: Bounding box to resize.
    :param expansion_ratio: Percent of resizing. [X, Y]
    :param image_shape: Shape of the image that contains the box. Needed to clip box values.
    :return: Expanded box.
    """
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)
    x_expansion, y_expansion = (expansion_ratio, expansion_ratio) if isinstance(expansion_ratio, float) else expansion_ratio

    x_min = x_min - x_expansion * bb_w
    y_min = y_min - y_expansion * bb_h
    x_max = x_max + x_expansion * bb_w
    y_max = y_max + y_expansion * bb_h

    new_box = np.array([x_min, y_min, x_max, y_max])

    if image_shape is not None:
        new_box = fix_off_coordinates(new_box, image_shape[:2])

    return new_box


def resize_bbox(bbox: np.ndarray, expansion_ratio: Union[List[float], float], image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Resizes given box by the expansion ratio.
    If image_shape is specified, off coordinates will be fixed too.
    :param bbox: Box to resize
    :param expansion_ratio: Percent of resizing. [X, Y]
    :param image_shape: Shape of the image that contains the box. Needed to clip box values.
    :return: Resized box.
    """
    x_expansion, y_expansion = (expansion_ratio, expansion_ratio) if isinstance(expansion_ratio, float) else expansion_ratio

    bbox[..., [0, 2]] *= x_expansion
    bbox[..., [1, 3]] *= y_expansion

    if image_shape is not None:
        bbox = fix_off_coordinates(bbox, image_shape[:2])

    return bbox


def get_max_image_square_around_bbox(bounding_box: np.ndarray, image_shape: Tuple):
    im_height, im_width = image_shape[:2]

    # print("get_max_image_square_around_bbox | bounding_box:", type(bounding_box), bounding_box.shape, bounding_box)
    bounding_box_height, bounding_box_width = get_bbox_dimensions(bounding_box)
    x_centroid, y_centroid = get_bbox_centroids(bounding_box)

    left_space, top_space = x_centroid, y_centroid

    right_space = im_width - 1 - x_centroid
    bot_space = im_height - 1 - y_centroid

    min_dimension = min([left_space, top_space, right_space, bot_space]) * 2
    min_dimension = max([min_dimension, bounding_box_height, bounding_box_width])

    max_image_square_from_bounding_box = standardise_xywh_bbox(np.array([x_centroid, y_centroid, min_dimension, min_dimension]))

    return max_image_square_from_bounding_box


def resize_4_point_bbox(bbox: np.ndarray, expansion_coefs: List[float]) -> np.ndarray:
    """
    Resizes a 4 point box based on the expansion coef.
    :param bbox:
    :param expansion_coefs:
    :return: box
    """
    box_h, box_w = get_4_point_bbox_dimensions(bbox)
    #     print("box_h, box_w", box_h, box_w)

    vertical_flat_expansion = int((box_h * expansion_coefs[0] - box_h) / 2)
    horizontal_flat_expansion = int((box_w * expansion_coefs[1] - box_w) / 2)

    flat_expansion_array = np.array([horizontal_flat_expansion, vertical_flat_expansion])
    #     print("flat_expansion_array", flat_expansion_array)

    bbox[0] -= flat_expansion_array

    bbox[1, 0] += horizontal_flat_expansion
    bbox[1, 1] -= vertical_flat_expansion

    bbox[2, 0] -= horizontal_flat_expansion
    bbox[2, 1] += vertical_flat_expansion

    bbox[3] += flat_expansion_array

    return bbox


def get_4_point_bbox_dimensions(bbox: Union[List, np.ndarray]) -> Tuple[int, int]:
    """
    Calculates mean height and width of the given box.
    :param bbox: 4 point box
    :return: box_h, box_w
    """
    box_h1 = __point_distance(bbox[0], bbox[2])
    box_h2 = __point_distance(bbox[1], bbox[3])
    box_h = int(np.mean((box_h1, box_h2)))

    box_w1 = __point_distance(bbox[0], bbox[1])
    box_w2 = __point_distance(bbox[2], bbox[3])
    box_w = int(np.mean((box_w1, box_w2)))

    return box_h, box_w


def parse_bbox(bounding_box: Union[List, np.ndarray]) -> Tuple[int, int, int, int, int, int]:
    """
    Parses box info. Mainly used for reducing code rows.
    :param bounding_box: Box to Parse.
    :return: x_min, y_min, x_max, y_max, bb_h, bb_w
    """
    x_min = bounding_box[0]
    y_min = bounding_box[1]
    x_max = bounding_box[2]
    y_max = bounding_box[3]

    bb_h, bb_w = get_bbox_dimensions(bounding_box)

    return x_min, y_min, x_max, y_max, bb_h, bb_w


def get_bbox_centroids(bounding_box: np.ndarray) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Parses the centers of the given bounding box
    :param bounding_box:
    :return: x_centroid, y_centroid
    """
    x_centroid = np.mean([bounding_box[..., 0], bounding_box[..., 2]], axis=0)
    y_centroid = np.mean([bounding_box[..., 1], bounding_box[..., 3]], axis=0)

    return x_centroid, y_centroid


def get_bbox_dimensions(bbox: Union[List, np.ndarray], do_squeeze: Union[bool, int] = True) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """
    Returns box's height and width.
    :param bbox: Box to estimate.
    :return: bb_h, bb_w
    """
    # print("get_bbox_dimensions | bounding_box:", type(box), box.shape, box)
    x_min, y_min, x_max, y_max = np.split(bbox, 4, axis=-1)

    bb_height = y_max - y_min
    bb_width = x_max - x_min

    if isinstance(bb_height, np.ndarray) and (do_squeeze or (type(do_squeeze) == int and do_squeeze == 0)):
        bb_height = np.squeeze(bb_height) if type(do_squeeze) == bool else np.squeeze(bb_height, axis=do_squeeze)
        bb_width = np.squeeze(bb_width) if type(do_squeeze) == bool else np.squeeze(bb_width, axis=do_squeeze)

    return bb_height, bb_width


def get_bbox_points(bbox: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    top_left_point = bbox[:2]
    bottom_right_point = bbox[2:]

    return top_left_point, bottom_right_point


def get_bbox_area(bbox: Union[List, np.ndarray]) -> Union[float, np.ndarray]:
    bb_height, bb_width = get_bbox_dimensions(bbox)

    area = bb_height * bb_width

    return area


def fix_off_coordinates(bbox: np.ndarray, image_shape: Tuple[int, int], shift: bool = False) -> np.ndarray:
    """
    Fixes box coordinates that are over the boarders of the image. It either simply clips the off points or
    shifts the box to maintain the aspect ratio.
    :param bbox:
    :param image_shape:
    :param shift:
    :return: new_box
    """
    if shift:
        new_box = shift_bbox_values_to_fit_image(bbox, image_shape[:2])

    else:
        new_box = clip_bbox_values(bbox, image_shape[:2])

    return new_box


def shift_bbox_values_to_fit_image(bbox: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Shifts the box to fit inside the image dimensions. If the box is too large, it resizes it while maintaining the box aspect ratio
    :param bbox:
    :param image_shape:
    :return: new_box
    """
    im_h, im_w = image_shape[:2]
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)

    assert bb_h <= im_h or bb_w <= im_w, "image size is smaller than box size: bb_h: {}, im_h: {}, bb_w: {}, im_w: {}".format(bb_h, im_h, bb_w, im_w)

    x_min, x_max = shift_single_dimension_to_fit(x_min, x_max, im_w)
    y_min, y_max = shift_single_dimension_to_fit(y_min, y_max, im_h)

    new_box = np.array([x_min, y_min, x_max, y_max])

    return new_box


def shift_single_dimension_to_fit(dimension_min: int, dimension_max: int, dimension_size: int) -> Tuple[int, int]:
    """
    Shifts a box along a dimension. This is done by: if the min is below 0, it is set to 0 and the max is added the difference, or,
    if max is bigger than dim_size, max is set to dim_size and min is deducted the difference.
    :param dimension_size: Image dimensions along an axis.
    :param dimension_min: Box minimum point along an axis.
    :param dimension_max: Box maximum point along an axis.
    :return: dimension_min, dimension_max
    """
    if dimension_min < 0:
        dimension_max -= dimension_min
        dimension_min = 0

    elif dimension_max > dimension_size:
        dimension_min -= dimension_max - dimension_size
        dimension_max = dimension_size

    return dimension_min, dimension_max


def shift_bbox(bbox: Union[List, np.ndarray], shift_x: int = 0, shift_y: int = 0, image_shape: Tuple[int, int] = (0, 0)) -> np.ndarray:
    im_h, im_w = image_shape[:2]
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)

    # assert bb_h <= im_h or bb_w <= im_w, "image size is smaller than box size: bb_h: {}, im_h: {}, bb_w: {}, im_w: {}".format(bb_h, im_h, bb_w, im_w)

    x_min, x_max = shift_single_dimension(x_min, x_max, shift_x, im_w, dimension_name='x')
    y_min, y_max = shift_single_dimension(y_min, y_max, shift_y, im_h, dimension_name='y')

    new_box = np.array([x_min, y_min, x_max, y_max])

    return new_box


def shift_single_dimension(dimension_min: int, dimension_max: int, dimension_shift: int = 0, dimension_size: int = 0, dimension_name: str = '',
                           verbose: int = 0) -> Tuple[int, int]:
    """
    Shifts a box along a dimension. This is done by: if the min is below 0, it is set to 0 and the max is added the difference, or,
    if max is bigger than dim_size, max is set to dim_size and min is deducted the difference.
    :param dimension_min: Box minimum point along an axis.
    :param dimension_max: Box maximum point along an axis.
    :param dimension_shift: Amount to shift along an axis.
    :param dimension_size: Image dimensions along an axis.
    :return: dimension_min, dimension_max
    """
    dimension_min, dimension_max = dimension_min + dimension_shift, dimension_max + dimension_shift

    if dimension_size != 0:
        dimension_min, dimension_max = shift_single_dimension_to_fit(dimension_min, dimension_max, dimension_size)

        verbose_print("Dimension {} went out of bounds while shifting and was shifted back to fit".format(dimension_name), verbose, 1)

    return dimension_min, dimension_max


def clip_bbox_values(bbox: np.ndarray, image_shape: Tuple[int, int], min_values: Union[List[int], Tuple[int, int]] = [0, 0]) -> np.ndarray:
    """
    Clips all box points to fit them in the image.
    :param bbox:
    :param image_shape:
    :param min_values: If this is specified, minimum values are compared with these values.
    :return: box
    """
    im_h, im_w = image_shape[:2]

    x_min = np.clip(bbox[0], a_min=min_values[0], a_max=im_w)
    y_min = np.clip(bbox[1], a_min=min_values[1], a_max=im_h)
    x_max = np.clip(bbox[2], a_min=min_values[0], a_max=im_w)
    y_max = np.clip(bbox[3], a_min=min_values[1], a_max=im_h)

    return np.array([x_min, y_min, x_max, y_max])

def bbox_keep_asp_ratio(bbox: Union[List, np.ndarray], image_shape: Tuple[int,int]=None,
                        target_aspect_ration: float=3., verbose: int=0) -> Union[List, np.ndarray]:
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)
    new_height = bb_w * target_aspect_ration
    diff_height = new_height - bb_h

    y_min = ceil(y_min - diff_height/2)
    y_max = floor(y_max + diff_height/2)
    x_min = ceil(x_min)
    x_max = floor(x_max)

    new_box = np.array([x_min, y_min, x_max, y_max])

    if image_shape is not None:
        # if image size is smaller than the box size, shrink the box
        im_h, im_w = image_shape[:2]
        bb_h, bb_w = get_bbox_dimensions(new_box)

        if max([bb_h - im_h, bb_w - im_w]) > 0:
            if verbose > 0:
                print("square box is bigger than the image")
                print("bb_h, bb_w", bb_h, bb_w)
                print("im_h, im_w", im_h, im_w)

            new_box = shrink_bbox_to_fit_image(new_box, (im_h, im_w), verbose=verbose)
            new_box = bbox_keep_asp_ratio(new_box, (im_h, im_w))

            return new_box

        if verbose > 1:
            print("new_box before fix_off_coordinates", new_box)

        new_box = fix_off_coordinates(new_box, image_shape, shift=True)

        if verbose > 1:
            print("new_box before fix off coor", new_box)

    return new_box

def bbox_to_square(bbox: Union[List, np.ndarray], image_shape: Tuple[int, int] = None, verbose: int = 0) -> Union[List, np.ndarray]:
    """
    Transforms the box into a square by expanding the box to the bigger box dimension. Then, if the box goes off the image dimensions,
    the box is shrinked and shifted to fit it.
    :param bbox:
    :param image_shape:
    :param verbose: When 1, prints a message when the box goes off the image dimensions. When >1, also prints new_box before and after clipping.
    :return: new_box
    """
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)

    if bb_h > bb_w:
        x_min, x_max = expand_dims_to_match(bb_h, bb_w, x_min, x_max)

    elif bb_h < bb_w:
        y_min, y_max = expand_dims_to_match(bb_w, bb_h, y_min, y_max)

    new_box = np.array([x_min, y_min, x_max, y_max])

    if image_shape is not None:
        # if image size is smaller than the box size, shrink the box
        im_h, im_w = image_shape[:2]
        bb_h, bb_w = get_bbox_dimensions(new_box)

        if max([bb_h - im_h, bb_w - im_w]) > 0:
            if verbose > 0:
                print("square box is bigger than the image")
                print("bb_h, bb_w", bb_h, bb_w)
                print("im_h, im_w", im_h, im_w)

            new_box = shrink_bbox_to_fit_image(new_box, (im_h, im_w), verbose=verbose)
            new_box = bbox_to_square(new_box, (im_h, im_w))

            return new_box

        if verbose > 1:
            print("new_box before fix_off_coordinates", new_box)

        new_box = fix_off_coordinates(new_box, image_shape, shift=True)

        if verbose > 1:
            print("new_box before fix off coor", new_box)

    return new_box


def shrink_bbox_to_fit_image(bbox: Union[List, np.ndarray], image_shape: Tuple[int, int], verbose: int = 0) -> np.ndarray:
    """
    Works with a single axis being off.
    :param bbox: Box to shrink.
    :param image_shape: Image shape that contains the box.
    :param verbose: When >1, prints before and after shrinking.
    :return: Shrinked box.
    """
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)
    im_h, im_w = image_shape[:2]

    dimension_diff = np.array([bb_h - im_h, bb_w - im_w])
    if verbose > 0:
        print("shrinking box. Before shrinking:")
        print(x_min, y_min, x_max, y_max)
        print("dimension_diff", dimension_diff)

    if dimension_diff[0] > 0:
        half_diff = np.ceil(dimension_diff[0] / 2)
        x_min += half_diff
        x_max -= half_diff

        y_max = y_max - (dimension_diff[0] + y_min) if y_min < 0 else y_max - dimension_diff[0]
        y_min = 0 if y_min < 0 else y_min

    elif dimension_diff[1] > 0:
        half_diff = np.ceil(dimension_diff[1] / 2)
        y_min += half_diff
        y_max -= half_diff

        x_max = x_max - (dimension_diff[1] + x_min) if x_min < 0 else x_max - dimension_diff[1]
        x_min = 0 if x_min < 0 else x_min

    new_box = np.array([x_min, y_min, x_max, y_max], dtype=int)

    if verbose > 0:
        print("shrinking box. After shrinking:")
        print(new_box)

    return new_box


def expand_dims_to_match(bigger_dim: int, smaller_dim: int, smaller_dim_min: int, smaller_dim_max: int) -> Tuple[int, int]:
    """
    Expands the smaller box dimension to match the size of the bigger one. Expands equally to all sides.
    :param bigger_dim:
    :param smaller_dim:
    :param smaller_dim_min:
    :param smaller_dim_max:
    :return: smaller_dim_min, smaller_dim_max
    """
    dimension_difference = bigger_dim - smaller_dim
    expansion_size = dimension_difference / 2

    smaller_dim_min -= expansion_size
    smaller_dim_max += expansion_size

    # if dimension_difference % 2:  # if the dimension difference is odd, have to add an additional 1 to make dimensions equal.
    #     smaller_dim_max += 1

    return smaller_dim_min, smaller_dim_max


def expand_bbox_to_fit_points(bbox: Union[List, np.ndarray], landmarks: np.ndarray) -> Union[List, np.ndarray]:
    """
    Expands the box so that it would fit all the given points inside.
    :param bbox:
    :param landmarks:
    :return: box
    """
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bbox)

    lm_min_x = np.min(landmarks[:, 0])
    lm_min_y = np.min(landmarks[:, 1])
    lm_max_x = np.max(landmarks[:, 0])
    lm_max_y = np.max(landmarks[:, 1])

    if lm_min_x < x_min:
        bbox[0] = lm_min_x
    if lm_min_y < y_min:
        bbox[1] = lm_min_y

    if lm_max_x > x_max:
        bbox[2] = lm_max_x
    if lm_max_y > y_max:
        bbox[3] = lm_max_y

    return bbox


def standardise_xywh_bbox(xywh_box: np.ndarray, centers=True) -> np.ndarray:
    bb_x, bb_y, width, height = np.split(xywh_box, 4, -1)

    if centers:
        h_width, h_height = width / 2, height / 2

        x_min, x_max = bb_x - h_width, bb_x + h_width
        y_min, y_max = bb_y - h_height, bb_y + h_height
    else:
        x_min, x_max = bb_x, bb_x + width
        y_min, y_max = bb_y, bb_y + height

    standard_x_min = np.minimum(x_min, x_max)
    standard_x_max = np.maximum(x_min, x_max)
    standard_y_min = np.minimum(y_min, y_max)
    standard_y_max = np.maximum(y_min, y_max)

    standard_bbox = np.squeeze([standard_x_min, standard_y_min, standard_x_max, standard_y_max])

    return standard_bbox


def bbox_to_string(bounding_box: Union[List, np.ndarray]) -> str:
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bounding_box)
    if type(x_min) == int:
        bounding_box_string = BOUNDING_BOX_PRINT_INTEGER_TEMPLATE.format(x_min, y_min, x_max, y_max, bb_h, bb_w)
    else:
        bounding_box_string = BOUNDING_BOX_PRINT_FLOAT_TEMPLATE.format(x_min, y_min, x_max, y_max, bb_h, bb_w)

    return bounding_box_string


def get_IoU(bboxA: Union[List, np.ndarray], bboxB: Union[List, np.ndarray]) -> float:
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = abs((bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1]))
    boxBArea = abs((bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1]))

    # compute the intersection over union by taking the intersection area and
    # dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def vectorized_IoU_many_to_many(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Efficiently calculates IoU between any number of bboxes.
    :param bboxes1:
    :param bboxes2:
    :return: IoUs structured in the following way:
    [[bboxes1[0]:bboxes2[0], bboxes1[0]:bboxes2[1], ..., bboxes1[0]:bboxes1[m]],
    [bboxes1[1]:bboxes2[0], bboxes1[1]:bboxes2[1], ..., bboxes1[1]:bboxes1[m]],
    ...,
    [bboxes1[n]:bboxes2[0], bboxes1[n]:bboxes2[1], ..., bboxes1[n]:bboxes1[m]],
    """
    x1_min, y1_min, x1_max, y1_max = np.split(bboxes1, 4, axis=1)
    x2_min, y2_min, x2_max, y2_max = np.split(bboxes2, 4, axis=1)

    x_mins = np.maximum(x1_min, x2_min.T)
    y_mins = np.maximum(y1_min, y2_min.T)
    x_maxes = np.minimum(x1_max, x2_max.T)
    y_maxes = np.minimum(y1_max, y2_max.T)

    widths = np.maximum((x_maxes - x_mins + 1), 0)
    heights = np.maximum((y_maxes - y_mins + 1), 0)
    interArea = widths * heights
    bboxes1_areas = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    bboxes2_areas = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    IoUs = interArea / (bboxes1_areas + bboxes2_areas.T - interArea)

    return IoUs


def vectorized_IoU_pairs(bboxes1, bboxes2):
    x1_min, y1_min, x1_max, y1_max = np.split(bboxes1, 4, axis=1)
    x2_min, y2_min, x2_max, y2_max = np.split(bboxes2, 4, axis=1)

    x_maxes = np.squeeze(np.maximum(x1_min, x2_min))
    y_maxes = np.squeeze(np.maximum(y1_min, y2_min))
    x_mins = np.squeeze(np.minimum(x1_max, x2_max))
    y_mins = np.squeeze(np.minimum(y1_max, y2_max))

    interArea = np.expand_dims(np.maximum((x_mins - x_maxes + 1), 0) * np.maximum((y_mins - y_maxes + 1), 0), axis=-1)
    boxes_1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    boxes_2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    iou = interArea / (boxes_1_area + boxes_2_area - interArea)

    return iou


def get_bboxes_overlap(bbox_1: Union[List, np.ndarray], bbox_2: Union[List, np.ndarray], which_result: Optional[str]= 'max') -> Union[float, Tuple[float, float]]:
    xA = max(bbox_1[0], bbox_2[0])
    yA = max(bbox_1[1], bbox_2[1])
    xB = min(bbox_1[2], bbox_2[2])
    yB = min(bbox_1[3], bbox_2[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return (0., 0.)

    boxes_1_area = abs((bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1]))
    boxes_2_area = abs((bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1]))

    overlap = [interArea / boxes_1_area, interArea / boxes_2_area]

    if which_result == 'max':
        overlap = max(overlap)
    elif which_result == 'min':
        overlap = min(overlap)

    return overlap


def get_bboxes_overlap_pairs(bboxes1: Union[List, np.ndarray], bboxes2: Union[List, np.ndarray], which_result: Optional[str]='max') -> float:
    x1_min, y1_min, x1_max, y1_max = np.split(bboxes1, 4, axis=1)
    x2_min, y2_min, x2_max, y2_max = np.split(bboxes2, 4, axis=1)

    x_maxes = np.squeeze(np.maximum(x1_min, x2_min))
    y_maxes = np.squeeze(np.maximum(y1_min, y2_min))
    x_mins = np.squeeze(np.minimum(x1_max, x2_max))
    y_mins = np.squeeze(np.minimum(y1_max, y2_max))

    interArea = np.expand_dims(np.maximum((x_mins - x_maxes + 1), 0) * np.maximum((y_mins - y_maxes + 1), 0), axis=-1)

    boxes_1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    boxes_2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    overlap = np.squeeze([interArea / boxes_1_area, interArea / boxes_2_area], axis=-1).T

    if which_result == 'max':
        overlap = np.max(overlap)
    elif which_result == 'min':
        overlap = np.min(overlap)

    return overlap


def get_bboxes_overlap_many_to_many(bboxes_1, bboxes_2, which_result='max'):
    # Axis 0 contains bboxes_1 iterations
    # Axis 1 contains bboxes_2 iterations
    # Axis 2 contains [show much of overlap is within bbox_1, show much of overlap is within bbox_2]
    x1_min, y1_min, x1_max, y1_max = np.split(bboxes_1, 4, axis=-1)
    x2_min, y2_min, x2_max, y2_max = np.split(bboxes_2, 4, axis=-1)

    x_mins = np.maximum(x1_min, x2_min.T)
    y_mins = np.maximum(y1_min, y2_min.T)
    x_maxes = np.minimum(x1_max, x2_max.T)
    y_maxes = np.minimum(y1_max, y2_max.T)

    widths = np.maximum((x_maxes - x_mins), 0)
    heights = np.maximum((y_maxes - y_mins), 0)

    interArea = widths * heights
    boxes_1_area = np.array((x1_max - x1_min) * (y1_max - y1_min))
    boxes_2_area = np.array((x2_max - x2_min) * (y2_max - y2_min))

    overlap = np.zeros((len(bboxes_1), len(bboxes_2), 2))
    for i in range(len(bboxes_1)):
        for j in range(len(bboxes_2)):
            overlap[i, j] = [interArea[i, j] / boxes_1_area[i], interArea[i, j] / boxes_2_area[j]]

    if which_result == 'max':
        overlap = np.max(overlap, axis=2)
    elif which_result == 'min':
        overlap = np.min(overlap, axis=2)

    return overlap


def merge_bboxes(bboxes: Union[List, np.ndarray], inner: bool = False) -> np.ndarray:
    if inner:
        x_min = np.max(bboxes[:, 0])
        y_min = np.max(bboxes[:, 1])
        x_max = np.min(bboxes[:, 2])
        y_max = np.min(bboxes[:, 3])
    else:
        x_min = np.min(bboxes[:, 0])
        y_min = np.min(bboxes[:, 1])
        x_max = np.max(bboxes[:, 2])
        y_max = np.max(bboxes[:, 3])

    return np.array([x_min, y_min, x_max, y_max])


def change_bbox_origin_point(bounding_box, current_origin_point: Iterable[int] = [0, 0], destination_origin_point: Iterable[int] = [0, 0]) -> np.ndarray:
    current_origin_x, current_origin_y = current_origin_point[:2]
    destination_origin_x, destination_origin_y = destination_origin_point[:2]

    origin_point_x_difference = current_origin_x - destination_origin_x
    origin_point_y_difference = current_origin_y - destination_origin_y

    x_min, x_max = bounding_box[0] + origin_point_x_difference, bounding_box[2] + origin_point_x_difference
    y_min, y_max = bounding_box[1] + origin_point_y_difference, bounding_box[3] + origin_point_y_difference

    return np.array([x_min, y_min, x_max, y_max])


def parse_bbox_from_points(points: np.ndarray) -> np.ndarray:
    min_x, max_x = np.min(points[..., 0], axis=-1), np.max(points[..., 0], axis=-1)
    min_y, max_y = np.min(points[..., 1], axis=-1), np.max(points[..., 1], axis=-1)

    return np.array([min_x, min_y, max_x, max_y]).T


def get_overlapping_bbox(bbox_1, bbox_2):
    x_min = max(bbox_1[0], bbox_2[0])
    y_min = max(bbox_1[1], bbox_2[1])
    x_max = min(bbox_1[2], bbox_2[2])
    y_max = min(bbox_1[3], bbox_2[3])

    interArea = abs(max((x_max - x_min, 0)) * max((y_max - y_min), 0))

    overlapping_bbox = np.array([x_min, y_min, x_max, y_max]) if interArea > 0 else np.zeros_like(bbox_1)

    return overlapping_bbox


def bbox_coordinates_to_percent(bounding_box_coors: Union[List, np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bounding_box_coors)

    x_min, x_max = x_min / width, x_max / width
    y_min, y_max = y_min / height, y_max / height

    return np.array([x_min, y_min, x_max, y_max])


def bbox_percent_to_coordinates(bounding_box_percent: Union[List, np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max, bb_h, bb_w = parse_bbox(bounding_box_percent)

    x_min, x_max = x_min * width, x_max * width
    y_min, y_max = y_min * height, y_max * height

    return np.array([x_min, y_min, x_max, y_max])


def get_random_bbox_in_image(image_shape, bounding_box_shape, allow_off=True):
    bbox_h, bbox_w = bounding_box_shape[:2]
    image_h, image_w = image_shape[:2]

    if not allow_off and (bbox_h > image_h or bbox_w > image_w):
        return [0, 0, min(image_w, bbox_w), min(image_h, bbox_h)]

    half_bbox_h, half_bbox_w = bbox_h // 2, bbox_w // 2

    centroid_limits_x = [half_bbox_w, image_w - half_bbox_w]
    centroid_limits_y = [half_bbox_h, image_h - half_bbox_h]

    random_centroid_x = np.random.randint(centroid_limits_x[0], centroid_limits_x[1]) if centroid_limits_x[0] < centroid_limits_x[1] else image_w / 2
    random_centroid_y = np.random.randint(centroid_limits_y[0], centroid_limits_y[1]) if centroid_limits_y[0] < centroid_limits_y[1] else image_h / 2

    random_box = standardise_xywh_bbox(np.array([random_centroid_x, random_centroid_y, bbox_w, bbox_h]))

    return random_box


def jitter_bboxes(bboxes: np.ndarray, jitter_coef: float = 0.1) -> np.ndarray:
    jitter_xy = (np.random.random((len(bboxes), 2)) - 0.5) * 2 * jitter_coef

    bboxes_h, bboxes_w = get_bbox_dimensions(bboxes)

    bboxes[:, 0] += bboxes_w * jitter_xy[:, 0]
    bboxes[:, 1] += bboxes_h * jitter_xy[:, 1]
    bboxes[:, 2] += bboxes_w * jitter_xy[:, 0]
    bboxes[:, 3] += bboxes_h * jitter_xy[:, 1]

    return bboxes


def __point_distance(point_1: Union[List, np.ndarray], point_2: Union[List, np.ndarray]) -> float:
    """
    Calculates the distance between 2 points.
    :param point_1: x, y of a point
    :param point_2: x, y of a point
    :return: Distance.
    """
    return abs(np.linalg.norm(point_1 - point_2))