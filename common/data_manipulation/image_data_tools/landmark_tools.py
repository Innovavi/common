from typing import List, Union, Optional, Iterable, Tuple, Dict

import numpy as np

from common.data_manipulation.checks import check_if_point_is_inside_bbox

LANDMARK_PARTS_NAMES = ['jaw', 'left_eyebrow', 'right_eyebrow', 'nose_top', 'nose_bot', 'left_eye', 'right_eye', 'outter_mouth', 'inner_mouth', 'eye_centers']

standard_68_landmark_parts_indice_dict = {
    LANDMARK_PARTS_NAMES[0]: np.arange(0, 17),
    LANDMARK_PARTS_NAMES[1]: np.arange(17, 22), LANDMARK_PARTS_NAMES[2]: np.arange(22, 27),
    LANDMARK_PARTS_NAMES[3]: np.arange(27, 31), LANDMARK_PARTS_NAMES[4]: np.arange(31, 36),
    LANDMARK_PARTS_NAMES[5]: np.arange(36, 42), LANDMARK_PARTS_NAMES[6]: np.arange(42, 48),
    LANDMARK_PARTS_NAMES[7]: np.arange(48, 60), LANDMARK_PARTS_NAMES[8]: np.arange(60, 68),
}

standard_98_landmark_parts_indice_dict = {
    LANDMARK_PARTS_NAMES[0]: np.arange(0, 33),
    LANDMARK_PARTS_NAMES[1]: np.arange(33, 42), LANDMARK_PARTS_NAMES[2]: np.arange(42, 51),
    LANDMARK_PARTS_NAMES[3]: np.arange(51, 55), LANDMARK_PARTS_NAMES[4]: np.arange(55, 60),
    LANDMARK_PARTS_NAMES[5]: np.arange(60, 68), LANDMARK_PARTS_NAMES[6]: np.arange(68, 76),
    LANDMARK_PARTS_NAMES[7]: np.arange(76, 88), LANDMARK_PARTS_NAMES[8]: np.arange(88, 96),
    LANDMARK_PARTS_NAMES[9]: np.arange(96, 98),  # 98 has eye centers
}


def landmarks_pts_to_percents(landmarks: Iterable[Iterable[Union[int, float]]], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Converts landmark coordinates to percents based on the image.
    :param landmarks: Landmarks to convert.
    :param image_shape: Image shape in which the landmarks are.
    :return: Converted percent landmarks.
    """
    percent_landmarks = []

    for landmark in landmarks:
        if len(landmark) != 2:
            print("landmark format is bad. Has to be (?, 2), but is ({}, {})".format(len(landmarks), len(landmark)))
            return None

        percent_landmarks.append(landmark_pt_to_percent(landmark, image_shape))

    return np.array(percent_landmarks)


def landmark_pt_to_percent(landmark: Iterable[Union[int, float]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts a single landmark's point coordinates to percents based on the image.
    :param landmark: Landmark to convert.
    :param image_shape: Image shape in which the landmark is.
    :return: Converted percent landmark.
    """
    h, w = image_shape[:2]
    x, y = landmark

    x_percent = x / w
    y_percent = y / h

    return np.array([x_percent, y_percent])


def percents_to_coors(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts landmarks percents to coordinates. Reverse of landmarks_pts_to_percents.
    :param landmarks: Percent landmarks to be converted.
    :param image_shape: Image shape in which the landmarks are.
    :return: Converted coordinate landmarks.
    """
    coor_landmarks = [percent_to_coor(landmark, image_shape) for landmark in landmarks]

    return np.array(coor_landmarks)


def percent_to_coor(landmark: Iterable[Union[int, float]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts landmark percents to coordinates. Reverse of landmark_pt_to_percent.
    :param landmarks: Percent landmark to be converted.
    :param image_shape: Image shape in which the landmark is.
    :return: Converted coordinate landmark.
    """
    h, w = image_shape[:2]
    x, y = landmark

    x_coor = int(x * w)
    y_coor = int(y * h)

    return np.array([x_coor, y_coor], dtype=np.int32)


def change_landmarks_origin_point(landmarks: Union[List, np.ndarray], current_origin_point: Iterable[int] = [0, 0],
                                  destination_origin_point: Iterable[int] = [0, 0]) -> np.ndarray:
    """
    Changes the axis origin point of the given landmarks. This is used when landmark coordinate systems are changed.
    For example when Face Verificator's landmarks have to be drawn on the original image instead of the crop.
    :param landmarks: Points under the current origin point system.
    :param current_origin_point: Origin point under the which given landmarks are under. Can be left default if the current system is global.
    For example when converting from the original image origin point.
    :param destination_origin_point: Origin point to which to convert the landmark coordinates. Can be left default if the destination system is global.
    For example when converting to the original image origin point.
    :return: Landmarks under the destination origin point system.
    """

    current_origin_x, current_origin_y = current_origin_point[:2]
    destination_origin_x, destination_origin_y = destination_origin_point[:2]

    origin_point_x_difference = current_origin_x - destination_origin_x
    origin_point_y_difference = current_origin_y - destination_origin_y

    new_landmarks_x = landmarks[:, 0] + origin_point_x_difference
    new_landmarks_y = landmarks[:, 1] + origin_point_y_difference

    new_landmarks = np.stack([new_landmarks_x, new_landmarks_y], axis=1)

    return new_landmarks


def convert_flat_to_standard(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert flat landmark array to 2 dimensional one, where first dimension describes different landmarks and
    the second - landmark x and y coordinates/percents.
    The function first checks whether the given array is actually flat, so the function can be used to simply make sure that landmarks are standard.
    :param landmarks: Landmarks to convert.
    :return: 2D landmark array.
    """
    if len(landmarks.shape) == 1:
        reshaped_landmarks = [[landmarks[i], landmarks[i+1]] for i in range(0, len(landmarks), 2)]
    else:
        reshaped_landmarks = landmarks
        
    return np.array(reshaped_landmarks)


def flatten_landmarks(landmarks: List[List[Union[int, float]]]) -> np.ndarray:
    """
    Flattens the landmark list/array.
    :param landmarks: Landmark list/array to flatten.
    :return: Flat landmark array.
    """
    landmarks = np.array(landmarks)
    flat_mandmarks = landmarks.flatten()
    
    return flat_mandmarks


def find_landmarks_in_bounding_box(landmarks: np.ndarray, bounding_box: np.ndarray) -> List[int]:
    """
    Finds landmarks that are inside the given bounding box and returns all their indice.
    :param landmarks: Landmarks to search through.
    :param bounding_box: Bounding box to locate landmarks in.
    :return: Indice of landmarks that are within the given bounding box.
    """
    indice = [i for i, landmark in enumerate(landmarks) if check_if_point_is_inside_bbox(landmark, bounding_box)]

    return indice


def calculate_middle_landmark_from_symmetry(landmarks: np.ndarray, symetry_tuples: List[Tuple[int, int]], confidences: np.ndarray = None) -> Tuple[np.ndarray, float]:
    included_lms_x = []
    included_lms_y = []
    included_lms_confs = []

    for symmetry_tuple in symetry_tuples:
        lm_1_x, lm_1_y = landmarks[symmetry_tuple[0]]
        lm_2_x, lm_2_y = landmarks[symmetry_tuple[1]]

        if np.isnan(lm_1_x) or np.isnan(lm_2_x):
            continue

        included_lms_x.extend([lm_1_x, lm_2_x])
        included_lms_y.extend([lm_1_y, lm_2_y])

        if confidences is not None:
            included_lms_confs.append(confidences[symmetry_tuple[0]])
            included_lms_confs.append(confidences[symmetry_tuple[1]])

    if len(included_lms_x) == 0 or len(included_lms_y) == 0:
        mean_x, mean_y, mean_conf = -1., -1., -1.
    else:
        mean_x = np.mean(included_lms_x)
        mean_y = np.mean(included_lms_y)
        mean_conf = np.mean(included_lms_confs, dtype=float) if confidences is not None else -1

    return np.array([mean_x, mean_y]), mean_conf


def get_landmark_parts(landmarks: np.ndarray) -> Dict[str, np.ndarray]:
    assert len(landmarks) == 68 or len(landmarks) == 98, "Landmark array is not of standard type (68 or 98). Its length is {}".format(len(landmarks))
    landmark_parts_indice_dict = standard_68_landmark_parts_indice_dict if len(landmarks) == 68 else standard_98_landmark_parts_indice_dict

    landmark_parts_dict = {lm_part_name: landmarks[lms_part_indice] for lm_part_name, lms_part_indice in landmark_parts_indice_dict.items()}

    return landmark_parts_dict


def convert_98_to_68_lms(landmarks):
    standard_lms = np.concatenate([
        landmarks[:33:2],
        landmarks[33:38], landmarks[42:47],
        landmarks[51:60],
        landmarks[60:62], landmarks[63:66], landmarks[67:68],
        landmarks[68:70], landmarks[71:74], landmarks[75:76],
        landmarks[76:96]
    ])

    return standard_lms
