from typing import Optional, List, Tuple, Any, Dict

import numpy as np

from common.data_manipulation.dictionary_tools import convert_dict_lists_to_list_of_dicts
from common.data_manipulation.image_data_tools.landmark_tools import change_landmarks_origin_point
from common.data_manipulation.image_data_tools.pose_tools import rpy_to_rotation_matrix
from common.image_tools.cropping import crop_box
from common.visualizations.image_visualizations import draw_landmarks, draw_pose, draw_bounding_box, draw_confidences


def visualize_image_data(image: np.ndarray, bboxes: Optional[np.ndarray] = None, landmarks: Optional[np.ndarray] = None,
                         poses: Optional[np.ndarray] = None, confidences_dict: Optional[Dict[Any, List[Any]]] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """

    :param image:
    :param bboxes:
    :param landmarks:
    :param poses:
    :param confidences_dict:
    :return: drawn_image, crops
    """
    drawn_image = image.copy()
    crops = []
    max_lenght = max([len(annot) for annot in [bboxes, landmarks, poses] if annot is not None])

    confidences_list = convert_dict_lists_to_list_of_dicts(confidences_dict) if confidences_dict is not None else None

    for i in range(max_lenght):
        drawn_image = draw_landmarks(drawn_image, landmarks[i], thickness=None) if landmarks is not None else drawn_image

        if bboxes is not None:
            bbox = bboxes[i]
            drawn_image = draw_bounding_box(drawn_image, bbox, thickness=None)

            drawn_image = draw_confidences(drawn_image, confidences_list[i], bbox[:2]) if confidences_list is not None else drawn_image

            cropped_image = crop_box(image, bbox[:4])

            aligned_landmarks = change_landmarks_origin_point(landmarks[i], destination_origin_point=bbox[:2]) if landmarks is not None else None
            cropped_image = draw_landmarks(cropped_image, aligned_landmarks, thickness=None) if landmarks is not None else cropped_image
            cropped_image = draw_pose(cropped_image, rpy_to_rotation_matrix(poses[i]), line_length=None) if poses is not None else cropped_image

            crops.append(cropped_image)

    return drawn_image, crops

# def visualize_verification(image, decoded_predictions, viz_size=256, model_input_size = 128):
#     pred_bbox, pred_rvec, pred_lms, pred_face_cls, pred_mask_cls, pred_face_quality = decoded_predictions
#     pred_rot_mat = rotation_vector_to_rotation_matrix(pred_rvec / 2)
#
#     _, pred_bbox, pred_lms = resize_image_with_data(np.zeros((model_input_size, model_input_size)), (viz_size, viz_size), pred_bbox, pred_lms)
#
#     drawn_image = image
#     gt_bbox *= image_shape
#     drawn_image = draw_bounding_box(drawn_image, gt_bbox, thickness=1)
#
#     gt_landmarks *= image_shape
#     drawn_image = draw_landmarks(drawn_image, gt_landmarks, add_default_lm_names=True, thickness=2, text_size=TEXT_PARAMS)
#
#     gt_rot_mat = rotation_vector_to_rotation_matrix(gt_rvec)
#     drawn_image = draw_pose(drawn_image, gt_rot_mat, line_length=50, lines_thicknesses=[2, 2, 2])
#
#     drawn_image = draw_confidences(
#         drawn_image, {'q': gt_face_quality},
#         top_left_point=[7, 30], color=(0, 128, 255), font_scale=TEXT_PARAMS[0], thickness=TEXT_PARAMS[1]
#     )
#
#     drawn_image = draw_confidences(
#         drawn_image, {'f': gt_face_class, 'm': gt_mask_class},
#         top_left_point=[7, 15], color=(0, 128, 255), font_scale=TEXT_PARAMS[0], thickness=TEXT_PARAMS[1]
#     )
#
#     return drawn_image