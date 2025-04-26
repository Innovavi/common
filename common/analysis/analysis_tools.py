from typing import List, Any, Union, Tuple, Optional, Dict, NamedTuple

import numpy as np
import pandas as pd

from common.data_manipulation.pandas_tools import rvec_columns, ryp_columns

# ANALYSIS_PATH = "/home/ignas/ml/Model_Analysis"

MEAN_LANDMARK_ERROR_NAME = ('error', 'mean_dist')
MAX_RELEVANT_LANDMARK_ERROR_NAME = ('error', 'max_relevant_dist')
BBOX_IOU_NAME = ('error', 'IoU')
MATCH_COLUMNS = pd.MultiIndex.from_product([['match'], ['best_gt_face_id', 'IoU', 'overlap', 'error_type']])
ROI_COLUMNS = pd.MultiIndex.from_product([['RoI'], ['scale', 'shift_x', 'shift_y']])

GT_PRED_RVEC_COLUMN_NAMES = pd.MultiIndex.from_product([['ground_truth', 'prediction'], rvec_columns])
GT_PRED_RYP_COLUMN_NAMES = pd.MultiIndex.from_product([['ground_truth', 'prediction'], ryp_columns])

VECTOR_ERROR_COLUMN_NAMES = pd.MultiIndex.from_product([['error'], ['Dot_Products', 'Axis_Direction_Angle', 'Angle_error', 'Len_error']])
RYP_ERROR_COLUMN_NAMES = pd.MultiIndex.from_product([['error'], ['roll_error', 'yaw_error', 'pitch_error']])

BB_SIZE_CATEGORIES = [0, 16, 32, 64, 128, 256, 512, 1024, 9999]
# ROI_BBOX_SIZE_COLUMN_NAME = ('RoI_bbox', 'size')
DATASET_COLUMN = ('metadata', 'dataset')

POSE_ANALYSIS_RANGES = np.array([-120, -90, -60, -15, 15, 60, 90, 120])
# FACE_SCALE_ANALYSIS_THRESHS = np.arange(-0.4, 0.401, 0.05)
LMS_CED_THRESHS = np.arange(0.005, 0.201, 0.005)

DEFAULT_VALUE = -1
POSE_DEFAULT_VALUE = -99999


class DatasetIndicatorTuple(NamedTuple):
    dataset_gen: str
    dataset_use: str
    dataset_set: str
    iterable_options: Optional[List[Dict[str, List[Any]]]] = None
    filter_strings: Optional[List[str]] = None


def calculate_acceptable_precision_error(error, thresholds: np.ndarray, descending: bool = False, absolute: bool = False) -> np.ndarray:
    """
    Depreciated, use calculate_CED
    """
    print("Depreciated, use calculate_CDF")

    return calculate_CDF(error, thresholds, descending, absolute)


def calculate_CDF(x, thresholds: Optional[np.ndarray] = None, descending: bool = False, absolute: bool = False) -> np.ndarray:
    """
    Calculates CDF for vector x.
    :param x: Vector ot consider.
    :param thresholds: CDF thresholds. If None, all unique steps are considered
    :param descending: Collect ascending or descending curve.
    :param absolute: Convert to absolute error.
    :return: An array containing percent of data points under each given threshold.
    """
    total_size = len(x)
    if thresholds is None:
        thresholds = np.array(sorted(pd.unique(x)))

    x_distribution = np.abs(x) if absolute else np.array(x)

    CDF = [
        (np.sum(x_distribution >= thresh) / total_size)
        if descending
        else (np.sum(x_distribution <= thresh) / total_size)
        for thresh in thresholds
    ]

    # More readable format of above list comprehension.
    # CED = []
    # for thresh in thresholds:
    #     if descending:
    #         CE_size = np.sum(error_distribution >= thresh)
    #     else:
    #         CE_size = np.sum(error_distribution <= thresh)
    #
    #     CE_percent = CE_size / total_size
    #
    #     CED.append(CE_percent)

    return np.array(CDF)


def calculate_CDF_counted(x, thresholds: Optional[np.ndarray] = None, descending: bool = False, absolute: bool = False) -> np.ndarray:
    """
    Calculates CDF for vector x.
    :param x: Vector ot consider.
    :param thresholds: CDF thresholds. If None, all unique steps are considered
    :param descending: Collect ascending or descending curve.
    :param absolute: Convert to absolute error.
    :return: An array containing numbers of data points under each given threshold.
    """
    error_sizes = []
    if thresholds is None:
        thresholds = np.array(sorted(pd.unique(x)))

    x_distribution = np.abs(x) if absolute else np.array(x)

    for thresh in thresholds:
        if descending:
            # PE_size = len(df.loc[column_series >= thresh])
            PE_size = np.sum(x_distribution >= thresh)
        else:
            # PE_size = len(df.loc[column_series <= thresh])
            PE_size = np.sum(x_distribution <= thresh)

        error_sizes.append(PE_size)

    return np.array(error_sizes)


def calculate_pose_prediction_error_from_row(pandas_row: pd.Series) -> pd.Series:
    """
    Wrapper function of 'calculate_pose_prediction_error' for pandas DataFrame rows.
    :param pandas_row:
    :return: The output of 'calculate_pose_prediction_error' but in a form of Series.
    """
    pose_gt = pandas_row['gt'].to_numpy()
    pose_pred = pandas_row['pred'].to_numpy()

    angle_error, axis_direction_error, dots, mean_vector_error = calculate_pose_prediction_error(pose_gt, pose_pred)

    return pd.Series([dots, axis_direction_error, angle_error, mean_vector_error])


def calculate_pose_prediction_error(pose_gt: np.ndarray, pose_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates pose error for a single pose gt-pred pair in the form of mean angle error, axis direction error, error dot sum and mean vector error.
    :param pose_gt: Ground truth rotation vector.
    :param pose_prediction: Predicted rotation vector.
    :return: Tuple of arrays containing: angle_error, axis_direction_error, dots, mean_vector_error
    """
    gt_norms = np.linalg.norm(pose_gt, keepdims=True)
    pred_norms = np.linalg.norm(pose_prediction, keepdims=True)

    gt_normalized = pose_gt / gt_norms
    pred_normalized = pose_prediction / pred_norms

    dots = np.sum(gt_normalized * pred_normalized)

    axis_direction_error = np.rad2deg(np.arccos(np.mean(dots)))

    angle_error = np.rad2deg(np.mean(np.abs(gt_norms - pred_norms)))

    vec_dif_len = np.linalg.norm(pose_prediction - pose_gt)

    opposite_dir = -pred_normalized
    opposite_angles = 2 * np.pi - pred_norms
    pred_np_2 = opposite_dir * opposite_angles
    vec_dif_len2 = np.linalg.norm(pred_np_2 - pose_gt)
    vector_error_length_combination = np.minimum(vec_dif_len, vec_dif_len2)
    mean_vector_error = np.mean(vector_error_length_combination)

    return angle_error, axis_direction_error, dots, mean_vector_error


def calculate_pose_prediction_error_vectorized(pose_gt: np.ndarray, pose_pred: np.ndarray, is_default_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates pose error for pose gt-pred pairs in the form of angle errors, axis direction errors, error dot sums and vector errors.
    :param pose_gt: Ground truth rotation vectors.
    :param pose_prediction: Predicted rotation vectors.
    :param is_default_mask:
    :return: Tuple of arrays containing: angle_error, axis_direction_error, dots, mean_vector_error
    """
    gt_norms = np.linalg.norm(pose_gt, axis=1, keepdims=True)
    pred_norms = np.linalg.norm(pose_pred, axis=1, keepdims=True)

    gt_normalized = np.divide(pose_gt, gt_norms, out=np.zeros_like(pose_gt), where=pose_gt != np.zeros(3))
    pred_normalized = np.divide(pose_pred, pred_norms, out=np.zeros_like(pose_pred), where=pose_pred != np.zeros(3))

    dots = np.sum(pred_normalized * gt_normalized, axis=1)
    axis_direction_error = np.rad2deg(np.arccos(dots))

    angle_error = np.squeeze(np.rad2deg(np.squeeze(np.abs(pred_norms - gt_norms))))

    vec_dif_len = np.linalg.norm(pose_pred - pose_gt, axis=1)

    opposite_dir = -pred_normalized
    opposite_angles = 2 * np.pi - pred_norms
    pred_np_2 = opposite_dir * opposite_angles
    vec_dif_len2 = np.linalg.norm(pred_np_2 - pose_gt, axis=1)

    mean_vector_error = np.minimum(vec_dif_len, vec_dif_len2)

    if is_default_mask is not None:
        angle_error = overwrite_defaults_in_array(angle_error, is_default_mask, POSE_DEFAULT_VALUE)
        axis_direction_error = overwrite_defaults_in_array(axis_direction_error, is_default_mask, POSE_DEFAULT_VALUE)
        dots = overwrite_defaults_in_array(dots, is_default_mask, POSE_DEFAULT_VALUE)
        mean_vector_error = overwrite_defaults_in_array(mean_vector_error, is_default_mask, POSE_DEFAULT_VALUE)

    return dots, axis_direction_error, angle_error, mean_vector_error


def calculate_ryp_error_from_row(pandas_row: pd.Series, gt_ryp_column_names: List[Union[str, Tuple[str, str]]],
                                 pred_ryp_column_names: List[Union[str, Tuple[str, str]]], absolute: bool = False) -> np.ndarray:
    """
    Calculates the difference of Roll Yaw Pitch angles between the columns of the given pandas row.
    :param pandas_row:
    :param gt_ryp_column_names: A list of Ground Truth column names for Roll, Yaw and Pitch.
    :param pred_ryp_column_names: A list of Predicted column names for Roll, Yaw and Pitch.
    :param absolute: Should the error value be absolute (positive) or natural.
    :return: An array containing Roll, Yaw and Pitch errors.
    """
    gt_ryp = pandas_row[gt_ryp_column_names].to_numpy()
    pred_ryp = pandas_row[pred_ryp_column_names].to_numpy()

    ryp_error = pred_ryp - gt_ryp

    ryp_error = np.absolute(ryp_error) if absolute else ryp_error

    return ryp_error


def calculate_ryp_error(gt_ryp: np.ndarray, pred_ryp: np.ndarray, is_default_mask: Optional[np.ndarray] = None, absolute: bool = False) -> np.ndarray:
    """
    Calculates the difference of Roll Yaw Pitch angles between the given arrays.
    :param gt_ryp: An array containing Ground Truth Roll, Yaw and Pitch values.
    :param pred_ryp: An array containing Predicted Roll, Yaw and Pitch values.
    :param is_default_mask:
    :param absolute: Should the error value be absolute (positive) or natural.
    :return: An array containing Roll, Yaw and Pitch errors.
    """
    ryp_error = pred_ryp - gt_ryp

    ryp_error = np.absolute(ryp_error) if absolute else ryp_error

    if is_default_mask is not None:
        ryp_error = overwrite_defaults_in_array(ryp_error, is_default_mask, POSE_DEFAULT_VALUE)

    return ryp_error


def get_class_distribution_in_df(df: pd.DataFrame, unique_classes: List[Union[str, int]], gt_column: Union[str, Tuple[str, str]] = ('ground_truth', 'class'),
                                 prediction_column: Union[str, Tuple[str, str]] = ('prediction', 'class'), verbose: int = 0) -> Tuple[Dict, Dict]:
    """
    Calculates the class distributions in GT and Pred columns of the given DataFrame.
    :param df: DataFrame with the classification columns.
    :param unique_classes: Unique classes of the columns.
    :param gt_column: The name of the column in which Ground Truth class labels are.
    :param prediction_column: The name of the column in which Predicted class labels are.
    :param verbose: If higher than 2: prints the number of samples in GT and Pred columns for each class.
    :return: A tuple of Ground Truth and Predicted class distribution dictionaries.
    """
    class_distribution_dict = {}
    pred_distribution_dict = {}

    for class_i in unique_classes:
        num_samples_gt = len(df.loc[df[gt_column] == class_i])
        num_samples_pred = len(df.loc[df[prediction_column] == class_i])

        class_distribution_dict[class_i] = num_samples_gt
        pred_distribution_dict[class_i] = num_samples_pred

        if verbose > 2:
            print('{} has {} samples and was predicted {} times'.format(class_i, num_samples_gt, num_samples_pred))

    return class_distribution_dict, pred_distribution_dict


def get_class_confusion_matrix(ground_truth_classes: np.ndarray, predicted_classes: np.ndarray, unique_classes: Optional[List[Union[str, int]]] = None,
                               remove_diagonal: bool = True, dataset_size: Optional[int] = None) -> np.ndarray:
    """
    Calculates the confusion matrix.
    :param ground_truth_classes: An array containing the Ground Truth classes.
    :param predicted_classes: An array containing the Predicted classes.
    :param unique_classes: If defined, the matrix will contain the given classes.
    If not, they will be drawn from the unique classes contained within Ground Truth class array.
    :param remove_diagonal: If True, the diagonal will be set to 0, as it contains the correct predictions.
    :param dataset_size: If defined, the confusion matrix will be divided by this number to obtain the percentile values of the confusion matrix.
    :return: Confusion matrix.
    """
    unique_classes = unique_classes if unique_classes is not None else np.unique(ground_truth_classes)
    class_confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

    for i in unique_classes:
        for j in unique_classes:
            if remove_diagonal and i == j:
                class_count = 0
            else:
                class_count = len(np.where((ground_truth_classes == i) & (predicted_classes == j))[0])

            class_confusion_matrix[i, j] = class_count

    class_confusion_matrix = class_confusion_matrix / dataset_size if dataset_size is not None else class_confusion_matrix

    return class_confusion_matrix


def get_class_confusion_matrices_over_a_column(df: pd.DataFrame, column_to_separate_over: Union[str, Tuple[str, str]],
                                               gt_class_column_name: Union[str, Tuple[str, str]], pred_class_column_name: Union[str, Tuple[str, str]],
                                               unique_classes: Optional[List[Union[str, int]]] = None) -> Tuple[List[np.ndarray], Any]:
    """
    Calculates multiple confusion matrices, each for a category found in separation column.
    :param df: DataFrame with the classification columns.
    :param column_to_separate_over: Categorical column.
    :param gt_class_column_name: The name of the column in which Ground Truth class labels are.
    :param pred_class_column_name: The name of the column in which Predicted class labels are.
    :param unique_classes: If defined, the matrix will contain the given classes.
    If not, they will be drawn from the unique classes contained within Ground Truth class array.
    :return: A list of confusion matrices and the categories that each confusion matrix represents.
    """
    unique_instances = sorted(pd.unique(df[column_to_separate_over].astype(str)))
    unique_classes = list(set(
        pd.unique(df[gt_class_column_name]).tolist() + pd.unique(df[pred_class_column_name]).tolist())) if unique_classes is None else unique_classes
    column_confusion_matrices = []

    for unique_instance in unique_instances:
        partial_df = df.loc[df[column_to_separate_over] == unique_instance]

        ground_truth_classes = partial_df[gt_class_column_name].to_numpy(dtype=np.int32)
        predicted_classes = partial_df[pred_class_column_name].to_numpy(dtype=np.int32)

        class_confusion_matrix = get_class_confusion_matrix(ground_truth_classes, predicted_classes, unique_classes=unique_classes, remove_diagonal=False)

        column_confusion_matrices.append(class_confusion_matrix)

    return column_confusion_matrices, unique_instances


def confusion_matrix_to_nested_dict(class_confusion_matrix: np.ndarray, class_names: Optional[List[str]] = None, add_zeros: bool = True) -> Dict:
    """
    Converts the confusion matrix to a nested dictionary with keys to each class.
    :param class_confusion_matrix: Confusion matrix to convert.
    :param class_names: If defined, the dictionary keys will be drawn from this list. If not, the keys will be integers beginning with 0.
    :param add_zeros: If True, the dictionary will contain 0 values, if the matrix does.
    If False, instances with the value of 0 will not be included in the dictionary.
    :return: Nested dictionary converted from the given confusion matrix.
    """
    dict_keys = class_names if class_names is not None else np.arange(len(class_confusion_matrix))

    nested_dict = {dict_keys[i]: {dict_keys[j]: class_confusion_matrix[i, j] for j in range(len(class_confusion_matrix)) if
                                  not (class_confusion_matrix[i, j] == 0 and not add_zeros)}
                   for i in range(len(class_confusion_matrix))}

    nested_dict = nested_dict if add_zeros else {key: inside_dict for key, inside_dict in nested_dict.items() if len(inside_dict) > 0}

    return nested_dict


def generate_dataset_error_count_nested_dict(df: pd.DataFrame, dataset_column_name: Union[str, Tuple[str, str]],
                                             gt_class_column_name: Union[str, Tuple[str, str]], pred_class_column_name: Union[str, Tuple[str, str]],
                                             normalize: bool = False, add_zeros: bool = True) -> Dict:
    """

    :param df:
    :param dataset_column_name:
    :param gt_class_column_name:
    :param pred_class_column_name:
    :param normalize:
    :param add_zeros:
    :return:
    """
    unique_datasets = sorted(pd.unique(df[dataset_column_name]))

    dataset_incorrect_prediction_counts = {}
    for unique_dataset in unique_datasets:
        dataset_df = df.loc[df[dataset_column_name] == unique_dataset]

        ground_truth_classes = dataset_df[gt_class_column_name].to_numpy(dtype=np.int32)
        predicted_classes = dataset_df[pred_class_column_name].to_numpy(dtype=np.int32)

        dataset_size_to_normalize_with = len(dataset_df) if normalize else None
        class_confusion_matrix = get_class_confusion_matrix(ground_truth_classes, predicted_classes, dataset_size=dataset_size_to_normalize_with)

        confusion_matrix_nested_dict = confusion_matrix_to_nested_dict(class_confusion_matrix, add_zeros=add_zeros)

        if len(confusion_matrix_nested_dict) == 0 and not add_zeros:
            continue

        dataset_incorrect_prediction_counts[unique_dataset] = confusion_matrix_nested_dict

    return dataset_incorrect_prediction_counts


def analyse_DET_on_df(df: pd.Series, gt_class_column_name: Union[Tuple[str, str], str], score_column_name: Union[Tuple[str, str], str],
                      threshold_tic: float = 0.01, default_value: int = DEFAULT_VALUE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    false_positive_rates = []
    false_negative_rates = []
    thresholds = []

    non_default_df = df.loc[df[gt_class_column_name] > default_value]

    negative_df = non_default_df.loc[non_default_df[gt_class_column_name] == 0]
    positive_df = non_default_df.loc[non_default_df[gt_class_column_name] == 1]

    if len(negative_df) == 0 or len(positive_df) == 0:
        print("len(negative_df) == 0 or len(positive_df) == 0")
        print("len(negative_df)", len(negative_df))
        print("len(positive_df)", len(positive_df))

    else:
        negative_scores = negative_df[score_column_name].to_numpy()
        positive_scores = positive_df[score_column_name].to_numpy()

        false_positive_rates, false_negative_rates, thresholds = calculate_DET(negative_scores, positive_scores, threshold_tic)

    return false_positive_rates, false_negative_rates, thresholds


def calculate_DET(negative_scores, positive_scores, threshold_tic):
    false_positive_rates = []
    false_negative_rates = []

    thresholds = np.arange(np.min(negative_scores), np.max(positive_scores) + threshold_tic, threshold_tic)

    for threshold in thresholds:
        false_positive_ratio = np.sum(negative_scores >= threshold) / len(negative_scores)
        false_negative_ratio = np.sum(positive_scores < threshold) / len(positive_scores)

        false_positive_rates.append(false_positive_ratio)
        false_negative_rates.append(false_negative_ratio)

    return np.array(false_positive_rates), np.array(false_negative_rates), thresholds


def calculate_EER(FPRs, FNRs, threshs):
    """
    Calculates Equal Error Rate.
    """
    min_distance = 99999
    min_index = -1

    for i, (fpr, fnr) in enumerate(zip(FPRs, FNRs)):
        distance = abs(fpr - fnr)

        if min_distance > distance:
            min_distance = distance
            min_index = i

    min_fpr = FPRs[min_index]
    min_fnr = FNRs[min_index]
    min_thresh = threshs[min_index]

    return min_distance, min_fpr, min_fnr, min_thresh


def overwrite_defaults_in_array(some_array, is_default_mask=None, default_value=DEFAULT_VALUE):
    default_array_value = np.ones(some_array.shape[1:]) * default_value
    is_default_mask = np.equal(np.max(some_array, axis=1), default_value) if is_default_mask is None else is_default_mask

    some_array[is_default_mask] = default_array_value

    return some_array