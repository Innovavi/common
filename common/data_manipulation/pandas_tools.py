import os
from datetime import datetime
from enum import Enum
from typing import Tuple, List, Union, Optional, Dict

import numpy as np
import pandas as pd

from common.data_manipulation.image_data_tools.bounding_box_tools import get_bbox_dimensions

generic_pandas_columns = ['image_id', 'face_id', 'image_name', 'set_name']
bbox_columns = ['x_min', 'y_min', 'x_max', 'y_max']
irrectangle_columns = ['angle', 'top_left_x',  'top_left_y', 'top_right_x', 'top_right_y', 'bot_left_x',  'bot_left_y', 'bot_right_x', 'bot_right_y']
rvec_columns = ['angle_x', 'angle_y', 'angle_z']
ryp_columns = ['roll', 'yaw', 'pitch']


class PoseType(Enum):
    RYP = ryp_columns
    XYZ = rvec_columns


def load_df(csv_fullname: str, multiindex: bool = False, compression: str = "infer", index_col = 0, verbose: int = 1) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a CSV file.

    :param csv_fullname: The full path or name of the CSV file to be loaded. .csv extention is not necessary.
    :param multiindex: If True, the DataFrame will have a MultiIndex.
    :param compression: Compression algorithm to use during reading.
    :param verbose: Verbosity level. If 0, no messages are printed. If 1, basic information is printed.
    :return: dataframe: The loaded DataFrame.
    """
    csv_fullname = csv_fullname if csv_fullname.endswith('.csv') else csv_fullname + '.csv'
    dataframe = pd.read_csv(csv_fullname, index_col=index_col, header=[0, 1], compression=compression) \
        if multiindex \
        else pd.read_csv(csv_fullname, index_col=index_col, compression=compression)

    if verbose:
        print('Loaded a Dataframe with shape {} × {} from {}'.format(len(dataframe), len(dataframe.columns), csv_fullname))
        print("at ", datetime.now())

    return dataframe


def save_dataframe(dataframe: pd.DataFrame, csv_fullname: str, verbose: int = 1, compression: Optional[str] = None) -> None:
    """
    TODO: implement xz compression
    Saves Pandas DataFrame to specified location, prints it and prints the datetime when it was saved.
    :param dataframe: DataFrame to save.
    :param csv_fullname: Full destination path name.
    :param verbose: Default 1 prints.
    :param compression: Compression to use.
    """
    if not csv_fullname.endswith('.csv'):
        csv_fullname += '.csv'

    dataframe.to_csv(csv_fullname, compression=compression)

    if verbose:
        print('Saved a Dataframe with shape {} × {} to {}'.format(len(dataframe), len(dataframe.columns), csv_fullname))
        print("at ", datetime.now())


def parse_generic_pandas_row(pandas_row: Union[pd.Series, pd.DataFrame], multiindex: bool = False) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    Parses any pandas row into a tuple containing [image_id, face_id, image_name, bounding_box].
    :param pandas_row: Row to be parsed.
    :return: image_id, face_id, image_name, bounding_box.
    """
    generic_panda = pandas_row['metadata'] if multiindex else pandas_row

    image_id = generic_panda.get(generic_pandas_columns[0], None)
    face_id = generic_panda.get(generic_pandas_columns[1], None)
    image_name = generic_panda.get(generic_pandas_columns[2], None)
    image_set = generic_panda.get(generic_pandas_columns[3], '')

    image_id = np.array(image_id, dtype=int) if image_id is not None else None
    face_id = np.array(face_id, dtype=int) if face_id is not None else None

    return image_id, face_id, image_name, image_set


def get_bbox_from_pandas(pandas_row: pd.Series) -> np.ndarray:
    """
    Parses Face Verification pandas row for bounding box info.
    :param pandas_row: Face Verification row.
    :param bbox_multicolumn: Bounding Box columns. Leave default for [x_min, y_min, x_max, y_max].
    :return: An array containing bbox info in the following format: x_min, y_min, x_max, y_max.
    """
    bbox = pandas_row.get(bbox_columns, None)
    bbox = np.array(bbox, dtype=np.float32) if bbox is not None else None

    return bbox


def get_irrectangle_from_pandas(pandas_row: pd.Series) -> Tuple[float, np.ndarray]:
    irrectangle_data = pandas_row[irrectangle_columns].values
    angle, irrectangle = irrectangle_data[:1], irrectangle_data[1:]

    irrectangle = np.reshape(irrectangle, (4, 2))

    return angle, irrectangle


def parse_bbox_size_from_pandas(pandas_row):
    bbox = get_bbox_from_pandas(pandas_row)
    bb_h, bb_w = get_bbox_dimensions(bbox)
    bbox_size = np.mean([bb_h, bb_w], axis=0)

    return bbox_size


def get_landmarks_from_pandas(panda: Union[pd.DataFrame, pd.Series], from_lm: int = 0, to_lm: int = 68, with_conf: bool = False, filter_default: bool = False) -> np.ndarray:
    """
    Collects all landmarks from the row and puts the into an array with the following shape: (?, 2)
    :param panda:
    :param basic_landmarks:
    :return:
    """
    landmarks = np.array([
        panda.get(list(coor_names), [None] * len(coor_names))
        for coor_names in generate_landmark_names(from_lm, to_lm, with_conf)
    ], dtype=np.float32)

    if len(landmarks.shape) > 2:
        landmarks = landmarks.transpose((1, 0, 2))

    if filter_default:
        landmarks = landmarks[(landmarks[..., 0] != -1) | (landmarks[..., 1] != -1)]

    return landmarks


def get_pose_from_pandas(pandas_row: Union[pd.Series, pd.DataFrame], pose_type: PoseType = PoseType.RYP) -> np.ndarray:
    """
    Parses pose from pandas row or DataFrame.
    :param pandas_row:
    :param pose_type: One of PoseType's enum values.
    :return:
    """
    pose = pandas_row.get(pose_type.value, None)
    pose = pose.to_numpy(np.float32) if pose is not None else None

    return pose


def generate_landmark_names(from_lm: int = 0, to_lm: int = 68, generate_conf: bool = False) -> Tuple[str, str]:
    """
    Generator for landmark names. Formats like this: "lm_x{:02d}", "lm_y{:02d}"
    :param basic_landmarks:
    :param from_lm: Starting landmark index.
    :param to_lm: Ending landmark index.
    """
    for i in range(from_lm, to_lm):
        lm_x = "lm_{:02d}_x".format(i)
        lm_y = "lm_{:02d}_y".format(i)
        lm_score = "lm_{:02d}_score".format(i)

        landmark_names = (lm_x, lm_y, lm_score) if generate_conf else (lm_x, lm_y)

        yield landmark_names


def generate_3D_landmark_names(from_lm: int = 0, to_lm: int = 68, generate_conf: bool = False) -> Tuple[str, str, str]:
    """
    Generator for landmark names. Formats like this: "lm_x{:02d}", "lm_y{:02d}"
    :param basic_landmarks:
    :param from_lm: Starting landmark index.
    :param to_lm: Ending landmark index.
    """
    for i in range(from_lm, to_lm):
        lm_x = "lm_{:02d}_x".format(i)
        lm_y = "lm_{:02d}_y".format(i)
        lm_z = "lm_{:02d}_z".format(i)
        lm_score = "lm_{:02d}_score".format(i)

        landmark_names = (lm_x, lm_y, lm_z, lm_score) if generate_conf else (lm_x, lm_y, lm_z)

        yield landmark_names


def get_landmark_count(df: pd.DataFrame):
    landmark_count = 0
    df_columns = df.columns.tolist()

    for lm_name_x, lm_name_y in generate_landmark_names(False, to_lm=999):
        if lm_name_x in df_columns:
            landmark_count += 1
        else:
            break

    return landmark_count


def add_face_id(df: pd.DataFrame, face_id_start: int = 0, assign_to_first_col=True) -> pd.DataFrame:
    """
    Adds face id as a column to the given dataframe.
    :param df: Dataframe to add the face id column to.
    :param face_id_start: Face id start index.
    :param assign_to_first_col:
    :return: Dataframe with face id column.
    """
    if 'metadata' in df:
        face_id_column_name = ('metadata', 'face_id')
    else:
        face_id_column_name = 'face_id'

    if face_id_start is None:
        face_id_start = np.power(10, int(np.log10(len(df))))

    face_ids = np.arange(face_id_start, face_id_start + len(df))

    if face_id_column_name in df:
        df[face_id_column_name] = face_ids

    else:
        insertion_idx = 0 if assign_to_first_col else len(df.columns)
        df.insert(loc=insertion_idx, column=face_id_column_name, value=face_ids)

    return df


def add_image_id(df: pd.DataFrame, image_id_start: int = None, assign_to_first_col=True) -> pd.DataFrame:
    """
    Adds face id as a column to the given dataframe.
    :param df: Dataframe to add the face id column to.
    :param image_id_start: Face id start index.
    :param assign_to_first_col:
    :return: Dataframe with face id column.
    """
    if 'metadata' in df:
        image_name_column_name = ('metadata', 'image_name')
        image_id_column_name = ('metadata', 'image_id')
    else:
        image_name_column_name = 'image_name'
        image_id_column_name = 'image_id'

    if image_id_start is None:
        image_id_start = np.power(10, int(np.log10(len(pd.unique(df[image_name_column_name])))))

    image_ids = df[image_name_column_name].astype('category').cat.codes.astype(np.int32) + image_id_start

    if image_id_column_name in df:
        df[image_id_column_name] = image_ids

    else:
        insertion_idx = 0 if assign_to_first_col else len(df.columns)
        df.insert(loc=insertion_idx, column=image_id_column_name, value=image_ids)

    return df


def separate_df_based_on_column_range(df: pd.DataFrame, column_name: Union[str, Tuple[str, str]], min_value: Union[float, int], max_value: Union[float, int], do_absolute=False):
    separated_df = df.loc[(df[column_name].abs() > min_value) & (df[column_name].abs() <= max_value)] \
        if do_absolute \
        else df.loc[(df[column_name] > min_value) & (df[column_name] <= max_value)]

    return separated_df


def parse_digibody_pandas_row(pandas_row: Union[pd.Series, pd.DataFrame]):
    metadata = pandas_row['metadata']

    image_id = metadata.get('image_id', -1)
    #face_id = metadata.get('face_id', -1)
    #dataset = metadata.get('dataset', '')
    #et_name = metadata.get('set', '')
    image_name = metadata.get('image_name', '')
    batch_id = metadata.get('image_batch', -1)
    camera_type = metadata.get('camera_type', '')

    #dome_rotation = pandas_row['dome_rotation'].values
    measures = pandas_row['measures'].values
    parameters = pandas_row['parameters'].values
    bbox = get_bbox_from_pandas(pandas_row['bbox'])
    #coefs = pandas_row['coefs'].values

    full_landmarks = pandas_row['landmarks'].to_numpy().reshape(-1,2)#get_landmarks_from_pandas(pandas_row['landmarks'], to_lm=2502, filter_default=False)
    return image_id, image_name, camera_type, measures, parameters, bbox, full_landmarks


def parse_digibody_pandas_row(pandas_row: Union[pd.Series, pd.DataFrame]):
    metadata = pandas_row['metadata']

    image_id = metadata.get('image_id', -1)
    #face_id = metadata.get('face_id', -1)
    #dataset = metadata.get('dataset', '')
    #et_name = metadata.get('set', '')
    image_name = metadata.get('image_name', '')
    batch_id = metadata.get('image_batch', -1)
    camera_type = metadata.get('camera_type', '')

    #dome_rotation = pandas_row['dome_rotation'].values
    measures = pandas_row['measures'].values
    parameters = pandas_row['parameters'].values
    bbox = get_bbox_from_pandas(pandas_row['bbox'])
    #coefs = pandas_row['coefs'].values

    full_landmarks = pandas_row['landmarks'].to_numpy().reshape(-1,2)#get_landmarks_from_pandas(pandas_row['landmarks'], to_lm=2502, filter_default=False)
    return image_id, image_name, camera_type, measures, parameters, bbox, full_landmarks
