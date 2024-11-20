from itertools import product
from typing import Tuple, List, Optional, Iterable, Union, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt

from common.data_manipulation.image_data_tools.landmark_tools import convert_flat_to_standard
from common.visualizations.viz_utils import DEFAULT_PLOT_SIZE, __show_save_logic


def show_image(image: np.ndarray, axis: bool = False, **kwargs):
    """
    Displays the image.
    :param image: Image to display.
    :param axis: Do show axis.
    """
    assert image is not None, "That's not an image"

    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)

    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray', interpolation='none')
    else:
        plt.imshow(image, interpolation='none')

    if not axis:
        plt.axis("off")

    title = kwargs.get('title', None) or "{}x{}".format(image.shape[0], image.shape[1])
    plt.title(title)

    __show_save_logic(figure, **kwargs)


def show_images(image_list: List[np.ndarray], columns: int = 3, axis: bool = True, **kwargs):
    """
    Subplots images. Will plot all images in given list using specified column number. When titles is not None, also adds titles to all images.
    :param image_list: Images to display.
    :param columns: How many images should fit in a row.
    :param titles: Titles of the images. Must be the same length as the image list.
    :param fig_size_per_row: Size of the displayed figure.
    """
    titles = kwargs.get('titles', [])

    assert image_list is not None, "Image list is None"
    assert len(titles) == 0 or len(image_list) == len(titles), "Image list and titles are not the same size"

    image_list_length = len(image_list)

    if image_list_length == 0:
        print("No Images in List")
        return

    if image_list_length == 1:
        kwargs['title'] = titles[0] if titles else None
        return show_image(image_list[0], axis, **kwargs)

    elif image_list_length < columns:
        columns = image_list_length

    n_rows = len(image_list) // columns

    if kwargs.get('fig_size', None):
        figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    else:
        fig_size_per_row = kwargs.get('fig_size_per_row', DEFAULT_PLOT_SIZE)
        fig_size = (fig_size_per_row[0], fig_size_per_row[1] * n_rows)
        figure, ax = plt.subplots(figsize=fig_size, clear=True)

    for i in range(len(image_list)):
        plt.subplot(n_rows + 1, columns, i + 1)
        plt.imshow(image_list[i])

        if titles: plt.title(titles[i])
        if not axis: plt.axis('off')

    __show_save_logic(figure, **kwargs)


def draw_landmarks(image: np.ndarray, landmarks: Iterable[Union[int, float]], lm_names: Optional[List[str]] = None,
                   confidences: Optional[Union[List[float], np.ndarray]] = None, radius: Optional[int] = 1, thickness: Optional[int] = 1,
                   color: np.ndarray = np.array([255, 0, 0]), text_size: Tuple[float, int] = (0.5, 1)) -> np.ndarray:
    """
    Draws circles where landmark points are on the given image.
    :param image:
    :param landmarks:
    :param lm_names:
    :param confidences:
    :param radius: Radius of the circles.
    :param thickness: Thickness of the circles.
    :param color: The color of the circles.
    :param text_size: landmark names size parameters. First is text scale and the second is text thickness.
    :return: visualization
    """
    landmarks = np.array(landmarks)
    landmarks = landmarks[np.where(np.any(landmarks!=None, axis=-1))]
    visualization = image.copy()
    annotation_strings = None

    landmarks = convert_flat_to_standard(landmarks)

    if 0 <= np.max(landmarks) <= 1:
        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]

    # if add_default_lm_names:
    #     assert len(landmarks) == len(BASIC_LANDMARK_SHORT_NAMES), \
    #         "the lenght of basic landmark names and landmarks ({}) do not match".format(len(BASIC_LANDMARK_SHORT_NAMES), len(landmarks))
    #     annotation_strings = BASIC_LANDMARK_SHORT_NAMES

    if lm_names is not None:
        assert len(lm_names) == len(landmarks), \
            "the lenght of lm_names ({}) and landmarks ({}) do not match".format(len(lm_names), len(landmarks))
        annotation_strings = lm_names

    if confidences is not None:
        assert len(confidences) == len(landmarks), \
            "the lenght of confidences ({}) and landmarks ({}) do not match".format(len(confidences), len(landmarks))

    if radius is None or thickness is None:
        thickness = max(1, int((min(image.shape[:2]) + 10) / 75))
        radius = max(0, int(thickness / 2.5))

    elif type(radius) == float or type(thickness) == float:
        coef = radius if radius is not None else thickness

        thickness = max(1, int((min(image.shape[:2]) + 10) * coef / 75))
        radius = max(0, int(thickness * coef / 2.5))

    for i, landmark in enumerate(landmarks):
        if np.isnan(landmark[0]) or np.isnan(landmark[1]):
            continue

        x_coor = int(np.round(landmark[0]))
        y_coor = int(np.round(landmark[1]))

        circle_color = color if confidences is None else [255 * confidences[i], 0, 255 - 255 * confidences[i]]
        circle_color = tuple([int(c) for c in circle_color])

        cv2.circle(visualization, (x_coor, y_coor), radius, circle_color, thickness)

        if annotation_strings is not None:
            cv2.putText(visualization, annotation_strings[i], (x_coor - (radius * 2), y_coor - (radius * 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size[0], (36, 255, 12), text_size[1])

    return visualization


def draw_pose(image: np.ndarray, rotation_matrix: np.ndarray, axis_center: Optional[Tuple[int, int]] = None,
              line_length: Optional[int] = 100, lines_thicknesses: Optional[Tuple[int, int, int]] = (3, 3, 2)) -> np.ndarray:
    """
    Draws pose axises on top of the image.
    By default, green points down, red points towards the camera and blue points to the left from the bodies' perspective.
    :param image:
    :param rotation_matrix:
    :param axis_center_x: Center x of the axises.
    :param axis_center_y: Center y of the axises.
    :param line_length:
    :param lines_thicknesses:
    :return: visualization
    """
    visualization = image.copy()
    height, width = image.shape[:2]
    mid_dim = min(width, height)

    if axis_center is None:
        axis_center = (width / 2, height / 2)

    if line_length is None:
        line_length = max(1, mid_dim * 7 / 16)

    if lines_thicknesses is None:
        xy_thickness = max(1, int(mid_dim / 50))
        z_thickness = max(1, int(xy_thickness * 0.8))
        lines_thicknesses = (xy_thickness, xy_thickness, z_thickness)
        # print("lines_thicknesses", lines_thicknesses)

    x1 = int(line_length * rotation_matrix[0][0] + axis_center[0])
    y1 = int(line_length * rotation_matrix[1][0] + axis_center[1])

    x2 = int(line_length * rotation_matrix[0][1] + axis_center[0])
    y2 = int(line_length * rotation_matrix[1][1] + axis_center[1])

    x3 = int(line_length * rotation_matrix[0][2] + axis_center[0])
    y3 = int(line_length * rotation_matrix[1][2] + axis_center[1])

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    points = [(x1, y1), (x2, y2), (x3, y3)]

    order = np.argsort(rotation_matrix[2, :])
    for idx, line_thickness in zip(order, lines_thicknesses):
        cv2.line(visualization, (int(axis_center[0]), int(axis_center[1])), points[idx], colors[idx], line_thickness)

    return visualization


def draw_bounding_box(image: np.ndarray, bounding_box: Union[List, np.ndarray], thickness: int = 1,
                      color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    viz = image.copy()
    bounding_box = np.array(bounding_box, dtype=np.int32)

    if thickness is None:
        thickness = max(1, int(min(image.shape[:2]) / 200))

    elif type(thickness) == float:
        thickness = max(1, int(min(image.shape[:2]) * thickness / 200))

    return cv2.rectangle(viz, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color, thickness)


def draw_irrectangle(image: np.ndarray, irrectangle: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), thickness: Optional[int] = 2, fill=False) -> np.ndarray:
    viz = image.copy()
    irrectangle = np.array([irrectangle[2], irrectangle[0], irrectangle[1], irrectangle[3]]).astype(int)

    thickness = thickness or max(1, int(min(image.shape[:2]) / 200))

    if fill:
        cv2.fillPoly(viz, [irrectangle], color)
    else:
        cv2.drawContours(viz, [irrectangle], -1, color, thickness)

    return viz


def draw_blob(image: np.ndarray, bounding_box: Union[List, np.ndarray], color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    viz = image.copy()
    bounding_box = np.array(bounding_box, dtype=np.int32)

    return cv2.rectangle(viz, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color, -1)


def draw_confidences(image: np.ndarray, confidences_dict: Dict[str, float], top_left_point: Union[List, np.ndarray, Tuple[int, int]] = (0, 0),
                     color: Tuple[int, int, int] = (0, 0, 255), font: int = cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 0.5, thickness: int = 2) -> np.ndarray:
    viz = image.copy()
    conf_names = list(confidences_dict.keys())

    for i in range(0, len(confidences_dict), 2):
        conf_texts_list = []

        for conf_name in conf_names[i: i+2]:
            conf_value = confidences_dict[conf_name]
            conf_text = "{}: {}".format(conf_name, conf_value) if int(conf_value) == conf_value else "{}: {:.2f}".format(conf_name, conf_value)

            conf_texts_list.append(conf_text)

        conf_line = " | ".join(conf_texts_list)

        # x_offset = (int(i % 2 == 0) + 1) * 10
        y_offset = ((i // 2) + 1) * (30 * font_scale)

        # conf_text = " | ".join(["{}: {:.2f}".format(conf_name, conf_value) for conf_name, conf_value in confidences_dict.items()])
        cv2.putText(viz, conf_line, (int(top_left_point[0] + 10), int(top_left_point[1] + y_offset)), font, font_scale, color, thickness=thickness)

    return viz


def draw_line(image, xy1, xy2, color=(255, 0, 0), thickness=2):
    visualization = image.copy()

    if xy1[0] != xy2[0]:
        x_coors = [0, image.shape[1]]
        x = [xy1[0], xy2[0]]
        y = [xy1[1], xy2[1]]

        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)

        y_coors = polynomial(x_coors).astype(int)

    else:
        x_coors = [xy1[0], xy2[0]]
        y_coors = [0, image.shape[0]]

    visualization = cv2.line(visualization, (x_coors[0], y_coors[0]), (x_coors[1], y_coors[1]), color, thickness)

    return visualization


def visualize_heatmaps(heatmaps, columns=3, add_text=True, **kwargs):
    titles = kwargs.get('titles', [])

    assert heatmaps is not None, "heatmaps list is None"
    assert len(titles) == 0 or len(heatmaps) == len(titles), "Image list and titles are not the same size"

    total_plots = len(heatmaps)
    n_columns = min(columns, total_plots)
    n_rows = np.ceil(total_plots / n_columns).astype(int)

    if kwargs.get('fig_size', None):
        fig_size = kwargs.get('fig_size', DEFAULT_PLOT_SIZE)
    else:
        fig_size_per_row = kwargs.get('fig_size_per_row', DEFAULT_PLOT_SIZE)
        fig_size = (fig_size_per_row[0], fig_size_per_row[1] * n_rows)

    figure = plt.figure(figsize=fig_size)
    plt.grid(False)

    plot_idx = 0
    for row_idx in range(n_rows):
        for column_idx in range(n_columns):
            if plot_idx >= len(heatmaps):
                break

            gridded_subplot = plt.subplot2grid((n_rows, n_columns), (row_idx, column_idx))
            subplot_data = heatmaps[plot_idx]
            plt.imshow(subplot_data)

            if titles: plt.title(titles[plot_idx])

            if add_text:
                mean_value = (subplot_data.min() + subplot_data.max()) / 2

                for i, j in product(range(subplot_data.shape[0]), range(subplot_data.shape[1])):
                    class_text = "{:0.2f}".format(subplot_data[i, j])
                    text_color = "black" if subplot_data[i, j] > mean_value else "white"

                    gridded_subplot.text(j, i, class_text, horizontalalignment="center", color=text_color)

            plot_idx += 1

    plt.subplots_adjust(top=0.99, bottom=0.01, wspace=0.5, hspace=0.5)
    plt.tight_layout()

    __show_save_logic(figure, **kwargs)