import cv2
import numpy as np
from skimage import metrics


def prepare_image_for_comparison(image: np.ndarray, comparison_size: int = 128) -> np.ndarray:
    # resize
    image = cv2.resize(image, (comparison_size, comparison_size))

    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def compare_images(prepared_image_1: np.ndarray, prepared_image_2: np.ndarray) -> float:
    """
    Compares images using skimage.metrics.structural_similarity
    :param prepared_image_1:
    :param prepared_image_2:
    :return:
    """
    assert prepared_image_1.shape == prepared_image_2.shape, "Prepared image shapes do not match. image_1: {}, image_2: {}".format(
        prepared_image_1.shape, prepared_image_2.shape
    )

    similarity = metrics.structural_similarity(prepared_image_1, prepared_image_2)

    return similarity