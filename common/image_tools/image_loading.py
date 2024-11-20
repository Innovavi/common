import os
from typing import Optional, List, Callable

import cv2
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageOps


def load_image(image_fullname: str, force_standard: bool = True, force_mpimg: bool = False, read_alpha: bool = False, verbose: int = 1) -> Optional[np.ndarray]:
    if not os.path.exists(image_fullname):
        print("Image not found under fullname:")
        print(image_fullname)

        return None

    try:
        if read_alpha:
            image = cv2.imread(image_fullname, flags=cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        elif force_mpimg:
            image = mpimg.imread(image_fullname)

        else:
            image = Image.open(image_fullname)

            try:
                image = ImageOps.exif_transpose(image)
            except:
                print("Error while ImageOps.exif_transpose {}".format(image_fullname))

            image = np.array(image)

    except:
        print("Error while loading image {}".format(image_fullname))

        return None

    if len(image.shape) < 2:
        if verbose > 0:
            print("Bad loaded image shape: {} | image_fullname: {}".format(image.shape, image_fullname))

        return None

    if force_standard:
        image = standardise_image(image)

    return image


def load_image_batch(image_names: List[str], image_load_fn: Callable):
    return [image_load_fn(image_name) for image_name in image_names]


def standardise_image(image: np.ndarray) -> np.ndarray:
    if np.max(image) <= 1:
        image = to_uint8(image, invert=False)

    image = format_image_to_rgb(image)

    return image


def format_image_to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image


def to_uint8(image: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    Converts the image to uint8 and inverts it if required.
    :param image: Image to convert.
    :param invert: Invert if the objects are black, while the background is white.
    :return: Converted Image.
    """
    if np.max(image) <= 1:
        if invert:
            image = 1 - image

        return (image * 255).astype(np.uint8)

    else:
        if invert:
            image = 255 - image

        return image.astype(np.uint8)

