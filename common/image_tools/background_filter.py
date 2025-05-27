import os
from typing import Optional

import cv2
import numpy as np

from common.data_manipulation.image_data_tools.bounding_box_tools import get_random_bbox_in_image
from common.image_tools.morphological_filters import closing_filter, open_filter, erosion
from common.image_tools.connected_object_tools import filter_by_object_property, PropertyType
from common.image_tools.cropping import crop_box
from common.image_tools.resizer import resize_image, ResizingType
from common.image_tools.morphological_filters import dilation
from common.miscellaneous import verbose_print
from common.visualizations.image_visualizations import show_image


def get_image_background_mask(image: np.ndarray, kernel_fraction: float = 0.02, filter_area_fraction: float = 0.01) -> Optional[np.ndarray]:
    image_height, image_width = image.shape[:2]
    kernel_size = int(min(image_height, image_width) * kernel_fraction)
    kernel_shape = (kernel_size, kernel_size)

    maxed_image = np.max(image, axis=2)
    thresh = maxed_image < 1

    if thresh.all():
        return thresh

    # Mask fine-tuning
    mask = closing_filter(thresh, kernel_shape)
    mask = erosion(mask, kernel_shape)
    mask = open_filter(mask, kernel_shape)

    if len(np.unique(mask)) < 2:
        return mask

    if filter_area_fraction > 0:
        filter_area_threshold = image_height * image_width * filter_area_fraction
        mask, object_count = filter_by_object_property(mask, PropertyType.PROPERTY_AREA, 0, filter_area_threshold, invert=True)

    return mask


def fill_image_background_with_another_image(image: np.ndarray, background_image: np.ndarray, mask: Optional[np.ndarray] = None,
                                             blur_kernel_fraction: float = 0.02, verbose: int = 0, viz_folder_fullpath: Optional[str] = None):
    if mask is None:
        mask = get_image_background_mask(image, blur_kernel_fraction)

    if len(np.unique(mask)) < 2:
        return image

    bool_mask = mask.astype(bool)
    image_height, image_width = image.shape[:2]

    if image_height < background_image.shape[0] or image_width < background_image.shape[1]:
        random_box = get_random_bbox_in_image(background_image.shape, image.shape, False)
        background_image = crop_box(background_image, random_box)

    background_image = resize_image(background_image, image.shape, resizing_type=ResizingType.FIXED)

    image_with_background = np.clip(bool_mask[:, :, np.newaxis] * background_image +
                                    ~bool_mask[:, :, np.newaxis] * image, 0, 255).astype(np.uint8)

    if viz_folder_fullpath is not None and verbose > 0:
        mask_save_fullname = os.path.join(viz_folder_fullpath, "background_mask.jpg")
        show_image(mask, axis=True, save_fullname=mask_save_fullname, only_save=True)

        background_crop_save_fullname = os.path.join(viz_folder_fullpath, "background_crop.jpg")
        show_image(background_image, axis=True, save_fullname=background_crop_save_fullname, only_save=True)

        image_with_background_save_fullname = os.path.join(viz_folder_fullpath, "image_with_background.jpg")
        show_image(image_with_background, axis=True, save_fullname=image_with_background_save_fullname, only_save=True)

    if blur_kernel_fraction > 0:
        blur_kernel_size = max(int(min(image_height, image_width) * blur_kernel_fraction), 3)
        blur_kernel_shape = (blur_kernel_size, blur_kernel_size)
        verbose_print("blur_kernel_size: {}".format(blur_kernel_size), verbose, 3)

        blurred_image_with_background = cv2.blur(image_with_background, blur_kernel_shape)

        erroded_mask = erosion(mask, blur_kernel_shape)
        dilated_mask = dilation(mask, blur_kernel_shape)
        contour_mask = (erroded_mask - dilated_mask).astype(bool)

        image_with_background = np.clip(contour_mask[:, :, np.newaxis] * blurred_image_with_background +
                                        ~contour_mask[:, :, np.newaxis] * image_with_background, 0, 255).astype(np.uint8)

        if viz_folder_fullpath is not None and verbose > 0:
            contour_mask_save_fullname = os.path.join(viz_folder_fullpath, "contour_mask.jpg")
            show_image(contour_mask, axis=True, save_fullname=contour_mask_save_fullname, only_save=True)

            image_with_background_save_fullname = os.path.join(viz_folder_fullpath, "image_with_background_post_blur.jpg")
            show_image(image_with_background, axis=True, save_fullname=image_with_background_save_fullname, only_save=True)

    return image_with_background


