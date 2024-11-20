from typing import Tuple, Optional

import numpy as np
from common.data_manipulation.image_data_tools.bounding_box_tools import change_bbox_origin_point, \
    get_max_image_square_around_bbox, get_bbox_dimensions, expand_bbox
from common.data_manipulation.image_data_tools.landmark_tools import change_landmarks_origin_point
from common.image_tools.cropping import crop_box
from common.miscellaneous import verbose_print


def prepare_image_for_augmentation(image: np.ndarray, main_bbox: np.ndarray, other_bboxes: np.ndarray = None, landmarks: np.ndarray = None,
                                   minimum_expansion_ratio: Optional[float] = None, verbose: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    max_image_square_around_bbox = np.squeeze(get_max_image_square_around_bbox(main_bbox.astype(int), image.shape))

    if minimum_expansion_ratio is not None:
        original_h, original_w = get_bbox_dimensions(main_bbox)
        exp_h, exp_w = get_bbox_dimensions(max_image_square_around_bbox)

        current_expansion_ratio = np.mean([((exp_h/original_h) - 1) / 2, ((exp_w/original_w) - 1) / 2])

        if current_expansion_ratio < minimum_expansion_ratio:
            missing_expansion_ratio = minimum_expansion_ratio - current_expansion_ratio
            max_image_square_around_bbox = expand_bbox(max_image_square_around_bbox, missing_expansion_ratio)

            verbose_print("max_image_square_around_bbox was too small: {}. Expanded further by {}".format(current_expansion_ratio, missing_expansion_ratio), verbose, 5)

    cropped_image = crop_box(image, max_image_square_around_bbox)

    aligned_main_bbox = change_bbox_origin_point(main_bbox, destination_origin_point=max_image_square_around_bbox)
    aligned_other_bboxes = np.array([change_bbox_origin_point(other_bbox, destination_origin_point=max_image_square_around_bbox) for other_bbox in other_bboxes]) \
        if other_bboxes is not None and len(other_bboxes) > 0 \
        else None
    aligned_all_bboxes = np.concatenate([aligned_main_bbox[np.newaxis], aligned_other_bboxes], axis=0) if len(other_bboxes) > 0 else aligned_main_bbox[np.newaxis]

    aligned_landmarks = change_landmarks_origin_point(landmarks, destination_origin_point=max_image_square_around_bbox) if landmarks is not None else None

    if verbose > 4:
        print("original bounding_box       ", other_bboxes)
        print("max_image_square_around_bbox", max_image_square_around_bbox)
        print("aligned_all_bboxes          ", aligned_all_bboxes)

    return cropped_image, aligned_all_bboxes, aligned_landmarks, max_image_square_around_bbox


# def mask_boarder_transform(image: np.ndarray, face_bounding_box: Optional[np.ndarray] = None, landmarks: Optional[np.ndarray] = None):
#     image_h, image_w = image.shape[:2]
#     face_h, face_w = get_bbox_dimensions(face_bounding_box) if face_bounding_box is not None else (image_h, image_w)
#     face_h, face_w = int(face_h), int(face_w)
#     boarder_start_x, boarder_start_y = face_bounding_box[:2].astype(int) if face_bounding_box is not None else (0, 0)
#     basic_lms = landmarks[:5] if landmarks is not None else np.zeros(5)
#     transform_params = [0, 0]
#
#     aug_image = image.copy()
#
#     # Which boarder to crop
#     boarder_ok = False
#     while not boarder_ok:
#         boarder_fraction = np.random.rand() / 2  # Max 0.5
#         boarder_side = np.random.randint(0, 4)  # Clockwise
#
#         if boarder_side == 0:  # top
#             boarder_size = int(face_h * boarder_fraction)
#             boarder_bbox = np.array([0, 0, image_w, boarder_start_y + boarder_size])
#
#         elif boarder_side == 1:  # right
#             boarder_size = face_w - int(face_w * boarder_fraction)
#             boarder_bbox = np.array([boarder_start_x + boarder_size, 0, image_w, image_h])
#
#         elif boarder_side == 2:  # bot
#             boarder_size = face_h - int(face_h * boarder_fraction)
#             boarder_bbox = np.array([0, boarder_start_y + boarder_size, image_w, image_h])
#
#         else:  # boarder_side == 4  left
#             boarder_size = int(face_w * boarder_fraction)
#             boarder_bbox = np.array([0, 0, boarder_start_x + boarder_size, image_h])
#
#         indice = find_landmarks_in_bounding_box(basic_lms, boarder_bbox) if landmarks is not None else [0]
#
#         if len(indice) + 1 < len(basic_lms):  # if 4 or 5 lms are within the boarder - it is not okay.
#             boarder_ok = True
#
#             aug_image[boarder_bbox[1]:boarder_bbox[3], boarder_bbox[0]:boarder_bbox[2]] = 0
#
#             transform_params = [boarder_fraction, boarder_side]
#
#     return aug_image, transform_params


# def fill_image_background_with_another_image(image: np.ndarray, background_image: np.ndarray, mask: Optional[np.ndarray] = None, verbose=0):
#     if mask is None:
#         mask = get_image_background_mask(image)
#
#     mask = mask.astype(np.bool)
#
#     if mask.all():
#         return image
#
#     if image.shape[0] > background_image.shape[0] or image.shape[1] > background_image.shape[1]:
#         background_image = resize_image(background_image, image.shape, dimension_to_match='smaller', add_padding=True)
#     random_box = get_random_bbox_in_image(background_image.shape, image.shape, False)
#     background_crop = crop_box(background_image, random_box)
#
#     if (background_crop.shape[0] != image.shape[0]) or (background_crop.shape[1] != image.shape[1]):
#         background_crop = resize_image(background_crop, image.shape)
#
#     mask = np.expand_dims(mask, axis=-1)
#     image_with_background = np.clip((1 - mask) * background_crop + mask * image, 0, 255).astype(np.uint8)
#
#     return image_with_background


# class Augmenter:
#
#     def __init__(self):
#         self.augmentation_sequence = iaa.Sequential(random_order=True)
#
#     def add_aspect_ratio_agmentation(self, probability, max_effect):
#         minimum_aspect_ratio = 1 - max_effect
#         maximum_aspect_ratio = 1 + max_effect
#
#         aspect_ratio_agmentation = iaa.Sometimes(probability,
#                                                  iaa.OneOf(
#                                                      [
#                                                          iaa.Resize(size={"height": (minimum_aspect_ratio, maximum_aspect_ratio)},
#                                                                     interpolation='area'),
#                                                          iaa.Resize(size={"width": (minimum_aspect_ratio, maximum_aspect_ratio)},
#                                                                     interpolation='area')
#                                                      ]
#                                                  ))
#
#         self.augmentation_sequence.add(aspect_ratio_agmentation)
#
#     def add_general_augmentations(self, probability=0.8, max_aug_count=3):
#         general_augmentations_list = [
#             # Blur each image with varying strength using gaussian blur, average/uniform blur,
#             # median blur or gaussian noise.
#             iaa.OneOf([
#                 iaa.GaussianBlur(sigma=(0, 3.0)), iaa.AverageBlur(k=(2, 4)),
#                 iaa.MedianBlur(k=(3, 7)), iaa.AdditiveGaussianNoise(scale=(0.0, 10)),
#             ]),
#
#             # Sharpen each image, overlay the result with the original image using an alpha
#             # between 0 (no sharpening) and 1 (full sharpening effect).
#             iaa.Sharpen(alpha=(0, 0.3), lightness=(0.8, 1.3)),
#
#             # Same as sharpen, but for an embossing effect.
#             iaa.Emboss(alpha=(0, 0.3), strength=(0, 1.0)),
#
#             # Either drop randomly % of all pixels (i.e. set them to black) or drop them on an image
#             # with 0.1-1% of the original size, leading to small dropped rectangles.
#             iaa.OneOf([
#                 iaa.Dropout((0.002, 0.02)),
#                 iaa.CoarseDropout((0.0005, 0.005), size_percent=(0.0005, 0.005)),
#             ]),
#
#             # Add a value to each pixel. Or Change brightness of images.
#             iaa.OneOf([
#                 iaa.Add((-10, 10)), iaa.Multiply((0.8, 1.2)),
#             ]),
#
#             # Improve or worsen the contrast of images.
#             iaa.contrast.LinearContrast((0.8, 1.3)),
#         ]
#
#         general_augmentations = iaa.Sometimes(probability, iaa.SomeOf((0, max_aug_count), general_augmentations_list, random_order=True))
#
#         self.augmentation_sequence.add(general_augmentations)
#
#     def add_rotation_augmentation(self, rotate_first=False, rotation_angle_range=[-90, 90]):
#         pass
