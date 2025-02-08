from typing import Union

import numpy as np
import scipy.stats as stats


def redistribute_offlier(random_num: float, loc: float, scale: float, min_value: float, max_value: float) -> float:
    new_random_num = random_num

    while min_value > new_random_num or max_value < new_random_num:
        new_random_num = np.random.normal(loc, scale)


    return new_random_num


def redistribute_offliers_array(random_num_array: np.ndarray, loc: float, scale: float, min_value: float, max_value: float) -> np.ndarray:
    new_random_num_array = random_num_array

    while any(min_value > new_random_num_array) or any(max_value < new_random_num_array):
        new_random_num_array = np.where((min_value > random_num_array) | (max_value < random_num_array),
                                        np.random.normal(loc, scale, random_num_array.shape),
                                        random_num_array)

    return new_random_num_array


def redistribute_top_proba(random_num: float, loc: float, scale: float, loc_offset: float = 0, scale_offset: float = 0, sharpness: float = 100,
                           max_proba: float = 1.) -> float:
    if np.random.rand() < quadratic_proba(random_num, loc + loc_offset, scale + scale_offset, sharpness, max_proba):
        new_random_num = np.random.normal(loc, scale)
    else:
        new_random_num = random_num

    return new_random_num


def redistribute_top_proba_array(random_num_array: np.ndarray, loc: float, scale: float, loc_offset: float = 0, scale_offset: float = 0, sharpness: float = 100,
                                 max_proba: float = 1.) -> np.ndarray:
    new_random_num_array = np.where(np.random.rand(len(random_num_array)) < quadratic_proba(random_num_array, loc + loc_offset, scale + scale_offset,
                                                                                            sharpness, max_proba),
                                    np.random.normal(loc, scale, random_num_array.shape),
                                    random_num_array)

    return new_random_num_array


def quadratic_proba(x: Union[np.ndarray, float], loc: float, scale: float, sharpness: float = 200, max_proba: float = 1.) -> Union[np.ndarray, float]:
    y = -(x - loc) ** 2 * (sharpness * scale * max_proba) + max_proba

    return y


def get_normal_in_abs_range(ranges: Union[np.ndarray, float], loc: float = 0, scale: float = 1) -> Union[np.ndarray, float]:
    """
    Randomises array or value in given ranges based on given normal distribution parameters.
    :param ranges:
    :param loc:
    :param scale:
    :return: normal_distribution_values
    """
    ranges_size = len(ranges) if type(ranges) == np.ndarray else None
    x_randoms = np.random.normal(loc, scale, ranges_size)

    normal_distribution_values = np.clip(ranges * x_randoms, -ranges, ranges)

    return normal_distribution_values


def get_flattened_distribution(loc: float = 0, scale: float = 0, min_scale: float = -1, max_scale: float = 1,
                               re_loc: float = 0, re_scale: float = 0, re_sharpness: float = 0, re_count: int = 1) -> float:
    if any(np.array([re_loc, re_scale, re_sharpness]) != 0) and re_count > 0:
        random_scale = np.random.normal(loc, scale)

        for i in range(re_count):
            random_scale = redistribute_top_proba(random_scale, loc, scale, re_loc, re_scale, re_sharpness)

        random_scale = redistribute_offlier(random_scale, loc, scale, min_scale, max_scale)

    else:
        random_scale = get_limited_gaussian(loc, scale, min_scale, max_scale)

    return random_scale


def get_limited_gaussian(loc: float, scale: float, min_value: float, max_value: float, sample_size: int = 1) -> Union[float, np.ndarray]:
    if scale > 0:
        limited_gaussian = stats.truncnorm((min_value - loc) / scale, (max_value - loc) / scale, loc=loc, scale=scale).rvs(sample_size)

    else:
        limited_gaussian = np.ones(sample_size) * loc

    if sample_size == 1:
        limited_gaussian = limited_gaussian[0]

    return limited_gaussian


def norm_values_to_range(values, min_x, max_x, min_y=-1, max_y=1):
    scale_change = abs((max_y - min_y) / (max_x - min_x))

    x_mean = ((max_x + min_x) / 2) * scale_change
    y_mean = (max_y + min_y) / 2
    mean_change = y_mean - x_mean

    norm_values = values * scale_change + mean_change

    return norm_values