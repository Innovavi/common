import os
from typing import List, Any, Union
import numpy as np


def print_indexed_list(some_list: List) -> None:
    """
    Prints the list content along rows together with the indices.
    :param some_list:
    """
    for i, list_value in enumerate(some_list):
        print(i, list_value)


def verbose_print(string_to_print: Any, verbose: int, required_verbose: int) -> bool:
    if verbose >= required_verbose:
        print("verbose({}/{}):".format(verbose, required_verbose), string_to_print)
        return True

    return False


def print_fraction(size_1, size_2, prestring: str = ''):
    print("{} {} / {} = {:.3f}".format(prestring, size_1, size_2, size_1 / size_2))


def over_shift(number: Union[float, int], min_value: Union[float, int] = 0, max_value: Union[float, int] = 2) -> Union[float, int]:
    """
    Circles the number around. If it is >max_value, sets it to min_value and vice versa.
    :param number:
    :param min_value:
    :param max_value:
    :return: number
    """
    if number > max_value:
        return min_value

    if number < min_value:
        return max_value

    return number


def get_folder_content_paths(folder_path, condition=None):
    files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path)])

    if condition is not None:
        files = [file for file in files if condition(file)]

    return files


def sigmoid_fn(array: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-array))


def softmax_fn(array: np.ndarray, sum_dim: int = 0) -> np.ndarray:
    return np.exp(array) / np.sum(np.exp(array), axis=sum_dim)


def harmonic_mean(x, y):
    return 2 * x * y / (x + y)


def geometric_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))



