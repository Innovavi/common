from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

import sys
import numpy as np


def get_spent_time(was: datetime, do_print: bool = True) -> Tuple[datetime, timedelta]:
    """
    Calculates the difference between the given time and now.
    :param was: Other time to compare now to.
    :param do_print: Print the difference between times.
    :return: The current time and the difference between the given time and now.
    """
    now = datetime.now()
    difference = now - was

    if do_print:
        print("time taken:", difference)

    return now, difference


def batch_num_blink(batch_num: int, additional_info: Dict[str, Any] = None, batch_max: Optional[int] = None):
    """
    Displays the current batch number and the additional info if given in a single line.
    :param batch_num: The number of the batch.
    :param batch_max: The number of total batches.
    :param additional_info: Additional information to display. Must be a dictionary where key is the name of a variable and the value is the variable.
    """
    batch_string = "\rBatch number: {}".format(batch_num+1)
    if batch_max is not None:
        batch_string += "/ {}".format(batch_max)

    if additional_info is not None:
        for key, item in additional_info.items():
            sys.stdout.write(" | {}: {}".format(key, item))

    sys.stdout.flush()


def train_val_test_split(X: np.ndarray, y: np.ndarray, val_size: float = 0.12, test_size: float = 0.08, random_state: int = 42):
    """
    Splits the train, validation and test set into separate ones using given parameters and sklearn train_test_split function.
    :param X: Input data array.
    :param y: Label data array.
    :param val_size: Percent of the whole dataset that validation set should have.
    :param test_size: Percent of the whole dataset that test set should have.
    :param random_state: State of randomness.
    :return: Train, Validation and Test subsets of X and y
    """
    assert val_size + test_size < 1, "The sizes sum is greater than 1"

    first_test_size = val_size + test_size
    second_test_size = test_size / first_test_size

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=first_test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=second_test_size, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def is_channels_first(data_format: str) -> bool:
    """
    Checks if the data format is channels first.
    :param data_format:
    :return: Bool indicating if the channels are first.
    """
    return data_format == "channels_first" or data_format == 'NCHW'

