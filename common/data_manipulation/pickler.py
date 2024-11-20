import pickle
from typing import Any


def load_pickle(pickle_file_path: str):
    """
    Loads a pickle object regardless of its Python version (can be 2 or 3).
    :param pickle_file: Path to pickled file
    :return: Unpickled file data.
    """
    try:
        with open(pickle_file_path, 'rb') as f:
            pickle_data = pickle.load(f)

    except UnicodeDecodeError as e:
        with open(pickle_file_path, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')

    except Exception as e:
        print('Unable to load data ', pickle_file_path, ':', e)
        raise

    return pickle_data


def write_pickle(pickle_full_path: str, object_to_pickle: Any):
    """
    Pickles an object to specified path.
    :param pickle_full_path: Full path to pickle to.
    :param object_to_pickle: Object that will be pickled.
    """
    with open(pickle_full_path, "wb") as fp:  # Pickling
        pickle.dump(object_to_pickle, fp)