import json
import os
from typing import Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from common.miscellaneous import print_indexed_list, verbose_print
from common.data_manipulation.string_tools import sort_strings_by_number_values


def get_model_config(model_dir: str) -> Dict[str, Any]:
    """
    Reads the latest config file in specified model directory.
    :param model_dir: Model directory with config file.
    :return: Configuration dictionary.
    """
    cfgs = sorted([x for x in os.listdir(model_dir) if x.endswith('json')])

    config_path = os.path.join(model_dir, cfgs[-1])

    with open(config_path) as f:
        config = json.load(f)

    return config


def load_keras_model(model_fullname: str, device: str = 'GPU'):
    initialize_memory_growth(device)

    loaded_model = tf.keras.models.load_model(model_fullname, compile=False)

    return loaded_model


def initialize_memory_growth(device: str = 'GPU'):
    if device == 'GPU':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("physical_devices", physical_devices)
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Successfully allowed memory growth")
        except:
            print("Failed to set memory growth. Invalid device or cannot modify virtual devices once initialized.")

    elif device == 'CPU':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        physical_devices = tf.config.experimental.list_physical_devices()
        print("physical_devices", physical_devices)


def find_tf_model_in_dir(model_folder_path, find_best=False, checkpoint_index=None, verbose=1):
    model_folder_files = os.listdir(model_folder_path)
    model_checkpoints = sort_strings_by_number_values([filename for filename in model_folder_files if '.hdf5' in filename or '.h5' in filename])

    checkpoint_indice = [int(os.path.splitext(model_checkpoint)[0].split('_')[-1].split('-')[0]) for model_checkpoint in model_checkpoints]

    # Sort model checkpoints by their indices.
    model_checkpoints = [model_checkpoint for _, model_checkpoint in sorted(zip(checkpoint_indice, model_checkpoints))]

    if verbose:
        print_indexed_list(model_checkpoints)

    if len(model_checkpoints) == 0:
        verbose_print("No models found under dir: {}".format(model_folder_path), verbose, 1)

        return None, None

    elif checkpoint_index is not None and checkpoint_index in checkpoint_indice:
        model_checkpoint_index = checkpoint_indice.index(checkpoint_index)
    else:
        if checkpoint_index is not None and verbose:
            print("Checkpoint with index '{}' not found in dir {}. Ignoring parameter".format(checkpoint_index, model_folder_path))

        if find_best:
            checkpoint_losses = [float(os.path.splitext(model_checkpoint.split('_')[-1].split('-')[1])[0]) for model_checkpoint in model_checkpoints]
            best_loss_index = np.argmin(checkpoint_losses)
            model_checkpoint_index = best_loss_index
            checkpoint_index = checkpoint_indice[model_checkpoint_index]

        else:
            model_checkpoint_index = -1
            checkpoint_index = checkpoint_indice[model_checkpoint_index]

    tf_model_fullname = os.path.join(model_folder_path, model_checkpoints[model_checkpoint_index])
    if verbose:
        print("Found model: {} With index: {}".format(tf_model_fullname, model_checkpoint_index))

    return tf_model_fullname, checkpoint_index


def get_model_size(model: tf.keras.Model):
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))

    return trainable_count, non_trainable_count



