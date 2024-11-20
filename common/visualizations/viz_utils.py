import numbers
import os
from typing import Optional, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

DEFAULT_PLOT_SIZE = (22, 8)
DEFAULT_SMALL_PLOT_SIZE = (11, 4)
DEFAULT_SQUARE_PLOT_SIZE = (22, 22)
DEFAULT_SMALL_SQUARE_PLOT_SIZE = (8, 8)
float_string_template = '{:.2f}'
integer_string_template = '{}'


def __show_save_logic(figure: plt.Figure, title: Optional[str] = '', save_fullname: Optional[str] = '', only_save: bool = False, do_close: bool = True, **kwargs) -> None:
    """
    Plot showing and saving logic.
    :param figure: Figure object.
    :param title: Plot name to be shown on top of figure.
    :param save_fullname: Full save path.
    :param only_save: Only save figure to disk, but not show.
    :param do_close: Close and clear figure after showing and/or saving. When set to False, the figure can be altered after function call.
    """
    if save_fullname:
        title_name = title or os.path.split(save_fullname)[1]
        plt.suptitle(title_name)

        plt.savefig(save_fullname, bbox_inches='tight', pad_inches=0.1)
        print("Saved plot at", save_fullname)

    elif title:
        plt.suptitle(title)

    if not only_save or not save_fullname:
        plt.show()

    if do_close:
        plt.clf()
        plt.close(figure)


def __set_log_scale(ax: plt.Axes, log_scale: Union[bool, str]) -> None:
    if log_scale:
        if (type(log_scale) == str and log_scale.lower() == 'x') or type(log_scale) == bool:
            ax.set_xscale("log")

        if (type(log_scale) == str and log_scale.lower() == 'y') or type(log_scale) == bool:
            ax.set_yscale("log")


def __set_axis_labels(ax, axis_labels):
    if axis_labels and (type(axis_labels) == list or type(axis_labels) == tuple):
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])


def __set_axis_limits(ax, axis_limits):
    """
    Acceptable formats of axis_limits:
    num1 - X axis from 0 to num1 or num1 to 0 if num1 is negative.
    (num1, num2) - X axis from num1 to num2.
    ((num1, num2), ()) - X axis from num1 to num2.
    ((), (num3, num4)) - Y axis from num3 to num4.
    ((num1, num2), (num3, num4)) - X axis from num1 to num2, Y axis from num3 to num4.
    """
    if axis_limits:
        if isinstance(axis_limits, numbers.Number):
            ax.set_xlim(0, axis_limits) if axis_limits >= 0 else ax.set_xlim(axis_limits, 0)

        elif (type(axis_limits) == list or type(axis_limits) == tuple) and len(axis_limits) == 2:
            if isinstance(axis_limits[0], numbers.Number) and isinstance(axis_limits[1], numbers.Number):
                ax.set_xlim(*axis_limits)

            elif (isinstance(axis_limits[0], tuple) or isinstance(axis_limits[0], list)):

                if len(axis_limits[0]) == 0 and (isinstance(axis_limits[1], tuple) or isinstance(axis_limits[1], list)) and len(axis_limits[1]) == 2:
                    ax.set_ylim(*axis_limits[1])

                elif (isinstance(axis_limits[0], tuple) or isinstance(axis_limits[0], list)) and len(axis_limits[0]) == 2 and \
                        (isinstance(axis_limits[1], tuple) or isinstance(axis_limits[1], list)) and len(axis_limits[1]) == 2:
                    ax.set_xlim(*axis_limits[0])
                    ax.set_ylim(*axis_limits[1])

            else:
                raise ValueError("Invalid axis_limits format. {}".format(axis_limits))
        else:
            raise ValueError("Invalid axis_limits format. {}".format(axis_limits))


def __get_bivariate_min_max(bivariate_list):
    return np.min([np.min(row) for row in bivariate_list]), np.max([np.max(row) for row in bivariate_list])


ALL_COLORS = list(matplotlib.colors.BASE_COLORS.keys())[:3] + list(matplotlib.colors.BASE_COLORS.keys())[3:5:-1] + list(matplotlib.colors.BASE_COLORS.keys())[5:7] + \
             list(matplotlib.colors.TABLEAU_COLORS.keys()) + list(matplotlib.colors.CSS4_COLORS.keys())
MARKER_LIST = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'X', 'D', 'H', '+', 'x']
LINESTYLES_LIST = ['-', '-.', '--', ':']
