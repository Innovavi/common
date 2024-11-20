from itertools import product
from typing import Optional, List, Dict

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from common.visualizations.viz_utils import DEFAULT_PLOT_SIZE, ALL_COLORS, LINESTYLES_LIST, MARKER_LIST, \
    DEFAULT_SQUARE_PLOT_SIZE, DEFAULT_SMALL_SQUARE_PLOT_SIZE, integer_string_template, float_string_template, \
    __set_log_scale, __set_axis_labels, __set_axis_limits, __show_save_logic

DEFAULT_QUANTILES = [0.1, 0.25, 0.75, 0.9]
ACCEPTED_KWARGS_LIST = ['fig_size', 'fig_size_per_row', 'log_scale', 'axis_labels', 'axis_limits', 'title', 'titles', 'save_fullname', 'only_save', 'do_close']


def plot_histogram(data_to_show, bins=50, add_quatinles=None, add_analysis=True, add_cumulative=False, x_ticks=None, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        max_tick = max(x_ticks)
        ax.set_xlim(min(x_ticks) - max_tick * 0.01, max(x_ticks) * 1.01)

    counts, thresholds, _ = ax.hist(data_to_show, bins=bins, zorder=1, density=add_cumulative)

    if add_analysis:
        mean_value_str = "mean ({:.3f})".format(np.mean(data_to_show))
        median_value_str = "median ({:.3f})".format(np.median(data_to_show))

        ax.axvline(np.mean(data_to_show), color='r', label=mean_value_str)
        ax.axvline(np.median(data_to_show), color='y', label=median_value_str)
        print("std:", np.std(data_to_show))

    if add_quatinles is not None:
        quantiles_to_calc = DEFAULT_QUANTILES if type(add_quatinles) == bool else add_quatinles
        quantiles = np.quantile(data_to_show, quantiles_to_calc)

        for i, (quantile, quantile_value) in enumerate(zip(quantiles, quantiles_to_calc)):
            ax.axvline(quantile, color=ALL_COLORS[i], linestyle=LINESTYLES_LIST[2], label="Quantile {:.2f}".format(quantile_value))

    if add_cumulative:
        ax2 = ax.twinx()
        n, bins, patches = ax2.hist(data_to_show, bins, color='black', density=True, histtype='step', cumulative=True, label='Cumulative')

    ax.legend()

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_multiple_histograms(datas_to_show, datas_names=None, bins=50, add_quatinles=None, add_analysis=True, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    alpha = 1 / len(datas_to_show)
    datas_names = datas_names if datas_names is not None else np.arange(len(datas_to_show))

    _, bins_lims = np.histogram(np.concatenate(datas_to_show), bins=bins)

    for i, (data_to_show, data_name) in enumerate(zip(datas_to_show, datas_names)):
        color_idx = i * 2
        counts, _, _ = ax.hist(data_to_show, bins=bins_lims, color=ALL_COLORS[color_idx], label=data_name, alpha=alpha)

        string_to_print = str(data_name) + ": "

        if add_analysis:
            mean_value = data_to_show.mean()
            median_value = np.median(data_to_show)
            max_value_idx = np.argmax(counts)
            max_value = bins_lims[max_value_idx]
            mean_value_str = "{} mean ({:.3f})".format(data_name, mean_value)
            median_value_str = "{} median ({:.3f})".format(data_name, median_value)
            max_value_str = "{} max ({:.3f})".format(data_name, max_value)

            ax.axvline(mean_value, color=ALL_COLORS[color_idx], label=mean_value_str, linestyle=LINESTYLES_LIST[0])
            ax.axvline(median_value, color=ALL_COLORS[color_idx], label=median_value_str, linestyle=LINESTYLES_LIST[2])
            ax.axvline(max_value, color=ALL_COLORS[color_idx], label=max_value_str, linestyle=LINESTYLES_LIST[3])

            string_to_print += "std: {:.3f}".format(np.std(data_to_show))

        if add_quatinles is not None:
            quantiles_to_calc = DEFAULT_QUANTILES if type(add_quatinles) == bool else add_quatinles
            quantiles = np.quantile(data_to_show, quantiles_to_calc)

            string_to_print += " | quants:"

            for i, (quantile, quantile_value) in enumerate(zip(quantiles, quantiles_to_calc)):
                ax.axvline(quantile, color=ALL_COLORS[color_idx], linestyle=LINESTYLES_LIST[1])

                string_to_print += " {:.2f}:{:.2f} ;".format(quantile_value, quantile)

        if add_analysis or add_quatinles is not None:
            print(string_to_print)

    ax.legend()

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_scatter(x, y, s=None, **kwargs):
    #TODO: implement names at each point with ax.annotate()
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.scatter(x, y, s)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_xy_curves(x, ys, names=None, marker=True, **kwargs):
    """

    :param x:
    :param ys:
    :param names:
    :param marker:
    :param kwargs: axis_labels = ['Threshold', 'Percent']
    """
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    for i, y in enumerate(ys):
        this_marker_index = MARKER_LIST[i % len(MARKER_LIST)] if marker else None
        ax.plot(x, y, marker=this_marker_index)

    if names:
        plt.legend(names)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['Threshold', 'Percent']))

    __show_save_logic(figure, **kwargs)


def plot_xs_and_ys(xs, ys, names=None, marker=True, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    for i, (x, y) in enumerate(zip(xs, ys)):
        marker_type = MARKER_LIST[i % len(MARKER_LIST)] if marker else None
        linestyle_type = LINESTYLES_LIST[i % len(LINESTYLES_LIST)]
        ax.plot(x, y, marker=marker_type, linestyle=linestyle_type, color=ALL_COLORS[i])

    if names:
        plt.legend(names)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_CED_curve(CED, thresholds, marker='o', **kwargs):
    print("Depreciated, use plot_CDF")
    plot_CDF(CED, thresholds, marker, **kwargs)


def plot_CDF(CDF, thresholds, marker='o', **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_yticks(np.arange(0, 1.01, 0.1))
    # ax.set_xticks(thresholds)

    ax.plot(thresholds, CDF, marker=marker)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', [[], [-0.05, 1.05]]))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_multiple_CED_curves(multiple_CED, thresholds, names, marker=True, **kwargs):
    print("Depreciated, use plot_CDFs")
    plot_CDFs(multiple_CED, thresholds, names, marker, **kwargs)


def plot_CDFs(CDFs, thresholds, names, marker=True, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # ax.set_xticks(thresholds, minor=True)
    # ax.xaxis.set_minor_locator(MultipleLocator(thresholds[2] - thresholds[0]))

    for i, error_percents in enumerate(CDFs):
        if len(thresholds) == len(error_percents):
            this_marker_index = MARKER_LIST[i % len(MARKER_LIST)] if marker else None
            ax.plot(thresholds, error_percents, marker=this_marker_index)

    plt.legend(names)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', [[], [-0.05, 1.05]]))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['Threshold', 'Percent']))

    __show_save_logic(figure, **kwargs)


def plot_multiple_bidir_CED_curves(multiple_percentile_CED, multiple_counted_CED, thresholds, names, marker=True, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.set_xticks(thresholds, minor=True)
    ax.xaxis.set_minor_locator(MultipleLocator(thresholds[2] - thresholds[0]))

    ax2 = ax.twinx()

    for i, (error_percents, error_counts) in enumerate(zip(multiple_percentile_CED, multiple_counted_CED)):
        if len(thresholds) == len(error_percents) == len(error_counts):
            this_marker_index = MARKER_LIST[i % len(MARKER_LIST)] if marker else None
            ax.plot(thresholds, error_percents, marker=this_marker_index)
            ax2.plot(thresholds, error_counts, marker=this_marker_index)

    ax2.set_ylabel('Counts', color='b')

    plt.legend(names)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', [[], [-0.05, 1.05]]))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['Threshold', 'Percent']))

    __show_save_logic(figure, **kwargs)


def plot_named_curves_dict(named_curves_dict: Dict[str, List[float]], x_tick_names: List[str], bar_data: List[float] = None, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_xticks(range(len(x_tick_names)))
    ax.set_xticklabels(x_tick_names, fontsize=10)

    for i, curve_values in enumerate(named_curves_dict.values()):
        marker_index = i % len(MARKER_LIST)
        ax.plot(x_tick_names, curve_values, marker=MARKER_LIST[marker_index])

    ax.legend(named_curves_dict.keys())

    if bar_data is not None:
        ax_bar = ax.twinx()

        bar_rects = ax_bar.bar(x_tick_names, bar_data, align='center', alpha=0.5, width=0.3)

        __autolabel_bars(ax_bar, bar_rects)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['Threshold', 'Error']))

    __show_save_logic(figure, **kwargs)


def plot_scatter_historgram(x: np.ndarray, y: np.ndarray, n_bins: int = 200, major_tick_step: Optional[float] = None, **kwargs) -> None:
    # Estimate the 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=n_bins)

    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)

    # Mask zeros
    H_masked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

    # Plot 2D histograms using pcolor
    figure = plt.figure(figsize=kwargs.get('fig_size', DEFAULT_SQUARE_PLOT_SIZE), clear=True)
    axHist2d = plt.subplot2grid((9, 9), (2, 1), colspan=6, rowspan=6)#, fig=figure, ax=ax)
    axHistx = plt.subplot2grid((9, 9), (0, 1), colspan=6, rowspan=2)#, fig=figure, ax=ax)
    axHisty = plt.subplot2grid((9, 9), (2, 7), colspan=2, rowspan=6)#, fig=figure, ax=ax)
    axHistCmap = plt.subplot2grid((9, 9), (8, 1), colspan=6, rowspan=1)#, fig=figure, ax=ax)

    color_mesh = axHist2d.pcolormesh(xedges, yedges, H_masked)

    axHistx.hist(x, bins=xedges, alpha=0.5)
    axHisty.hist(y, bins=yedges, alpha=0.5, orientation='horizontal')

    #region Axis limits
    axis_limits = kwargs.get('axis_limits', None)
    if axis_limits is None:
        x_lim = max(abs(min(x)), max(x)) * 1.01  # plus 1% so that the plot boarders would be slightly further from points.
        y_lim = max(abs(min(y)), max(y)) * 1.01  # plus 1% so that the plot boarders would be slightly further from points.
    else:
        x_lim, y_lim = axis_limits, axis_limits

    x_lower_lim = -x_lim if min(x) < 0 else 0
    y_lower_lim = -y_lim if min(y) < 0 else 0

    __set_axis_limits(axHist2d, [[x_lower_lim, x_lim], [y_lower_lim, y_lim]])
    axHistx.set_xlim(x_lower_lim, x_lim)
    axHisty.set_ylim(y_lower_lim, y_lim)
    #endregion

    # Ticks
    if major_tick_step is not None:
        major_ticks = np.arange(-max(x_lim, y_lim), max(x_lim, y_lim), major_tick_step)
        axHist2d.set_xticks(major_ticks)
        axHist2d.set_yticks(major_ticks)

        axHistx.set_xticks(major_ticks)
        axHisty.set_yticks(major_ticks)

    # Log scale
    log_scale = kwargs.get('log_scale', None)
    if log_scale:
        if (type(log_scale) == str and log_scale.lower() == 'x') or type(log_scale) == bool:
            axHist2d.set_xscale("log")
            axHisty.set_xscale("log")

        if (type(log_scale) == str and log_scale.lower() == 'y') or type(log_scale) == bool:
            axHist2d.set_yscale("log")
            axHistx.set_yscale("log")

    # Add heatmap bar
    cbar = figure.colorbar(color_mesh, cax=axHistCmap, orientation="horizontal", use_gridspec=True, fraction=0.5)
    cbar.ax.set_ylabel('Counts')

    axHist2d.grid()
    axHistx.grid()
    axHisty.grid()

    __set_axis_labels(axHist2d, kwargs.get('axis_labels', ['X', 'Y']))
    __show_save_logic(figure, **kwargs)


def plot_multiple_precision_vs_recall_curves(precisions, recalls, names=[], average_precisions=None, title='Precision vs Recall', **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_SQUARE_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    if kwargs.get('log_scale', None):
        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_xlim(0.1, 110)
        ax.set_ylim(0.1, 110)

        precisions = [100 - precision * 100 for precision in precisions]  #
        recalls = [np.maximum(100 - recall * 100, 0.1) for recall in recalls]  #

        ax.plot([0, 100], [0, 100], color='darkblue', linestyle='--')

    else:
        ax.set_xticks(np.arange(0, 1.01, 0.05))
        ax.set_yticks(np.arange(0, 1.01, 0.05))

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

        ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    names = names if len(names) else np.arange(0, len(precisions))
    if average_precisions is not None:
        names = ["{}|AP:{:.3f}".format(name, AP) for name, AP in zip(names, average_precisions)]

    for i, (precision, recall, name) in enumerate(zip(precisions, recalls, names)):
        linestyle_type = LINESTYLES_LIST[i % len(LINESTYLES_LIST)]
        ax.plot(recall, precision, label=name, linestyle=linestyle_type, color=ALL_COLORS[i])

    plt.legend()
    plt.suptitle(title)

    __show_save_logic(figure, **kwargs)


def plot_precision_vs_recall_curve_triplet(precision, recall, threshs, cls_data=None, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_yticks(np.arange(0, 1.01, 0.05))
    ax.set_xticks(np.arange(0, 1.01, 0.05))

    ax.plot(threshs, precision[:-1], label='precision')
    ax.plot(threshs, recall[:-1], label='recall')
    ax.plot(recall, precision, label='precision_vs_recall')

    if cls_data is not None:
        ax_bar = ax.twinx()

        negative_scores = cls_data[np.where(cls_data[:, 0] == 0)]
        positive_scores = cls_data[np.where(cls_data[:, 0] == 1)]

        ax_bar.hist(negative_scores, align='center', alpha=0.5, width=0.3, color='red')
        ax_bar.hist(positive_scores, align='center', alpha=0.5, width=0.3, color='green')

    ax.legend()

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', [[-0.01, 1.01], [-0.01, 1.01]]))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_multiple_precision_vs_recall_curve_triplets(precisions, recalls, threshs, names=[], **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_SQUARE_PLOT_SIZE), clear=True)

    precision_plot = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=1)
    recall_plot = plt.subplot2grid((3, 1), (1, 0), colspan=1, rowspan=1)
    vs_plot = plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan=1)

    precision_plot.set_yticks(np.arange(0, 1.01, 0.05))
    precision_plot.set_xticks(np.arange(0, 1.01, 0.05))
    precision_plot.grid(True)
    precision_plot.set_ylim(-0.01, 1.01)
    precision_plot.set_xlim(-0.01, 1.01)
    precision_plot.set_xlabel("Threshold")
    precision_plot.set_ylabel("Precision")

    recall_plot.set_yticks(np.arange(0, 1.01, 0.05))
    recall_plot.set_xticks(np.arange(0, 1.01, 0.05))
    recall_plot.grid(True)
    recall_plot.set_ylim(-0.01, 1.01)
    recall_plot.set_xlim(-0.01, 1.01)
    recall_plot.set_xlabel("Threshold")
    recall_plot.set_ylabel("Recall")

    vs_plot.set_yticks(np.arange(0, 1.01, 0.05))
    vs_plot.set_xticks(np.arange(0, 1.01, 0.05))
    vs_plot.grid(True)
    vs_plot.set_ylim(-0.01, 1.01)
    vs_plot.set_xlim(-0.01, 1.01)
    vs_plot.set_xlabel("Recall")
    vs_plot.set_ylabel("Precision")

    names = names if len(names) else np.arange(0, len(precisions))

    for i, (precision, recall, thresh, name) in enumerate(zip(precisions, recalls, threshs, names)):
        marker_index = i % len(LINESTYLES_LIST)

        precision_plot.plot(thresh, precision[:-1], label=name, linestyle=LINESTYLES_LIST[marker_index])
        recall_plot.plot(thresh, recall[:-1], label=name, linestyle=LINESTYLES_LIST[marker_index])
        vs_plot.plot(recall, precision, label=name, linestyle=LINESTYLES_LIST[marker_index])

    precision_plot.set_title('precisions')
    precision_plot.legend()
    recall_plot.set_title('recalls')
    recall_plot.legend()
    vs_plot.set_title('precision_vs_recall')
    vs_plot.legend()

    __show_save_logic(figure, **kwargs)


def plot_roc_curve(fpr, tpr, limit=0, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_SQUARE_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    plt.xticks(np.arange(0, 1.01, 0.05))
    plt.yticks(np.arange(0, 1.01, 0.05))

    plt.xlim(left=-0.025, right=1.025 - limit)
    plt.ylim(bottom=limit - 0.025, top=1.025)

    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['False Positive Rate', 'True Positive Rate']))

    __show_save_logic(figure, **kwargs)


def plot_multiple_roc_curves(fprs, tprs, names, limit=0, **kwargs):
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_SQUARE_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    ax.set_xticks(np.arange(0, 1.01, 0.05))
    ax.set_yticks(np.arange(0, 1.01, 0.05))

    ax.set_xlim(left=-0.025, right=1.025 - limit)
    ax.set_ylim(bottom=limit - 0.025, top=1.025)

    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        linestyle_index = i % len(LINESTYLES_LIST)
        ax.plot(fpr, tpr, linestyle=LINESTYLES_LIST[linestyle_index], linewidth=2)

    ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(names, loc="upper left")

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['False Positive Rate', 'True Positive Rate']))

    __show_save_logic(figure, **kwargs)


# def plot_bars(data_to_show, axis_labels=None, x_tick_labels=None, y_ticks=None, fig_size=DEFAULT_PLOT_SIZE, save_fullname=None, only_save=False):
#
#     assert axis_labels is None or len(axis_labels) == 2, \
#         "axis_labels len {} must be 2 as there are but 2 axis".format(len(axis_labels))
#     assert x_tick_labels is None or len(x_tick_labels) == group_count, \
#         "x_tick_labels len {} must be the same as group_count {}".format(len(x_tick_labels), group_count)
#     assert category_labels is None or len(category_labels) == category_count, \
#         "category_labels len {} must be the same as category_labels {}".format(len(category_labels), category_count)
#
#     width = 0.95 / category_count  # the width of the bars
#
#     figure, ax = plt.subplots(figsize=fig_size, clear=True)
#
#     for category_index in range(category_count):
#         category_label = category_labels[category_index] if category_labels is not None else category_index
#
#         bar_x_positions = x + width * category_index
#         bar_rect = ax.bar(bar_x_positions, grouped_bar_array[..., category_index], width, label=category_label)
#
#         __autolabel_bars(ax, bar_rect)
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     if axis_labels is not None:
#         ax.set_xlabel(axis_labels[0])
#         ax.set_ylabel(axis_labels[1])
#
#     if not isinstance(x, int):
#         ax.set_xticks(x)
#
#     if y_ticks is not None:
#         ax.set_yticks(y_ticks)
#     ax.yaxis.grid(True)
#
#     xticklabels = x_tick_labels if x_tick_labels is not None else np.arange(group_count)
#     ax.set_xticklabels(xticklabels)
#     ax.legend()
#
#     figure.tight_layout()
#
#     __show_save_logic(figure, save_fullname, only_save)


def plot_grouped_bars(grouped_bar_array, x_tick_labels=None, y_ticks=None, category_labels=None, **kwargs):
    """
    Plots bars with their value above them.
    :param grouped_bar_array: An iterable with the following structure:
    [value_1, value_2, ..., value_n]
    :param x_tick_labels:
    :param y_ticks:
    :param category_labels: Labels along x axis. Must be length of n.
    """
    if len(grouped_bar_array.shape) == 1:
        category_count = len(grouped_bar_array)
        group_count = 1
        x = 0
    else:
        group_count, category_count = grouped_bar_array.shape
        x = np.arange(category_count)  # the label locations

    assert x_tick_labels is None or len(x_tick_labels) == group_count, \
        "x_tick_labels len {} must be the same as group_count {}".format(len(x_tick_labels), group_count)
    assert category_labels is None or len(category_labels) == category_count, \
        "category_labels len {} must be the same as category_labels {}".format(len(category_labels), category_count)

    width = 0.95 / category_count  # the width of the bars

    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)

    for category_index in range(category_count):
        category_label = category_labels[category_index] if category_labels is not None else category_index

        bar_x_positions = x + width * category_index
        bar_rect = ax.bar(bar_x_positions, grouped_bar_array[..., category_index], width, label=category_label)

        __autolabel_bars(ax, bar_rect)

    if not isinstance(x, int):
        ax.set_xticks(x)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    ax.yaxis.grid(True)

    xticklabels = x_tick_labels if x_tick_labels is not None else np.arange(group_count)
    ax.set_xticklabels(xticklabels)
    ax.legend()

    figure.tight_layout()

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', [[], [0, 1]]))
    __set_axis_labels(ax, kwargs.get('axis_labels', None))

    __show_save_logic(figure, **kwargs)


def plot_confusion_matrix(confusion_matrix, class_names=None, **kwargs):
    """
    Given a sklearn confusion matrix (cm), make a nice plot
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Taken from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix and modified.
    :param confusion_matrix: confusion matrix from sklearn.metrics.confusion_matrix
    :param class_names: given classification classes such as [0, 1, 2] the class names, for example: ['high', 'medium', 'low']
    """
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_SMALL_SQUARE_PLOT_SIZE), clear=True)
    ax.grid(False)

    accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    misclass = 1 - accuracy

    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names is not None:
        tick_marks = np.arange(len(class_names))

        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names)

        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, rotation='vertical')

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    for i, j in product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        class_text = "{:0.2f}% \n{:,}".format(normalised_confusion_matrix[i, j] * 100, confusion_matrix[i, j])
        text_color = "white" if normalised_confusion_matrix[i, j] > 0.5 else "black"

        ax.text(j, i, class_text, horizontalalignment="center", color=text_color)

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', [
        'Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), 'True label'
    ]))

    __show_save_logic(figure, **kwargs)


# def plot_multiple_confusion_matrices(confusion_matrices, matrix_names=None, class_names=None, columns=3,
#                                      fig_size_per_row=DEFAULT_PLOT_SIZE, save_fullname=None, only_save=False):
#     """
#     Used to work using plot_confusion_matrix() function by passing it axis data, but this functionality is removed and
#     the plot can be done in an easier way if ever needed by replacing the function call with actual cnf mat plotting.
#     """
#     total_plots = len(confusion_matrices)
#     n_columns = min(columns, total_plots)
#     n_rows = np.ceil(total_plots / n_columns).astype(int)
#
#     fig_size = (fig_size_per_row[0], fig_size_per_row[1] * n_rows)
#     figure = plt.figure(figsize=fig_size)
#
#     plot_idx = 0
#
#     for row_idx in range(n_rows):
#         for column_idx in range(n_columns):
#             if plot_idx >= total_plots:
#                 break
#
#             gridded_subplot = plt.subplot2grid((n_rows, n_columns), (row_idx, column_idx))
#             subplot_data = confusion_matrices[plot_idx]
#             matrix_name = matrix_names[plot_idx]
#
#             plot_confusion_matrix(subplot_data, class_names, figure_to_plot_on=(figure, gridded_subplot))
#             gridded_subplot.title.set_text(matrix_name)
#
#             plot_idx += 1
#
#     plt.subplots_adjust(top=0.99, bottom=0.01,
#                         wspace=0.5, hspace=0.5)
#     plt.grid(False)
#     plt.tight_layout()
#
#     __show_save_logic(figure, save_fullname, only_save)


def plot_DET_curves(false_positive_rates, false_negative_rates, curve_names=None, add_analysis=False, **kwargs):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    Taken from: https://github.com/DistrictDataLabs/yellowbrick/issues/453
    """
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)
    ax.grid(True, which='both')

    assert len(false_positive_rates) == len(false_negative_rates), \
        "The lengths of false_positive_rates ({}) and false_negative_rates ({}) do not match.".format(len(false_positive_rates), len(false_negative_rates))

    false_positive_rates = false_positive_rates if type(false_positive_rates) == list else [false_positive_rates]
    false_negative_rates = false_negative_rates if type(false_negative_rates) == list else [false_negative_rates]

    for i, (false_positive_rate, false_negative_rate) in enumerate(zip(false_positive_rates, false_negative_rates)):
        linestyle = LINESTYLES_LIST[i % len(LINESTYLES_LIST)]
        color = ALL_COLORS[i]
        ax.plot(false_positive_rate, false_negative_rate, linestyle=linestyle, linewidth=2, color=color)

        if add_analysis:
            rates_differences = false_positive_rate - false_negative_rate
            min_diff_index = np.argmin(rates_differences)
            min_diff_fp_fn = [false_positive_rate[min_diff_index], false_negative_rate[min_diff_index]]
            min_diff_fp_fn_str = "fp: {:.2f} | fn: {:.2f}".format(min_diff_fp_fn[0], min_diff_fp_fn[1])

            ax.axvline(min_diff_fp_fn[0], label=min_diff_fp_fn_str, linestyle=linestyle, color=color)

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))

    ticks_to_use = np.array([[1, 2, 5]]) * np.array([[0.001], [0.01], [0.1], [1], [10]])
    ticks_to_use = ticks_to_use.flatten()
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)

    if curve_names is not None and len(curve_names) == len(false_positive_rates):
        ax.legend(curve_names)

    __set_log_scale(ax, kwargs.get('log_scale', True))
    __set_axis_limits(ax, kwargs.get('axis_limits', [
        [np.min(ticks_to_use), np.max(ticks_to_use)],
        [np.min(ticks_to_use), np.max(ticks_to_use)]
    ]))
    __set_axis_labels(ax, kwargs.get('axis_labels', ['false_positive_rate (in %)', 'false_negative_rate (in %)']))

    __show_save_logic(figure, **kwargs)


def plot_heatmaps(heatmaps, titles=None, columns=3, add_text=True, grid=False, **kwargs):
    total_plots = len(heatmaps)
    n_columns = min(columns, total_plots)
    n_rows = np.ceil(total_plots / n_columns).astype(int)

    fig_size_per_row = kwargs.get('fig_size_per_row', DEFAULT_PLOT_SIZE)
    fig_size = (fig_size_per_row[0], fig_size_per_row[1] * n_rows)
    figure = plt.figure(figsize=fig_size)
    plt.grid(grid)

    plot_idx = 0
    for row_idx in range(n_rows):
        for column_idx in range(n_columns):
            if plot_idx >= len(heatmaps):
                break

            gridded_subplot = plt.subplot2grid((n_rows, n_columns), (row_idx, column_idx))
            subplot_data = heatmaps[plot_idx]
            sns.heatmap(subplot_data, ax=gridded_subplot, square=True, annot=add_text)

            if titles is not None:
                gridded_subplot.set_title(titles[plot_idx])

            plot_idx += 1

    plt.subplots_adjust(top=0.99, bottom=0.01, wspace=0.5, hspace=0.5)
    plt.tight_layout()

    __show_save_logic(figure, **kwargs)


def plot_nested_dict_bars(nested_dictionary, x_level_labels=None, y_ticks=None, display_legend=True, bar_value_on_top=True, **kwargs):
    """
    Plots bars for each category in the dictionary.
    Code found @ https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts-in-matplotlib/19242176#19242176.
    :param nested_dictionary:
    :param x_level_labels:
    :param y_label:
    :param fig_size:
    :param save_fullname:
    :param only_save:
    :return:
    """
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_ylim(min(y_ticks), max(y_ticks) * 1.01)

    __nested_bar(ax, nested_dictionary, x_level_labels, bar_value_on_top)
    figure.subplots_adjust(bottom=0.3)

    if display_legend:
        plt.legend()

    __set_axis_labels(ax, kwargs.get('axis_labels', None))
    __show_save_logic(figure, **kwargs)


def plot_boxplots(grouped_values, x_tick_labels=None, **kwargs):
    """
    Plot box plots for each bin.

    Parameters:
    binned_errors (dict): Dictionary with bin ranges as keys and lists of absolute errors for each bin.
    """
    figure, ax = plt.subplots(figsize=kwargs.get('fig_size', DEFAULT_PLOT_SIZE), clear=True)

    plt.boxplot(grouped_values, labels=x_tick_labels, vert=True, patch_artist=True)

    plt.xticks(rotation=45, ha="right")

    __set_log_scale(ax, kwargs.get('log_scale', None))
    __set_axis_limits(ax, kwargs.get('axis_limits', None))
    __set_axis_labels(ax, kwargs.get('axis_labels', ["Ground Truth Bins", "Error rates"]))

    __show_save_logic(figure, **kwargs)

# def plot_dist_dict(dist_dict, axis_labels, columns=2, plot_limit=None, **kwargs):
#     """
#     Not sure why this function is so complicated, it seems like it's a simple scatter plot with a name at each point.
#     Can be implemented in a much easier way with ax.scatter(x[i], y[i]) + ax.annotate(names[i], x[i], y[i])
#     Plots named scatter plot.
#     :param dist_dict: A dictionary of the following structure:
#     {
#         point_name:
#         {'dataset_size': size_of_the_point, axis_labels[0]: value_along_x_axis, axis_labels[1]: value_along_y_axis},
#         ...
#     }
#     :param axis_labels: Axis labels [x, y].
#     :param columns:
#     :param plot_limit:
#     """
#     total_plots = len(axis_labels) * (len(axis_labels) - 1) // 2
#     n_columns = int(min(columns, total_plots))
#     n_rows = int(np.ceil(total_plots / n_columns))
#
#     fig_size_per_row = kwargs.get('fig_size_per_row', DEFAULT_SQUARE_PLOT_SIZE)
#     figure, ax = plt.subplots(figsize=(fig_size_per_row[0], fig_size_per_row[1] * n_rows), clear=True)
#
#     for i, (axis_label_1, axis_label_2) in enumerate(combinations(axis_labels, 2)):
#         subplot = figure.add_subplot(n_rows, n_columns, i + 1)
#         max_distance = 0
#
#         for (dataset_name, dataset_dists_dict), color in zip(dist_dict.items(), ALL_COLORS):
#             if np.isnan(dataset_dists_dict[axis_label_1]):
#                 continue
#
#             dataset_size = dataset_dists_dict['dataset_size']
#
#             subplot.scatter(dataset_dists_dict[axis_label_1], dataset_dists_dict[axis_label_2], s=np.log2(dataset_size) * 6, color=color)
#             subplot.annotate(dataset_name, (dataset_dists_dict[axis_label_1], dataset_dists_dict[axis_label_2]))
#
#             this_max_distance = max(np.max(dataset_dists_dict[axis_label_1]), np.max(dataset_dists_dict[axis_label_1]))
#             max_distance = this_max_distance if this_max_distance > max_distance else max_distance
#
#         subplot.set_xlabel(axis_label_1)
#         subplot.set_ylabel(axis_label_2)
#         subplot.grid(True)
#
#         subplot.set_ylim(0, plot_limit)
#         subplot.set_xlim(0, plot_limit)
#
#     __set_log_scale(ax, kwargs.get('log_scale', None))
#     __set_axis_limits(ax, kwargs.get('axis_limits', [[], [-0.05, 1.05]]))
#     __set_axis_labels(ax, kwargs.get('axis_labels', ['Threshold', 'Error']))
#
#     __show_save_logic(figure, **kwargs)


def __make_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = __make_groups(value)

        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))

            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups

    return [thisgroup] + groups


def __add_separating_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos], transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def __nested_bar(ax, data, x_level_labels=None, bar_value_on_top=True):
    groups = __make_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = list(range(1, ly + 1))
    current_level = 0

    x_array = np.array(x)
    y_array = np.array(y)
    xticks_array = np.array(xticks)

    for i, unique_x in enumerate(set(x)):
        x_indice = np.argwhere(x_array == unique_x)
        xticks_based_on_x_category = np.squeeze(xticks_array[x_indice])
        y_based_on_x_category = np.squeeze(y_array[x_indice])
        color = ALL_COLORS[i]

        bar_rects = ax.bar(xticks_based_on_x_category, y_based_on_x_category, align='center', label=unique_x, color=color)

        if bar_value_on_top:
            __autolabel_bars(ax, bar_rects)

    ax.set_xticks(xticks)
    ax.set_xticklabels(x)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)

    if x_level_labels is not None:
        ax.text(-0.05, -0.045, x_level_labels[current_level], ha='left', transform=ax.transAxes)  # set label

    scale = 1. / ly
    for pos in range(ly + 1):
        __add_separating_line(ax, pos * scale, -.1)

    ypos = -.2
    while groups:
        group = groups.pop()
        current_level += 1

        pos = 0
        for i, (label, rpos) in enumerate(group):
            lxpos = (pos + .5 * rpos) * scale

            # if this is the last level, display labels in nearby parallel lines. This helps display the labels without them obscuring each other.
            text_y_pos = ypos - 0.04 if len(groups) == 0 and i % 2 == 1 else ypos
            ax.text(lxpos, text_y_pos, label, ha='center', transform=ax.transAxes)

            __add_separating_line(ax, pos * scale, ypos)
            pos += rpos

        __add_separating_line(ax, pos * scale, ypos)

        if x_level_labels is not None:
            ax.text(-0.05, ypos, x_level_labels[current_level], ha='left', transform=ax.transAxes)

        ypos -= .1


def __autolabel_bars(ax, bar_rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for bar_rect in bar_rects:
        height = bar_rect.get_height()
        string_template = integer_string_template if isinstance(height, int) or isinstance(height, np.int) else float_string_template

        ax.annotate(string_template.format(height),
                    xy=(bar_rect.get_x() + bar_rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



def _get_plot_ticks(min_value, max_value, ticks_count, values_dtype):
    if np.issubdtype(values_dtype, np.integer):
        ticks_type = int
        ticks_count = ticks_count if (max_value - min_value) / ticks_count >= 1 else max_value - min_value
        ticks_step = np.round((max_value - min_value) / ticks_count)

    else:
        ticks_type = float
        ticks_step = (max_value - min_value) / ticks_count

    ticks = np.arange(min_value, max_value + ticks_step, ticks_step, dtype=ticks_type)

    return ticks


# def _calculate_plot_limits(values, offset_coef=0.05):
#     min_values, max_values = np.min(values), np.max(values)
#     total_range_offset = (max_values - min_values) * offset_coef
#
#     plot_min = min_values - total_range_offset
#     plot_max = max_values + total_range_offset
#
#     return min_values, max_values, plot_min, plot_max