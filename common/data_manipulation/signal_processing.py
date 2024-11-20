import numpy as np
from common.miscellaneous import verbose_print
from matplotlib import pyplot as plt


def diff_sums(binary_map, axis, smoothing=1):
    sums = np.sum(binary_map, axis=axis)

    ma_sums = moving_average(sums, smoothing) if smoothing > 1 else sums

    differences = ma_sums[1:] - ma_sums[:-1]

    return differences, sums, ma_sums


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def merge_tendencies(signal_rises, signal_falls):
    merging_falls = signal_falls.copy()
    merging_falls[:, 2] *= -1

    tendencies = np.concatenate([signal_rises, merging_falls])
    tendencies = tendencies[tendencies[:, 0].argsort()]

    return tendencies


def join_sequential_tendencies(tendencies):
    positive_indice = np.squeeze(np.argwhere(tendencies[:, 2] > 0), axis=1)
    negative_indice = np.squeeze(np.argwhere(tendencies[:, 2] < 0), axis=1)

    # joined_positive = []
    # if positive_indice:
    #     if len(positive_indice) > 1:
    #         joined_positive = join_signed_sequential_tendencies(tendencies, positive_indice)
    #     else:
    #         joined_positive = tendencies[positive_indice[0]]
    #
    # joined_negative = []
    # if negative_indice:
    #     if len(negative_indice) > 1:
    #         joined_negative = join_signed_sequential_tendencies(tendencies, negative_indice)
    #     else:
    #         joined_negative = tendencies[negative_indice[0]]

    joined_positive = join_signed_sequential_tendencies(tendencies, positive_indice) if len(positive_indice) else None
    joined_negative = join_signed_sequential_tendencies(tendencies, negative_indice) if len(negative_indice) else None

    if joined_positive is not None and joined_negative is not None:
        joined_tendencies = np.concatenate([joined_positive, joined_negative])
    elif joined_positive is not None:
        joined_tendencies = np.array(joined_positive)
    elif joined_negative is not None:
        joined_tendencies = np.array(joined_negative)
    else:
        return None

    joined_tendencies = joined_tendencies[np.argsort(joined_tendencies[:, 0])]

    return joined_tendencies


def join_signed_sequential_tendencies(tendencies, signed_indice):
    # print("signed_indice", signed_indice, signed_indice.size)
    if signed_indice.size == 0:
        return None
    elif signed_indice.size == 1:
        return [tendencies[signed_indice[0]]]

    signed_indice_diffs = signed_indice[1:] - signed_indice[:-1]

    joined_tendencies = []
    joined_tendancy = None

    for i, signed_indice_diff in enumerate(signed_indice_diffs):
        if signed_indice_diff == 1:
            if joined_tendancy is None:
                tendancy_start, _, tendency_1_sum = tendencies[signed_indice[i]]
                _, tendancy_end, tendency_2_sum = tendencies[signed_indice[i + 1]]

                joined_tendancy = [tendancy_start, tendancy_end, tendency_1_sum + tendency_2_sum]
            else:
                _, tendancy_end, tendency_2_sum = tendencies[signed_indice[i + 1]]
                joined_tendancy[1] = tendancy_end
                joined_tendancy[2] += tendency_2_sum

        else:
            if joined_tendancy is not None:
                joined_tendencies.append(joined_tendancy)
                joined_tendancy = None
            else:
                joined_tendencies.append(tendencies[signed_indice[i]])

    if joined_tendancy is not None:
        #         print("last tendency not saved. Saving", joined_tendancy)
        joined_tendencies.append(joined_tendancy)

    if signed_indice_diffs[-1] != 1:
        #         print("last tendency asd", tendencies[signed_indice[-1]])
        joined_tendencies.append(tendencies[signed_indice[-1]])

    joined_tendencies = np.array(joined_tendencies)

    return joined_tendencies


def locate_signal_rises(differences, fall_tolerance, minimum_total_rise, smoothing, verbose=0):
    signal_rises = []
    is_rising = False
    rise_total = 0

    last_rise = []
    last_rising_index = -1
    for i in range(len(differences)):
        difference = differences[i]
        if difference > 0:
            rise_total += difference
            last_rising_index = i

            if not is_rising:
                # Init rising
                verbose_print("{} - Tendency start".format(i), verbose, 8)
                last_rise = [i + smoothing - 1, 0, 0]
                is_rising = True

        else:
            if is_rising:
                if fall_tolerance <= difference:
                    # Tolerate fall
                    verbose_print("{} - Tolerate fall: {}".format(i, difference), verbose, 8)
                    rise_total += difference
                else:
                    # Stop rising
                    verbose_print("{} - Tendency Stop: {} > {}".format(i, rise_total, minimum_total_rise), verbose, 8)
                    if rise_total > minimum_total_rise:
                        # Save rise if big enough
                        last_rise[1] = last_rising_index + smoothing
                        last_rise[2] = rise_total

                        signal_rises.append(last_rise)

                    # Reset rising
                    is_rising = False
                    last_rise = []
                    rise_total = 0

    if is_rising and rise_total > minimum_total_rise:
        # Save last rise if big enough
        last_rise[1] = last_rising_index + smoothing
        last_rise[2] = rise_total

        signal_rises.append(last_rise)

    signal_rises = np.array(signal_rises).astype(int)

    return signal_rises


def plot_tendencies(sums, ma_sums=None, rises=None, falls=None, tendencies=None, fig_size=(22, 5)):
    plt.figure(figsize=fig_size)
    plt.plot(sums, linewidth=2, label='sums')
    if ma_sums is not None and len(ma_sums) > 0:
        plt.plot(ma_sums, label='ma_sums')

    if rises is None and tendencies is not None:
        rises = tendencies[tendencies[:, 2] > 0]
    if falls is None and tendencies is not None:
        falls = tendencies[tendencies[:, 2] < 0]

    if rises is not None and len(rises) > 0:
        plt.axvline(x=rises[0, 0], color='black', linestyle='--', label='rises start')
        plt.axvline(x=rises[0, 1], color='green', linestyle='--', label='rises end')
        for rise in rises:
            plt.axvline(x=rise[0], color='black', linestyle='--')
            plt.axvline(x=rise[1], color='green', linestyle='--')

    if falls is not None and len(falls) > 0:
        plt.axvline(x=falls[0, 0] + 0.2, color='black', linestyle='-', label='falls start')
        plt.axvline(x=falls[0, 1] + 0.2, color='green', linestyle='-', label='falls end')
        for fall in falls:
            plt.axvline(x=fall[0] + 0.2, color='black', linestyle='-')
            plt.axvline(x=fall[1] + 0.2, color='green', linestyle='-')

    plt.grid()
    plt.legend()
    plt.show()
