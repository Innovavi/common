import os
import re
import numpy as np

from typing import List


def filter_string_list_based_on_substrings(full_string_list: List[str], substrings_to_filter: List[str], filter_out: bool = True) -> List[str]:
    if filter_out:
        filtered_string_list = [long_string for long_string in full_string_list
                                if not does_string_contain_substring_list(long_string, substrings_to_filter, do_any=True)]
    else:
        filtered_string_list = [long_string for long_string in full_string_list
                                if does_string_contain_substring_list(long_string, substrings_to_filter, do_any=True)]

    return filtered_string_list


def get_files_list_in_dir_with_substrings(directory: str, strings_to_find: List[str], do_all: bool = True) -> List[str]:
    files_list = [file_name for file_name in os.listdir(directory)
                  if does_string_contain_substring_list(file_name, strings_to_find, do_any=not do_all, do_all=do_all)]

    return files_list


def does_string_contain_substring_list(string: str, substring_list: List[str], do_any: bool = False, do_all: bool = False) -> bool:
    assert do_any or do_all, "One of do_any or do_all must be set to True"

    if do_any:
        return any([substring in string for substring in substring_list])

    elif do_all:
        return all([substring in string for substring in substring_list])

    return False


def sort_strings_by_number_values(string_list):
    string_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    return string_list


def filter_out_tuples_with_string(analysis_files_triplets, string_filter, filter_index=1, verbose=0):
    filtered_array = np.array(list(filter(lambda x: not string_filter in x[filter_index], analysis_files_triplets)))

    if verbose > 1:
        print("Filtered out {} rows out of {} with filter: {}".format(len(analysis_files_triplets) - len(filtered_array), len(analysis_files_triplets), string_filter))

    return filtered_array


def filter_out_tuples_without_string(analysis_files_triplets, string_filter, filter_index=1, verbose=0):
    filtered_array = np.array(list(filter(lambda x: string_filter in x[filter_index], analysis_files_triplets)))

    if verbose > 1:
        print("Filtered out {} rows out of {} with filter: {}".format(len(analysis_files_triplets) - len(filtered_array), len(analysis_files_triplets), string_filter))

    return filtered_array

