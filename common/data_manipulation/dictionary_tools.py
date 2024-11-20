from typing import Any, Union, List, Dict


def create_or_append_to_dict_of_lists(dict_of_lists: Dict[Any, Any], keys: Union[Any, List[Any]], value: Any = 1) -> Dict[Any, List[Any]]:
    """
    Either adds the value to the dictionary or creates an instance in it with the value.
    :param dict_of_lists:
    :param key:
    :param value:
    :return: dict_of_lists
    """
    if type(keys) == list and len(keys) > 1:
        if keys[0] not in dict_of_lists:
            dict_of_lists[keys[0]] = {}

        create_or_append_to_dict_of_lists(dict_of_lists[keys[0]], keys[1:], value)

    else:
        key = keys[0] if type(keys) == list else keys
        if key in dict_of_lists:
            dict_of_lists[key] += [value]

        else:
            dict_of_lists[key] = [value]

    return dict_of_lists


def nested_dict_add_element(dictionary: Dict[Any, Any], keys: List[Any], value: Any = 1) -> None:
    """
    Either adds the value to the dictionary or creates an instance in it with the value.
    :param dictionary:
    :param keys:
    :param value:
    :return: dictionary
    """
    if len(keys) == 1:
        dictionary[keys[0]] = value
    else:
        if keys[0] not in dictionary:
            dictionary[keys[0]] = {}

        nested_dict_add_element(dictionary[keys[0]], keys[1:], value)


def create_or_add_to_dict(dictionary: Dict[Any, Any], key: Any, value: Any = 1) -> Dict[Any, List[Any]]:
    if key in dictionary:
        dictionary[key] += value

    else:
        dictionary[key] = value

    return dictionary


def convert_list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    """
    https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    :param list_of_dicts:
    :return:
    """
    return {key: [dict[key] for dict in list_of_dicts] for key in list_of_dicts[0]}


def convert_dict_lists_to_list_of_dicts(dict_of_lists: Dict[Any, List[Any]]) -> List[Dict[Any, Any]]:
    """
    https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    :param dict_of_lists:
    """
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]