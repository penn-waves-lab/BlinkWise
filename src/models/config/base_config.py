import copy
import json
import shutil

from pathlib import Path
from typing import Union


def dict_to_markdown_table(dict_data):
    # Find the maximum length of keys and values
    max_key_length = max(len(str(key)) for key in dict_data.keys())
    max_value_length = max(len(str(value)) for value in dict_data.values())

    # Construct the Markdown table header
    markdown_table = f"| {'Key'.center(max_key_length)} | {'Value'.center(max_value_length)} |\n"
    markdown_table += f"| {'-' * max_key_length} | {'-' * max_value_length} |\n"

    # Construct the Markdown table rows
    for key, value in dict_data.items():
        markdown_table += f"| {str(key).center(max_key_length)} | {str(value).center(max_value_length)} |\n"

    return markdown_table


def reverse_nested_sequence(seq):
    """
    recursively reverse nested sequences.
    """
    # reverse the outer sequence
    reversed_seq = seq[::-1]
    # recursively reverse any inner sequences
    return [reverse_nested_sequence(inner) if isinstance(inner, tuple) or isinstance(inner, list) else inner for inner
            in reversed_seq]


def remove_all_contents(folder_path: Union[str, Path]):
    """
    remove the specified folder and all its contents.
    """
    path = Path(folder_path)
    if path.exists():
        shutil.rmtree(path)


# ===================
# the base class for all configs
# ===================

class BaseConfig:
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_dict(self):
        keys = sorted(self.__dict__)
        sorted_dict = {k: self.__dict__[k] for k in keys}
        return sorted_dict

    @classmethod
    def load(cls, json_file):
        with open(json_file, 'r') as config_file:
            config_dict = json.load(config_file)
        new_instance = cls()
        new_instance.__dict__ = config_dict
        return new_instance

    def save(self, json_file):
        with open(json_file, 'w') as config_file:
            json.dump(self.__dict__, config_file)

    def copy(self):
        return copy.deepcopy(self)
