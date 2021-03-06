# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_utils.ipynb (unless otherwise specified).

__all__ = ['flatten_dict', 'most_common', 'get_node', 'apply_nested', 'resolve_path', 'set_dir', 'generate_time_id']

# Cell

from pathlib import Path
from fastcore.basics import patch

# Cell

@patch
def ls_sorted(self:Path):
    "ls but sorts files by name numerically"
    return self.ls().sorted(key=lambda f: int(f.with_suffix('').name))

# Cell

def flatten_dict(d: dict):
    """flattens a nested dict one level"""
    def func(dct):
        for k, v in dct.items():
            if isinstance(v, dict):
                yield from v.items()
            else:
                yield k, v
    return dict(func(d))

# Cell
from collections import Counter

def most_common(lst):
    return Counter(lst).most_common(1)[0][0]

# Cell

def get_node(tree: dict, path: str, sep: str = '.'):
    if path is None or path == '':
        return tree
    fields = path.split(sep)
    node = tree
    for field in fields:
        node = node[field]
    return node


# Cell
def apply_nested(tree: dict, path: str, func, sep: str = '.'):
    parts = path.split(sep)
    parent_node = get_node(tree, sep.join(parts[:-1]))
    parent_node[parts[-1]] = func(parent_node[parts[-1]])
    return tree

# Cell
from pathlib import Path

def resolve_path(config, field_path, sep='.'):
    func = lambda s: str(Path(s).resolve())
    return apply_nested(config, field_path, func, sep)

# Cell

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union

# ref: https://dev.to/teckert/changing-directory-with-a-python-context-manager-2bj8
@contextmanager
def set_dir(path: Union[Path, str]):
    """Sets the cwd within the context"""
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

# Cell
def generate_time_id():
    from datetime import datetime
    return datetime.now().isoformat().rsplit('.', 1)[0].replace(':', '-')