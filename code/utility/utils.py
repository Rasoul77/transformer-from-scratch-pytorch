import os
import random
import time

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


class DotDict:
    """A dictionary wrapper that allows dot notation access to keys."""

    def __init__(self, dictionary: dict):
        self._data = {k: self._convert(v) for k, v in dictionary.items()}

    def _convert(self, obj: Any):
        if isinstance(obj, Mapping):
            return DotDict(obj)
        elif isinstance(obj, list):
            return [self._convert(item) for item in obj]
        return obj

    def __getattr__(self, name: str):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f"DotDict({self._data})"

    def to_dict(self):
        """Recursively convert back to a plain dictionary."""

        def _convert_back(obj):
            if isinstance(obj, DotDict):
                return {k: _convert_back(v) for k, v in obj._data.items()}
            elif isinstance(obj, list):
                return [_convert_back(item) for item in obj]
            else:
                return obj

        return _convert_back(self)


def get_seed(config: dict) -> int:
    """
    Get a random seed from config, or generate one if not provided.
    """
    random_seed = config.get("seed", None)
    return (
        random.randint(0, 4294967295)
        if random_seed is None or random_seed < 0
        else random_seed
    )


def set_random_seed(seed: int, deterministic: bool) -> None:
    """
    Set random seed for reproducibility across common libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def get_formatted_time_execution(start_time: float) -> str:
    """
    Return execution time as a formatted string mm:ss.
    """
    execution_time = time.time() - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    return f"{minutes:02d}:{seconds:02d}"


def read_number_from_file(variable_name: str, root_path: str, default: float) -> float:
    """
    Read a float number from a text file or return the default value.
    """
    file_path = os.path.join(root_path, f"{variable_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as infile:
            return float(infile.readline())
    return default


def save_number_to_file(variable_name: str, root_path: str, value: float) -> None:
    """
    Save a float number to a text file.
    """
    file_path = os.path.join(root_path, f"{variable_name}.txt")
    with open(file_path, "w") as outfile:
        outfile.write(str(value))


def save_model(model: torch.nn.Module, save_file_name: str, output_path: str) -> None:
    """
    Save model parameters to file.
    """
    torch.save(model.state_dict(), os.path.join(output_path, save_file_name))
