import torch
from pathlib import Path
from typing import Dict, Tuple
import os


def DelfileList(path, filestarts='checkpoint_last'):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(filestarts):
                os.remove(os.path.join(root, file))

def load_checkpoint(filepath: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    return state_dict


class Register(dict):
    def __init__(self):
        super().__init__()
        self._dict = {}
    
    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")
            if key in self._dict:
                print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
            self[key] = value
            return value
        
        if callable(target):
            return add_item(target.__name__, target)
        else:
            return lambda x: add_item(target, x)
        