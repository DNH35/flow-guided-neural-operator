#from .raw_wav_file_dataset import RawWavFileDataset
#from .debug_dataset import DebugDataset
#
#__all__ = ["RawWavFileDataset",
#           "DebugDataset"]

import importlib
import os
from pathlib import Path

DATASET_REGISTRY = {}

__all__ = ["build_dataset"]

def build_dataset(cfg, *args, **kwargs):
    dataset_name = cfg.name
    assert dataset_name in DATASET_REGISTRY
    dataset = DATASET_REGISTRY[dataset_name](cfg, *args, **kwargs)
    return dataset

def register_dataset(name):
    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f'{name} already in registry')
        else:
            DATASET_REGISTRY[name] = cls
        return cls
    return register_dataset_cls

def import_datasets():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("_"):
            module_name = str(Path(file).with_suffix(""))
            try:
                importlib.import_module('.' + module_name, package=__name__)
            except ImportError as e:
                print(f"data.BBT_data.datasets: skipping '{module_name}' ({e})")

import_datasets()
