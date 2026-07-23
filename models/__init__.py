import importlib
import os
from pathlib import Path

MODEL_REGISTRY = {}

__all__ = ["build_model"]

def build_model(cfg, *args, **kwargs):
    model_name = cfg.name
    print("MODEL NAME: ", model_name)
    print(MODEL_REGISTRY)
    assert model_name in MODEL_REGISTRY
    model = MODEL_REGISTRY[model_name]()
    model.build_model(cfg, *args, **kwargs)
    return model

def register_model(name):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f'{name} already in registry')
        else:
            MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls

def import_models():
    for file in os.listdir(os.path.dirname(__file__)):
        if file.endswith(".py") and not file.startswith("_"):
            module_name = str(Path(file).with_suffix(""))
            try:
                importlib.import_module('models.'+module_name)
            except ImportError as e:
                # A handful of legacy models pull in optional, task-specific
                # dependencies (e.g. h5py for Brain Treebank). Skipping a
                # module that fails to import keeps the registry usable for
                # every other model instead of hard-failing the whole package.
                print(f"models: skipping '{module_name}' ({e})")
import_models()
