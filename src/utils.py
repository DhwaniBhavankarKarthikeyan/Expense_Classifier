
import pandas as pd
import os

def load_csv(file_obj_or_path):
    if hasattr(file_obj_or_path, 'read'):
        return pd.read_csv(file_obj_or_path)
    else:
        return pd.read_csv(file_obj_or_path)

def ensure_models_dir():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir
