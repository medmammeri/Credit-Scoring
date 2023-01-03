import pandas as pd
from sklearn.pipeline import Pipeline
from creditscoring.settings import Settings
from creditscoring.modelling.utils import load_model
import numpy as np

s = Settings()


def inference(data: pd.DataFrame, model: Pipeline = None):
    """
    func
    inference
    :param data:
    :param model:
    :return:
    """
    if model is None:
        model = load_model()
    return model.predict_proba(data)
