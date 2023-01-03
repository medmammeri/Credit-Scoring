import pandas as pd
from sklearn.pipeline import Pipeline

from creditscoring.modelling.utils import load_model
from creditscoring.settings import Settings

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
