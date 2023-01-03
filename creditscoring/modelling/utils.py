from joblib import load

from creditscoring.settings import Settings

s = Settings()


def load_model():
    return load(s.path_final_model)
