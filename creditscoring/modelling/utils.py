from creditscoring.settings import Settings
from joblib import load
s = Settings()

def load_model():
	return load(s.path_final_model)