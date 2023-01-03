from creditscoring.settings import Settings
from sklearn.model_selection import train_test_split
import pandas as pd


s = Settings()


def read_data():
    train = pd.read_csv(s.path_train, index_col=0)
    test = pd.read_csv(s.path_test, index_col=0)
    return train, test


def get_modelling_data():
    train, _ = read_data()
    X = train.drop(columns=s.target)
    y = train[s.target]
    return train_test_split(X, y)
