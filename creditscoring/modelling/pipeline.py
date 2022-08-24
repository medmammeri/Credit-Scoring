from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from creditscoring.settings import Settings
from creditscoring.modelling.preprocessing import get_modelling_data
import time
from sklearn.model_selection import GridSearchCV
from creditscoring.modelling.metrics import scorer

s = Settings()

x_train, x_test, y_train, y_test = get_modelling_data()




def get_pipeline():
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                           ("scaler", StandardScaler()),
                           ("classifier", RandomForestClassifier())
                           ]
                    )
param_grid = {
    "imputer__strategy": ["mean"],
    "classifier__max_leaf_nodes": (10, 30),
    "classifier__bootstrap": [True, False],
    "classifier__max_depth": [40, 50],
    "classifier__min_samples_leaf": [2, 4],
    "classifier__min_samples_split": [5],
    "classifier__n_estimators": [400],
}

def train_model(save: bool = True):
    x_train, x_test, y_train, y_test = get_modelling_data()
    pipeline = get_pipeline()
    model_grid_search = GridSearchCV(
        pipeline, param_grid=param_grid, n_jobs=-1, cv=5, scoring=scorer()
    )
    model_grid_search.fit(x_train, y_train)
    model = model_grid_search.best_estimator_
    if save:
        model_name = time.strftime("%Hh%Mm%Ss_%d-%m-%Y") + ".joblib"
        joblib.dump(model, s.path_models / model_name)
    return model

if __name__ == "__main__":
    train_model(save=True)