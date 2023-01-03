# import enum
# import logging
# import os
import pathlib
# import typing
from typing import Union

import pydantic


class Settings(pydantic.BaseSettings):
    allowed_numbers: int = 3

    class Config:
        # .env variables must have following prefix
        env_prefix = "CREDIT_"
        env_file = pathlib.Path(__file__).parent / ".env"
        # working

    path_workdir: pathlib.Path = pathlib.Path(__file__).parent.parent
    path_models: pathlib.Path = path_workdir / "models"
    final_model_name: str = "23h33m43s_24-08-2022.joblib"
    path_final_model: pathlib.Path = path_models / final_model_name

    path_data: pathlib.Path = ".data"
    name_train_data: str = "cs-training.csv"
    name_test_data: str = "cs-test.csv"
    target: str = "SeriousDlqin2yrs"

    @property
    def path_train(self):
        return self.path_data / self.name_train_data

    @property
    def path_test(self):
        return self.path_data / self.name_test_data

    path_docs: pathlib.Path = path_workdir / "docs"
    path_mlruns: pathlib.Path = path_workdir / "mlruns"
    mlflow_tracking_uri: Union[str, pathlib.Path] = f"file:{path_mlruns}"
