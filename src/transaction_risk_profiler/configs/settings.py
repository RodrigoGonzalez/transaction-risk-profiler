""" Settings for the app. """
from pydantic_settings import BaseSettings

from transaction_risk_profiler import APP_DIR
from transaction_risk_profiler import PROJECT_DIR


class Settings(BaseSettings):
    """Settings for the app"""

    TITLE: str = "Transaction Risk Profiler"
    DESCRIPTION: str = "A simple transaction risk profiler application."
    APP_DIRECTORY: str = APP_DIR
    PROJECT_DIRECTORY: str = PROJECT_DIR
    TRAINED_MODEL_DIRECTORY: str = "trained_models"
    DATASET_DIRECTORY: str = "data"
    DATASET_FILENAME: str = "transactions.json"
    DATASET_SUBSET_FILENAME: str = "transactions_subset.json"
    DATASET_CARD = "dataset_card.yml"

    # The following are used for data preprocessing and model training
    GLOBAL_RANDOM_SEED: int = 1234

    class Config:
        """
        Configuration for the Settings class.
        """

        case_sensitive = True
