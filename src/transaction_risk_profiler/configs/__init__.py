""" Configs module. """
from dotenv import load_dotenv

from transaction_risk_profiler.configs.settings import Settings

load_dotenv()

settings = Settings()


__all__ = ["settings"]
