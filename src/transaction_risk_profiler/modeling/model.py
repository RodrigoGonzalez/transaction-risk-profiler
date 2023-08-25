import logging
import pickle
from os import path

from build_model import build_model

logger = logging.getLogger(__name__)


def load_model(data_filename="data/train.json", model_filename="model.pkl"):
    if path.isfile(model_filename):
        with open(model_filename) as f:
            model = pickle.load(f)
    else:
        model = build_model(data_filename, model_filename)
    return model
