import logging
import pickle
import socket
import time
from datetime import datetime

import pandas as pd
import requests
from flask import Flask
from flask import jsonify
from flask import request

from transaction_risk_profiler.preprocessing.preprocessing import DataPreProcessing

logger = logging.getLogger(__name__)
app = Flask(__name__)
PORT = 8080
REGISTER_URL = "http://10.6.80.122:5000/register"
DATA = []
TIMESTAMP = []
PREDICTIONS = []
PROBABILITIES = []
MODEL_PATH = "random_forest.pkl"
LOG_FILE_PATH = "app_log.txt"
FRAUD_THRESHOLD = 0.5


@app.route("/")
def uhhhhhh():
    return "Churning Sensation don't need no instructions to know how to ROCK!"


@app.route("/score", methods=["POST"])
def score():
    DATA.append(jsonify(request.json))
    TIMESTAMP.append(time.time())
    return ""


@app.route("/check2")
def check2():
    line1 = f"Number of data points: {len(DATA)}"
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        line2 = f"Latest datapoint received at: {data_time}"
        line3 = DATA[-1]
        output = f"{line1}\n\n{line2}\n\n{line3}"
    else:
        output = line1
    return output, 200, {"Content-Type": "text/css; charset=utf-8"}


@app.route("/check")
def check():
    if len(PREDICTIONS) != len(DATA):
        temp = DATA[len(PREDICTIONS) - len(DATA)]
        for obv in temp:
            pred = predict_obv(obv, model)
            PROBABILITIES.append(pred)
            if pred < FRAUD_THRESHOLD:
                PREDICTIONS.append("Quick, look out, it's FRAUD!")
            else:
                PREDICTIONS.append("No stress, you're good Mr. Premium")
            with open(LOG_FILE_PATH, "a") as f:
                f.write(obv)
                f.write(FRAUD_THRESHOLD)
                f.write(pred)
    string = ""
    for i, pred in enumerate(PREDICTIONS):
        string = string + str(i) + ": " + str(pred) + "\n"
    return string


def register_for_ping(ip, port):
    registration_data = {"ip": ip, "port": port}
    requests.post(REGISTER_URL, data=registration_data)


def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model


def predict_obv(point, model):
    df = pd.Series(point.encode("utf-8")).to_frame().T
    df.columns = point.keys()
    obs = DataPreProcessing(df)
    return model.predict_proba(obs)[:, 1]


if __name__ == "__main__":
    ip_address = socket.gethostbyname(socket.gethostname())
    logger.info("attempting to register %s:%d") % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    model = load_model()

    app.run(host="0.0.0.0", port=PORT, debug=True)
