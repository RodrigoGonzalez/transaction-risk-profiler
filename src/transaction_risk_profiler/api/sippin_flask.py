import json
import logging
import socket
import time
from datetime import datetime

import graphlab as gl
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
MODEL_PATH = "boosted_tree.model"
LOG_FILE_PATH = "app_log.txt"
FRAUD_THRESHOLD = 0.5


@app.route("/")
def uhhhhhh():
    return "Churning Sensation don't need no instructions to know how to ROCK!"


@app.route("/score", methods=["POST"])
def score():
    DATA.append(jsonify(request.json))
    # , sort_keys=True, indent=4, separators=(',', ': '))
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
        # with open('test.json', 'w') as f:
        # f.write(line3)

        # o = gl.SFrame.read_json(line3)
        output = f"{line1}\n\n{line2}\n\n{line3}"
    else:
        output = line1
        # o=line1
        # o = gl.SFrame.read_json(output)

    return output, 200, {"Content-Type": "text/css; charset=utf-8"}


@app.route("/check")
def check():
    # predict model based on Data[-new_entry_counter:]
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
    """
    Function to register this computer with server holding additional observations"""
    registration_data = {"ip": ip, "port": port}
    requests.post(REGISTER_URL, data=registration_data)


def load_model():
    """
    Loads previously pickled model to be used for predicitons
    """
    return gl.load_model(MODEL_PATH)


def predict_obv(point, model):
    """
    FROM CULLY AND ANUSHA
    """
    p = json.dumps(point, sort_keys=True, indent=4, separators=(",", ": "))
    c = p.keys().values
    # p = json.loads(point)
    # p = gl.SFrame.([point)
    # raw_json = json.loads(point)
    # raw_array = np.array(raw_json.values())
    # raw_array = raw_array.reshape(1,-1)
    # c = point.decode('ascii','ignore').keys()
    df = pd.Series(point.encode("utf-8")).to_frame().T
    df.columns = c
    # cols = df.columns
    # print(cols)
    # new_cols = []
    # for col in cols:
    #     new_cols.append(col.encode('utf-8'))
    # df.columns = new_cols
    # print(type(point))
    # df.columns = point.keys()
    # df = pd.DataFrame(data=raw_array, columns = raw_json.keys())
    obs = DataPreProcessing(df)
    return model.predict(obs.sf, output_type="probability")


if __name__ == "__main__":
    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    logger.info("attempting to register %s:%d") % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    # load pickled model
    model = load_model()

    # Start Flask app
    app.run(host="0.0.0.0", port=PORT, debug=True)
