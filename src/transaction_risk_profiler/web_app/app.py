import logging
import socket

import pymongo
import requests
from flask import Flask
from flask import render_template
from flask import request
from pymongo import MongoClient

from transaction_risk_profiler.model import load_model

logger = logging.getLogger(__name__)

app = Flask(__name__)
PORT = 5353
REGISTER_URL = "http://10.0.1.87:5000/register"


@app.route("/score", methods=["POST"])
def score():
    data = request.json
    try:
        prediction = model.predict(data)
        data["prediction"] = prediction
        db.fraud_predictions.insert(data)

    except Exception:
        logger.info("Bad data received.")


@app.route("/dashboard")
def dashboard():
    predictions = db.fraud_predictions.find().sort("prediction", pymongo.DESCENDING)
    return render_template("dashboard.html.jinja", predictions=predictions)


# HELPERS


def register_for_ping(ip, port):
    registration_data = {"ip": ip, "port": port}
    try:
        resp = requests.post(REGISTER_URL, data=registration_data)

    except requests.ConnectionError:
        return False

    return resp.status_code == 200


if __name__ == "__main__":
    # Setup database connection
    client = MongoClient()
    db = client.fraud_prediction_service
    logger.info("Connected to Mongo database")

    # Load model
    model = load_model()
    logger.info("Loaded predictive model")

    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    logger.info("attempting to register %s:%d") % (ip_address, PORT)
    if register_for_ping(ip_address, str(PORT)):
        logger.info("Registered with pinging service")
    else:
        logger.info("Registration with pinging service failed.")

    # Start Flask app
    # 0.0.0.0 is so that the Flask app can receive requests from external
    # sources
    app.run(host="0.0.0.0", port=PORT, debug=True)
