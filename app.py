from flask import Flask, request, render_template
from model import load_model
import requests
import pymongo
from pymongo import MongoClient
import json
import socket

app = Flask(__name__)
PORT = 5353
REGISTER_URL = "http://10.0.1.87:5000/register"


@app.route('/score', methods=['POST'])
def score():
    data = request.json
    try:
        prediction = model.predict(data)
        data['prediction'] = prediction
        db.fraud_predictions.insert(data)
    except:
        print "Bad data received."


@app.route('/dashboard')
def dashboard():
    predictions = db.fraud_predictions.find().sort('prediction',
                                                   pymongo.DESCENDING)
    return render_template('dashboard.html.jinja', predictions=predictions)


# HELPERS

def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    try:
        resp = requests.post(REGISTER_URL, data=registration_data)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


if __name__ == '__main__':
    # Setup database connection
    client = MongoClient()
    db = client.fraud_prediction_service
    print "Connected to Mongo database"

    # Load model
    model = load_model()
    print "Loaded predictive model"

    # Register for pinging service
    ip_address = socket.gethostbyname(socket.gethostname())
    print "attempting to register %s:%d" % (ip_address, PORT)
    if register_for_ping(ip_address, str(PORT)):
        print "Registered with pinging service"
    else:
        print "Registration with pinging service failed."

    # Start Flask app
    # 0.0.0.0 is so that the Flask app can receive requests from external
    # sources
    app.run(host='0.0.0.0', port=PORT, debug=True)
