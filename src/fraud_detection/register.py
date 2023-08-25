"""
To run:
python register.py test_filename.json

App will display the url students should send their post requests to.
"""


import logging
import os
import random
import socket
import sys
from threading import Thread

import flask_restful as restful
import pandas as pd
import pytz
import requests
import simplejson
from apscheduler.schedulers.blocking import BlockingScheduler
from flask import Flask
from flask import request
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

logging.basicConfig()

PORT = 5000
IP = socket.gethostbyname(socket.gethostname())
URLS = set()
REGISTER = "register"
SCORE = "score"
APP = None


def load_data(filename, label="acct_type"):
    data = pd.read_json(filename)
    data.drop(label, axis=1, inplace=True)
    return data


def start_server():
    print("Starting server...")
    app = Flask(__name__)
    api = restful.Api(app)
    api.add_resource(Register, "/{0}".format(REGISTER))
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(PORT, "0.0.0.0")
    thread = Thread(target=IOLoop.instance().start)
    thread.start()
    url = "http://{0}:{1}/{2}".format(IP, PORT, REGISTER)
    print("Server started. Listening for posts at: {0}".format(url))
    return app


def get_random_datapoint():
    index = random.randint(0, len(DATA))
    datapoint = DATA.loc[[index]].to_dict()
    # fix the formatting
    for key, value in datapoint.iteritems():
        datapoint[key] = value.values()[0]
    return index, datapoint


def ping():
    global APP
    if not APP:
        APP = start_server()

    index, datapoint = get_random_datapoint()
    print("Sending datapoint {0} to {1} urls".format(index, len(URLS)))

    to_remove = []

    for url in URLS:
        headers = {"Content-Type": "application/json"}
        try:
            requests.post(
                "{0}/{1}".format(url, SCORE), data=simplejson.dumps(datapoint), headers=headers
            )
            print("{0} sent data.".format(url))
        except requests.ConnectionError:
            print("{0} not listening. Removing...".format(url))
            to_remove.append(url)

    for url in to_remove:
        URLS.remove(url)


class Register(restful.Resource):
    def post(self):
        ip = request.form["ip"]
        port = request.form["port"]
        url = "http://{0}:{1}".format(ip, port)
        if url in URLS:
            print("{0} already registered".format(url))
        else:
            URLS.add(url)
            print("{0} is now registered".format(url))
        return {"Message": "Added url {0}".format(url)}

    def put(self):
        return simplejson.dumps(request.json)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python register.py test_filename.json")
        exit()
    print("It can take up to 30 seconds to start.")
    filename = sys.argv[1]

    if not os.path.isfile(filename):
        print("Invalid filename: {0}".format(filename))
        print("Goodbye.")
        exit()

    print("Starting scheduler...")
    scheduler = BlockingScheduler(timezone=pytz.utc)
    scheduler.add_job(ping, "interval", seconds=10, max_instances=100)

    print("Loading data...")
    DATA = load_data(filename)

    print("Press Ctrl+C to exit")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Goodbye!")
