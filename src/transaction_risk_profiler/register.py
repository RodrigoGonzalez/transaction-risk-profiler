"""
To run:
python register.py test_filename.json

App will display the url where you can post requests to.
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

logger = logging.getLogger(__name__)

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
    logger.info("Starting server...")
    app = Flask(__name__)
    api = restful.Api(app)
    api.add_resource(Register, f"/{REGISTER}")
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(PORT, "0.0.0.0")
    thread = Thread(target=IOLoop.instance().start)
    thread.start()
    url = f"http://{IP}:{PORT}/{REGISTER}"
    logger.info(f"Server started. listening for posts at: {url}")
    return app


def get_random_datapoint():
    index = random.randint(0, len(DATA))
    datapoint = DATA.loc[[index]].to_dict()
    # fix the formatting
    for key, value in datapoint.items():
        datapoint[key] = value.values()[0]
    return index, datapoint


def ping():
    global APP
    if not APP:
        APP = start_server()

    index, datapoint = get_random_datapoint()
    logger.info(f"Sending datapoint {index} to {len(URLS)} urls")

    to_remove = []

    for url in URLS:
        headers = {"Content-Type": "application/json"}
        try:
            requests.post(f"{url}/{SCORE}", data=simplejson.dumps(datapoint), headers=headers)
            logger.info(f"{url} sent data.")
        except requests.ConnectionError:
            logger.info(f"{url} not listening. Removing...")
            to_remove.append(url)

    for url in to_remove:
        URLS.remove(url)


class Register(restful.Resource):
    def post(self):
        ip = request.form["ip"]
        port = request.form["port"]
        url = f"http://{ip}:{port}"
        if url in URLS:
            logger.info(f"{url} already registered")
        else:
            URLS.add(url)
            logger.info(f"{url} is now registered")
        return {"Message": f"Added url {url}"}

    def put(self):
        return simplejson.dumps(request.json)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.info("Usage: python register.py test_filename.json")
        exit()
    logger.info("It can take up to 30 seconds to start.")
    filename = sys.argv[1]

    if not os.path.isfile(filename):
        logger.info(f"Invalid filename: {filename}")
        logger.info("Goodbye.")
        exit()

    logger.info("Starting scheduler...")
    scheduler = BlockingScheduler(timezone=pytz.utc)
    scheduler.add_job(ping, "interval", seconds=10, max_instances=100)

    logger.info("Loading data...")
    DATA = load_data(filename)

    logger.info("Press Ctrl+C to exit")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Goodbye!")
