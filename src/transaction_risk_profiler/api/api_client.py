import logging
import time

import requests

logger = logging.getLogger(__name__)


class EventAPIClient:
    """Realtime Events API Client"""

    def __init__(
        self,
        first_sequence_number=0,
        api_url="https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/",
        api_key="vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC",
        db=None,
        interval=30,
    ):
        """Initialize the API client."""
        self.next_sequence_number = first_sequence_number
        self.api_url = api_url
        self.api_key = api_key
        self.db = db
        self.interval = 30

    def save_to_database(self, row):
        """Save a data row to the database."""
        logger.info("Received data:\n" + repr(row) + "\n")  # replace this with your code

    def get_data(self):
        """Fetch data from the API."""
        payload = {"api_key": self.api_key, "sequence_number": self.next_sequence_number}
        response = requests.post(self.api_url, json=payload)
        data = response.json()
        self.next_sequence_number = data["_next_sequence_number"]
        return data["data"]

    def collect(self, interval=30):
        """Check for new data from the API periodically."""
        while True:
            logger.info("Requesting data...")
            data = self.get_data()
            if data:
                logger.info("Saving...")
                for row in data:
                    self.save_to_database(row)
            else:
                logger.info("No new data received.")
            logger.info(f"Waiting {interval} seconds...")
            time.sleep(interval)


def main():
    """Collect events every 30 seconds."""
    client = EventAPIClient()
    client.collect()


if __name__ == "__main__":
    main()
