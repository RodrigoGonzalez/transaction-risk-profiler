import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class EventAPIClient:
    """
    Realtime Events API Client.

    Attributes
    ----------
    next_sequence_number : int
        The next sequence number to fetch from the API.
    api_url : str
        The URL of the API to fetch data from.
    api_key : str
        The API key to use for authentication.
    db : Any, optional
        The database to save data to.
    interval : int
        The interval in seconds between data fetches.
    """

    def __init__(
        self,
        first_sequence_number: int = 0,
        api_url: str = "https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/",
        api_key: str = "vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC",
        db: Any | None = None,
        interval: int = 30,
    ) -> None:
        """
        Initialize the API client.

        Parameters
        ----------
        first_sequence_number : int, optional
            The first sequence number to fetch from the API.
        api_url : str, optional
            The URL of the API to fetch data from.
        api_key : str, optional
            The API key to use for authentication.
        db : Any, optional
            The database to save data to.
        interval : int, optional
            The interval in seconds between data fetches.
        """
        self.next_sequence_number = first_sequence_number
        self.api_url = api_url
        self.api_key = api_key
        self.db = db
        self.interval = interval

    def save_to_database(self, row: dict[str, Any]) -> None:
        """
        Save a data row to the database.

        Parameters
        ----------
        row : Dict[str, Any]
            The data row to save.
        """
        logger.info(f"Data received and will be saved to the database:\n{repr(row)}\n")

    def get_data(self) -> list[dict[str, Any]]:
        """
        Fetch data from the API.

        Returns
        -------
        List[Dict[str, Any]]
            The data fetched from the API.
        """
        payload = {"api_key": self.api_key, "sequence_number": self.next_sequence_number}
        response = requests.post(self.api_url, json=payload)
        data = response.json()
        self.next_sequence_number = data["_next_sequence_number"]
        return data["data"]

    def collect(self, interval: int = 30) -> None:
        """
        Check for new data from the API periodically.

        Parameters
        ----------
        interval : int, optional
            The interval in seconds between data fetches.
        """
        while True:
            logger.info("Initiating request for new data from the API...")
            data = self.get_data()
            if data:
                logger.info("Data received. Initiating saving process...")
                for row in data:
                    self.save_to_database(row)
            else:
                logger.info("No new data received from the API.")
            logger.info(f"Waiting for {interval} seconds before the next data fetch...")
            time.sleep(interval)


def main() -> None:
    """
    Collect events every 30 seconds.
    """
    client = EventAPIClient()
    client.collect()


if __name__ == "__main__":
    main()
