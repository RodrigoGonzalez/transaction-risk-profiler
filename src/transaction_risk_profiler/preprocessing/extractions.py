import pandas as pd


def generate_dataframe_from_list(ticket_types: list[list[dict] | None]) -> pd.DataFrame:
    """
    Generate a DataFrame from a list of ticket types.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a ticket type.
    """
    # Initialize an empty list to store individual ticket dictionaries
    ticket_data = []

    # Loop through each tickets list in ticket_types
    for tickets in ticket_types:
        if not tickets:  # Check if the tickets list is not empty or None
            continue
        ticket_data.extend(iter(tickets))
    return pd.DataFrame(ticket_data)


def extract_previous_payouts(previous_payouts: list) -> float:
    if not previous_payouts:
        return 0
    amount = sum(dic["amount"] or 0 for dic in previous_payouts)
    return float(amount) / len(previous_payouts)


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "ticket_types": [
                [{"id": 1, "price": 10}, {"id": 2, "price": 20}],
                [{"id": 3, "price": 30}],
                None,
                [{"id": 4, "price": 40}],
                [],  # Empty list
            ]
        }
    )

    ticket_types = df.ticket_types.head(10).values
    result_df = generate_dataframe_from_list(ticket_types)
    print(result_df)
