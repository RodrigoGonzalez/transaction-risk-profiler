def extract_previous_payouts(previous_payouts: list) -> float:
    if not previous_payouts:
        return 0
    amount = sum(dic["amount"] or 0 for dic in previous_payouts)
    return float(amount) / len(previous_payouts)
