"""head_json.py - extract a couple records from a huge json file.

Syntax: python head_json.py < data/transactions.json > data/transactions_subset.json
python head_json.py < data/transactions.json > data/train_subset.json
"""

import sys

start_char = "{"
stop_char = "}"
n_records = 100
level_nesting = 0

while n_records != 0:
    ch = sys.stdin.read(1)
    sys.stdout.write(ch)
    if ch == start_char:
        level_nesting += 1
    if ch == stop_char:
        level_nesting -= 1
        if level_nesting == 0:
            n_records -= 1

sys.stdout.write("]")
