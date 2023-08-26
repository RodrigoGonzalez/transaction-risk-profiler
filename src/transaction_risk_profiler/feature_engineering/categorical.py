from pandas import DataFrame


def create_binary_column(df: DataFrame, new_column: str, column: str, condition, *args):
    df[new_column] = df[column].apply(condition, args=args)


def create_binary_from_value(df: DataFrame, new_column: str, column: str, value):
    df[new_column] = df[column] != value
