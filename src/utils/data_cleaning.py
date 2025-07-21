import pandas as pd


def handle_missing_values(df, strategy_dict):
    """
    Handle missing values in a DataFrame according to a strategy dictionary.
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy_dict (dict): Keys are column names, values are 'mean', 'median', 'mode', or 'drop'.
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df = df.copy()
    for col, strategy in strategy_dict.items():
        if strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
        else:
            raise ValueError(f"Unknown strategy: {strategy} for column: {col}")
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates().reset_index(drop=True)


def correct_data_types(df, dtype_dict):
    """
    Convert columns in a DataFrame to specified data types.
    Args:
        df (pd.DataFrame): Input DataFrame.
        dtype_dict (dict): Keys are column names, values are target data types (e.g., 'int', 'float', 'datetime64[ns]').
    Returns:
        pd.DataFrame: DataFrame with corrected data types.
    """
    df = df.copy()
    for col, dtype in dtype_dict.items():
        if dtype.startswith('datetime'):
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = df[col].astype(dtype, errors='ignore')
    return df