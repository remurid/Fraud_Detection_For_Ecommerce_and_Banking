import pandas as pd
import ipaddress


def ip_to_int(ip_str):
    """
    Convert an IPv4 address string to an integer.
    Args:
        ip_str (str): IP address as string.
    Returns:
        int: Integer representation of the IP address.
    """
    return int(ipaddress.IPv4Address(ip_str))


def add_ip_integer_column(df, ip_column='ip_address', new_column='ip_int'):
    """
    Add a column with integer representation of IP addresses.
    Args:
        df (pd.DataFrame): Input DataFrame.
        ip_column (str): Name of the column with IP addresses.
        new_column (str): Name of the new column to add.
    Returns:
        pd.DataFrame: DataFrame with new integer IP column.
    """
    df = df.copy()
    df[new_column] = df[ip_column].apply(ip_to_int)
    return df


def merge_ip_to_country(transactions_df, ip_map_df, ip_column='ip_int', lower_col='lower_bound_ip_address', upper_col='upper_bound_ip_address', country_col='country'):
    """
    Merge transactions with IP-to-country mapping using IP integer ranges.
    Args:
        transactions_df (pd.DataFrame): Transactions DataFrame with integer IP column.
        ip_map_df (pd.DataFrame): IP-to-country DataFrame with lower/upper bounds as integers.
        ip_column (str): Name of the integer IP column in transactions_df.
        lower_col (str): Lower bound column in ip_map_df.
        upper_col (str): Upper bound column in ip_map_df.
        country_col (str): Country column in ip_map_df.
    Returns:
        pd.DataFrame: Transactions DataFrame with a new 'country' column.
    """
    def find_country(ip):
        row = ip_map_df[(ip_map_df[lower_col] <= ip) & (ip_map_df[upper_col] >= ip)]
        if not row.empty:
            return row.iloc[0][country_col]
        return 'Unknown'
    transactions_df = transactions_df.copy()
    transactions_df['country'] = transactions_df[ip_column].apply(find_country)
    return transactions_df


def add_time_features(df, purchase_time_col='purchase_time', signup_time_col='signup_time'):
    """
    Add time-based features: hour_of_day, day_of_week, time_since_signup (in hours).
    Args:
        df (pd.DataFrame): Input DataFrame with datetime columns.
        purchase_time_col (str): Name of purchase time column.
        signup_time_col (str): Name of signup time column.
    Returns:
        pd.DataFrame: DataFrame with new time-based features.
    """
    df = df.copy()
    df['hour_of_day'] = df[purchase_time_col].dt.hour
    df['day_of_week'] = df[purchase_time_col].dt.dayofweek
    df['time_since_signup'] = (df[purchase_time_col] - df[signup_time_col]).dt.total_seconds() / 3600
    return df


def add_transaction_frequency(df, group_col='user_id', freq_col='user_txn_count'):
    """
    Add a column for transaction frequency per user or device.
    Args:
        df (pd.DataFrame): Input DataFrame.
        group_col (str): Column to group by (e.g., 'user_id' or 'device_id').
        freq_col (str): Name of the new frequency column.
    Returns:
        pd.DataFrame: DataFrame with new frequency column.
    """
    df = df.copy()
    df[freq_col] = df.groupby(group_col)[group_col].transform('count')
    return df 