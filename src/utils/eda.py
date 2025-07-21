import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_univariate_distribution(df, column, bins=20, figsize=(8, 4)):
    """
    Plot the distribution of a single column (univariate analysis).
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to plot.
        bins (int): Number of bins for histograms (default 20).
        figsize (tuple): Figure size.
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column].dropna(), bins=bins, kde=True)
        plt.title(f'Distribution of {column}')
    else:
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Value Counts of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_bivariate_relationship(df, feature, target, figsize=(8, 4)):
    """
    Plot the relationship between a feature and the target variable (bivariate analysis).
    Args:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Feature column name.
        target (str): Target column name (e.g., 'class').
        figsize (tuple): Figure size.
    Returns:
        None
    """
    plt.figure(figsize=figsize)
    if pd.api.types.is_numeric_dtype(df[feature]):
        sns.boxplot(x=target, y=feature, data=df)
        plt.title(f'{feature} by {target}')
    else:
        sns.barplot(x=feature, y=target, data=df, estimator=lambda x: sum(x)/len(x))
        plt.title(f'Mean {target} by {feature}')
    plt.tight_layout()
    plt.show()


def summary_statistics(df, columns=None):
    """
    Return summary statistics for specified columns (or all columns if None).
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list or None): List of columns to summarize.
    Returns:
        pd.DataFrame: Summary statistics.
    """
    if columns is None:
        return df.describe(include='all')
    else:
        return df[columns].describe(include='all')
