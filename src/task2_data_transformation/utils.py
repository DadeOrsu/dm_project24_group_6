import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def detect_outliers_isolation_forest(df, columns, contamination=0.05):
    """
    Detects outliers in specified columns of a DataFrame using the Isolation
    Forest method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to analyze for outliers.
        contamination (float): Proportion of samples considered as outliers
        (default is 0.05).

    Returns:
        pd.DataFrame: The original DataFrame with an additional column
        'anomaly' indicating the outliers.
    """
    df_cleaned = df[columns].dropna()
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_cleaned['anomaly'] = iso_forest.fit_predict(df_cleaned[columns])
    df['anomaly'] = None
    df.loc[df_cleaned.index, 'anomaly'] = df_cleaned['anomaly']

    return df


def detect_outliers_lof(df, columns, contamination=0.05, n_neighbors=20):
    """
    Detects outliers in a DataFrame using the Local Outlier Factor (LOF)
    method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to analyze for outliers.
        contamination (float): Proportion of samples considered as outliers
        (default is 0.05).
        n_neighbors (int): Number of neighbors to use for the LOF algorithm
        (default is 20).

    Returns:
        pd.DataFrame: The original DataFrame with an additional column
        'anomaly_lof' indicating the outliers (-1 for outliers, 1 for
        normal points).
    """
    data = df[columns].dropna()

    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination)
    predictions = lof.fit_predict(data)
    df['anomaly_lof'] = None
    df.loc[data.index, 'anomaly_lof'] = predictions

    return df


def detect_outliers_zscore(df, column, threshold=3.0):
    """
    Detects outliers in a specified column of a DataFrame using Z-score method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to analyze for outliers.
        threshold (float): The Z-score threshold for determining outliers
                           (default is 3.0).

    Returns:
        pd.DataFrame: The original DataFrame with an additional column
        'anomaly_zscore' indicating the outliers.
    """
    data = df[column].dropna()
    z_scores = np.abs(stats.zscore(data))
    weights_with_index = df[column].dropna()
    outliers_indices = weights_with_index.index[z_scores > threshold]
    df['anomaly_zscore'] = None
    df.loc[outliers_indices, 'anomaly_zscore'] = True
    return df


def detect_outliers_iqr(df, column):
    """
    Detects outliers in a specified column of a DataFrame using the
    Interquartile Range (IQR) method.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to analyze for outliers.

    Returns:
        pd.DataFrame: A DataFrame containing the rows where the specified
                      column's values are considered outliers.
                      The rows returned will include all columns from the
                      original DataFrame.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
