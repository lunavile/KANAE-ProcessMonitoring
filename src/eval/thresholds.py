import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad
import numpy as np

def compute_kde_threshold(values, confidence_level):
    """
    Computes the KDE-based threshold for a given confidence level.
    
    Parameters:
    - values: A Pandas DataFrame column, Pandas Series, or NumPy array containing numerical data.
    - confidence_level: The confidence level (e.g., 0.99 for 99%).

    Returns:
    - threshold: The computed threshold at the given confidence level.
    - kde: The fitted KDE model.
    """

    # Ensure values is a NumPy array
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected a single column DataFrame.")
        values = values.iloc[:, 0].to_numpy()  # Convert to NumPy array

    elif isinstance(values, pd.Series):
        values = values.to_numpy()

    elif not isinstance(values, np.ndarray):
        raise TypeError("Input must be a Pandas Series, DataFrame column, or NumPy array.")

    # Fit KDE
    kde = stats.gaussian_kde(values)

    # Define CDF function
    def cdf(x):
        return quad(kde.pdf, -np.inf, x)[0]  # Integrate from -∞ to x

    # Generate value range for threshold search
    value_range = np.linspace(min(values), max(values), 1000)
    threshold = None

    for x in value_range:
        if cdf(x) >= confidence_level:
            threshold = x
            break

    if threshold is None:
        raise ValueError("Could not determine the confidence threshold.")

    return threshold, kde
