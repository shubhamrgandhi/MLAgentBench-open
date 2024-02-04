import as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from public_times_testing_util import Api
from sklearn.model_selection import make_scorer
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.utils import check_consistent_length

# Load the data
train_patients = pd.read_csv('train_patients.csv')
train_medications = pd.read_csv('train_medications.csv')
train_clinical_data = pd.read_csv('train_clinical_data.csv')
train_clinical_extracts = pd.read_csv('train_clinical_extracts.csv')
train_demographics = pd.read_csv('train_demographics.csv')
train_tides = pd.read_csv('train_tides.csv')

# Define the metric
def smae1(y_true, y_pred):
    """SMAPE of y+1, a nonnegative float, smaller is better

    Parameters: y_true, y_pred: array-like

    Returns 100 for 100 % error.
    y_true may have missing values.
"""
    check_consistent_length(y_true, y_pred)
    y_true = np.array(y_true, copy=False).ravel()
    y_pred = np.array(y_pred, copy=False).ravel()
    y_true, y_pred = y_true[np.isfinite(y_true)], y_pred[np.isfinite(y_pred)]
    if (y_true < 0).any(): raise ValueError('y_true < 0')
    if (y_pred < 0).any(): raise ValueError('y_pred < 0')