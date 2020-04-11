import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from paths import *
from scipy import stats


def impute(X, target, method="linear"):
    """
    Imputes target feature (with missing values) using all other features (no missing values) using the selected method

    :param X: (n x p-1 numpy array) features to use for imputation (no missing vals)
    :param target: (n x 1 numpy array) feature to impute
    :param method: (str) One of "linear", "mean", "zero", "mode", "logistic"
    :return: (n x 1 numpy array) target array with all values imputed
    """
    nan_values = pd.isnull(target)

    if method == "zero":
        target[nan_values] = 0

    elif method == "mean":
        target[nan_values] = np.mean(target[~nan_values])

    elif method == 'mode':
        target[nan_values] = stats.mode(target[~nan_values])

    elif method == "linear":
        regression = LinearRegression()
        regression.fit(X[~nan_values], target[~nan_values])
        target[nan_values] = regression.predict(X[nan_values])

    elif method == "logistic":
        regression = LogisticRegression(multi_class='multinomial',solver='newton-cg')
        regression.fit(X[~nan_values], target[~nan_values])
        target[nan_values] = regression.predict(X[nan_values])

    else:
        raise ValueError('Please choose one of "linear", "logistic", "mean", "zero" for choice of method')

    return target

def normalize(X, method="meanvar"):
    """
    Normalizes the features of X. If "meanvar", makes mean 0 and variance 1. If "minmax" makes min 0 and max 1.

    :param X: (n x p numpy array) features to normalize (no missing vals)
    :param method: (str) One of "meanvar", "minmax"
    :return:  (n x p) numpy array with normalized features
    """

    if method == "meanvar":
        normalized_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    elif method == "minmax":
        normalized_X = (X - np.amin(X, axis=0))/np.amax(X, axis=0)

    else:
        raise ValueError('Please choose one of "meanvar", "minmax" for choice of method')

    return normalized_X

def preprocessing(X, normalization="meanvar", numerical_imputation='linear',
                      categorical_imputation='logistic'):
    """
    :param X: (n x p numpy array)
    :param normalization: (str) One of "meanvar", "minmax"
    :param numerical_imputation: (str) One of "linear", "mean", "zero"
    :param categorical_imputation: (str) One of "mode", "logistic"
    :return:
    """

    numerical_columns = ['age','hypertension','heart_disease','avg_glucose_level']

    X_to_enc = X.drop(numerical_columns + ['bmi','smoking_status'], axis=1)

    enc = OneHotEncoder(sparse=False)
    X_out = enc.fit_transform(X_to_enc)

    X_out = np.append(X_out, X[numerical_columns], axis=1)

    bmi = impute(X_out, X['bmi'], method=numerical_imputation).to_numpy().reshape(-1,1)

    smoking_raw = impute(X_out, X['smoking_status'], method=categorical_imputation).to_numpy().reshape(-1,1)

    enc_smoking = OneHotEncoder(sparse=False)
    smoking = enc_smoking.fit_transform(smoking_raw)

    X_out = np.append(X_out, bmi, axis=1)
    X_out = np.append(X_out, smoking, axis=1)

    return normalize(X_out, method=normalization)


def get_preprocessed_data(normalization="meanvar", numerical_imputation='linear',
                      categorical_imputation='logistic', write=False, datafile="stroke_all.csv"):
    """
    :param normalization: (str) One of "meanvar", "minmax"
    :param numerical_imputation: (str) One of "linear", "mean", "zero"
    :param categorical_imputation: (str) One of "mode", "logistic"
    :param write: (Bool) Whether to write to a file
    :param datafile: (str) file name to retrieve data
    :return:
    """

    data = pd.read_csv(data_dir / datafile)

    X, y = data.iloc[:, 1:-1], data.iloc[:, -1].to_numpy().reshape(-1,1)
    X = preprocessing(X, normalization=normalization, numerical_imputation=numerical_imputation,
                      categorical_imputation=categorical_imputation)

    if write:
        pd.DataFrame(np.append(X, y, axis=1)).to_csv(data_dir / datafile[:-4]+'_preprocessed.csv')

    return(X, y)

if __name__ == "__main__":
    get_preprocessed_data(write=True)
