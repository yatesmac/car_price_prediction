'''data_preparation.py contain functions for data transformation, before model training.
X_train, y_train, X_test, y_test are created using the dataset function.
'''
import json

import pandas as pd
import numpy as np

from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

from flattencolumns import FlattenColumns


def is_json(data) -> bool:
    '''Check if given file is JSON formated'''
    try:
        json.loads(data)

    except ValueError as e:
        return
    
    return True


def prepare_dataset(data, target: str) -> tuple[pd.DataFrame, pd.Series]:
    '''Create X and y data given a the csv file.'''

    if is_json(data):
        df = pd.read_json(data)
    elif isinstance(data, dict):
        df = pd.DataFrame.from_dict(data, orient='records')
    else:
        df = pd.read_csv(data)

    target = target
    y = df[target]
    X = df.drop(columns=target)
    return X, y


def categorize(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    '''Defining column types.'''
    numerical = df.select_dtypes('number').columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()
    multi_label = [
        'comfort_convenience',
        'entertainment_media',
        'extras',
        'safety_security']
    categorical = list(set(categorical)-set(multi_label))
    return categorical, numerical, multi_label


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame, numerical: list[str])\
    -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Apply Standard Scaler to numeric columns in X_train and X_test dataframes.'''
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical)
        ],
        remainder='passthrough'
    )
    transformer.set_output(transform="pandas")
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test


def vectorize(X_train: pd.DataFrame, X_test: pd.DataFrame, multi_label: list[str])\
    -> tuple[NDArray[any], NDArray[any]]:
    '''Apply DictVectoizer to X_train and X_test dataframes.'''
    pipeline = Pipeline([
            ('flattern', FlattenColumns(multi_label)),
            ('vectorizer', DictVectorizer(sparse=False))  # returns dense array
            ])
    X_train =(pipeline.fit_transform(X_train))
    X_test = pipeline.transform(X_test)
    return X_train, X_test


def dataset(train_csv, test_csv, target: str, scaler: bool=False)\
    -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    '''Create X_train, y_train, X_test, y_test from train and test CSV files.'''
    X_train, y_train = prepare_dataset(train_csv, target)
    X_test, y_test = prepare_dataset(test_csv, target)

    categorical, numerical, multi_label = categorize(X_train)

    X_train = X_train[categorical + numerical + multi_label]
    X_test = X_test[categorical + numerical + multi_label]
    
    if scaler:
        X_train, X_test = scale(X_train, X_test, numerical)
    #   y_train = np.sqrt(y_train)
    #   y_test = np.sqrt(y_test)
    X_train, X_test = vectorize(X_train, X_test, multi_label)
    
    return X_train, y_train, X_test, y_test