'''dataset_preparation contain functions for data transformation, before model training.'''

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

from flattencolumns import FlattenColumns


def prepare_dataset(csv: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    '''Create X and y data given a the csv file.'''
    df = pd.read_csv(csv)
    target = target
    y = df[target]
    X = df.drop(columns=target)
    return X, y


def categorize(df: pd.DataFrame) -> tuple[str, str, str]:
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


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame, numerical: list)\
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


def vectorize(X_train: pd.DataFrame, X_test: pd.DataFrame, multi_label: list) -> tuple[np.ndarray, np.ndarray]:
    '''Apply DictVectoizer to X_train and X_test dataframes'''
    pipeline = Pipeline([
            ('flattern', FlattenColumns(multi_label)),
            ('vectorizer', DictVectorizer(sparse=False))  # returns dense array
            ])
    X_train =(pipeline.fit_transform(X_train))
    X_test = pipeline.transform(X_test)
    return X_train, X_test

def dataset(train_csv, test_csv, target, scaler=False):
    X_train, y_train = prepare_dataset(train_csv, target)
    X_test, y_test = prepare_dataset(test_csv, target)

    categorical, numerical, multi_label = categorize(X_train)

    X_train = X_train[categorical + numerical + multi_label]
    X_test = X_test[categorical + numerical + multi_label]
    
    if scaler:
        X_train, X_test = scale(X_train, X_test)
    X_train, X_test = vectorize(X_train, X_test, multi_label)
    
    return X_train, y_train, X_test, y_test