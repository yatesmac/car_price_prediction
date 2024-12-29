'''data_preparation.py - Create training, validation and testing CSV files.'''

import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from dataset_preparation import categorize


# Data root directory
root = '../../data'


def format_column_names(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_', regex=True)
    return df


def format_lists(df, multi_label):
    # Convert multi-label columns from strings to lists
    for col in multi_label:
        df[col] = df[col].str.split(',')
    return df


def format_numbers(df, numerical):
    '''Format all numerical values.'''
    for n in numerical:
        df[n] = pd.to_numeric(df[n], errors='coerce')
    return df


def format_text(df, categorical):
    '''Format all text values - removing spaces and hyphens.'''
    for c in categorical:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace('-', '_', regex=True)
    return df


def remove_column(col_names_list, df): 
    ''' Drop multiple columns based on their column names.'''
    df.drop(columns=col_names_list, axis=1, inplace=True)
    return df


def remove_duplicates(df):
    '''Removing duplicates.'''
    df = df.drop_duplicates()
    return df


def remove_missing_data(df):
    '''Remove any missing data in the df.'''
    if df.isna().sum().sum():
        df = df.dropna()
    return df


def remove_outliers(df, target):
    '''Removing Outliers using the Inter Quartile Range'''
    # Calculate the upper and lower limits
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(df[target] >= upper)[0]
    lower_array = np.where(df[target] <= lower)[0]

    # Removing the outliers
    df.drop(index=upper_array, inplace=True)
    df.drop(index=lower_array, inplace=True)
    return df


def split_save_data(df):
    '''
    Split the dataframe into 60%, 20%, 20% parts.
    Create Training, Validation and Testing datasets.
    '''
    # Split the dataframe
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    # Saving datasets
    train_dfs = [
        (df_full_train, 'df_full_train'), 
        (df_train, 'df_train'),
        (df_val, 'df_val')
    ]
    test_dfs = [
        (df_test, 'df_test')
        ]

    train_dir = f'{root}/raw/train'
    test_dir = f'{root}/raw/test'

    for df, df_name in train_dfs:
        df.to_csv(f'{train_dir}/{df_name}.csv', index=False)
        
    for df, df_name in test_dfs:
        df.to_csv(f'{test_dir}/{df_name}.csv', index=False)


def main():
    data = f'{root}external/final_scout_not_dummy.csv'
    
    df = pd.read_csv(data)
    df = remove_missing_data(df)
    df = remove_duplicates(df)
    df = format_column_names(df)
#   df = remove_outliers(df, target)

    categorical, numerical, multi_label = categorize(df)

    df = format_text(df, categorical=categorical)
    df = format_numbers(df, numerical=numerical)
    df = format_lists(df, multi_label=multi_label)

    split_save_data(df)


if __name__ == '__main__':
    main()