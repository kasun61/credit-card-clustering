import pandas as pd
from pathlib import Path
from src.utility import iqr_outlier_removal, log_transform, impute_median, apply_pca
import pytest


def test_preprocessing (path="data", n_components=2):

    data = pd.read_csv('data/CC_GENERAL.csv').set_index(['CUST_ID']) # set index to CUST_ID
    
    ## type check
    assert isinstance(data, pd.DataFrame)

    ## shape check
    assert data.index.names == ['CUST_ID']
    assert len(data.columns) == 17

    # Imputing missing values with mean
    df_median_imputed = impute_median(data)

    ## shape check
    assert isinstance(df_median_imputed, pd.DataFrame) # type check
    assert df_median_imputed.isnull().sum().sum() == 0 # check if there are any missing values

    # log transform
    df_log_transformed = log_transform(df_median_imputed, list(df_median_imputed.columns))

    assert isinstance(df_log_transformed, pd.DataFrame) # type check
    
    ## Remove outliers for specific attributes
    unwanted_ele = ['CUST_ID']
    # list of unwanted attributes
    attribute_list = [ele for ele in list(df_log_transformed.columns) if ele not in unwanted_ele] # removing unwanted attributes

    assert 'CUST_ID' not in attribute_list # check if CUST_ID is in the list of attributes

    df_without_outliers = iqr_outlier_removal(df_log_transformed,attribute_list) # remove outliers

    ## Shape check
    assert isinstance(df_without_outliers, pd.DataFrame) # type check
    assert len(df_without_outliers) < len(df_log_transformed) # check if outliers are removed

    # apply PCA on data
    data_pca, pca = apply_pca(df_without_outliers, n_components)

    ## Shape check
    assert isinstance(data_pca, pd.DataFrame) # type check
    assert len(data_pca) == len(df_without_outliers) # check if values are transformed to principal components without change in shape
    assert ['PC1', 'PC2'] == list(data_pca.columns) # check if the columns are PCA_1 and PCA_2
    assert data_pca.index.names == ['CUST_ID'] # check if the index is CUST_ID 

    return print("Preprocessing test passed")
    