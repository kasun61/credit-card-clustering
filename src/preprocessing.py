### Preprocessing file to clean the data set

import pandas as pd
from pathlib import Path

from .utility import set_logger, parse_config, iqr_outlier_removal, log_transform, impute_median, apply_pca

def preprocessing (data, path="data", n_components=2):
    """
    Function to preprocess data

    Args:
        data (pandas dataframe): dataframe

    Returns:
        data_pca [pandas dataframe]: dataframe after applying PCA
    
    """    
    
    # Imputing missing values with mean
    df_median_imputed = impute_median(data)

    # log transform
    df_log_transformed = log_transform(df_median_imputed, list(df_median_imputed.columns))

    ## Remove outliers for specific attributes
    unwanted_ele = ['CUST_ID']
    # list of unwanted attributes
    attribute_list = [ele for ele in list(df_log_transformed.columns) if ele not in unwanted_ele] # removing unwanted attributes
    df_without_outliers = iqr_outlier_removal(df_log_transformed,attribute_list) # remove outliers

    data_pca, pca = apply_pca(df_without_outliers, n_components)

    data_pca.to_csv(Path(path) / "data_preprocessed.csv")

    return data_pca


if __name__ == "__main__":
    
    data = pd.read_csv('data/CC_GENERAL.csv').set_index(['CUST_ID']) # set index to CUST_ID

    preprocessing(data)