import pandas as pd

from utility import iqr_outlier_removal, log_transform, impute_median, apply_pca

def preprocessing (df):

    data = df.set_index(['CUST_ID']) # set index to CUST_ID
    
    # Imputing missing values with mean
    df_median_imputed = impute_median(data)

    # log transform
    df_log_transformed = log_transform(df_median_imputed, list(df_median_imputed.columns))

    ## Remove outliers for specific attributes
    unwanted_ele = ['CUST_ID']
    # list of unwanted attributes
    attribute_list = [ele for ele in list(df_log_transformed.columns) if ele not in unwanted_ele] # removing unwanted attributes
    df_without_outliers = iqr_outlier_removal(df_log_transformed,attribute_list) # remove outliers

    data_pca, pca = apply_pca(df_without_outliers)

    data_pca.to_csv("data/data_preprocessed.csv")

    return data_pca


if __name__ == "__main__":
    df = pd.read_csv('data/CC_GENERAL.csv') # read data
    preprocessing(df)