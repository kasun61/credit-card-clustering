import pandas as pd


from utility import iqr_outlier_removal, log_transform, impute_mean, apply_pca




def preprocessing (df):
    
    # Imputing missing values with mean
    df_mean_imputed = impute_mean(df)

    # log transform
    df_log_transformed = log_transform(df_mean_imputed, list(df_mean_imputed.columns))

    ## Remove outliers for specific attributes
    unwanted_ele = ['CUST_ID','BALANCE_FREQUENCY','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY'] # list of unwanted attributes
    attribute_list = [ele for ele in list(df_log_transformed.columns) if ele not in unwanted_ele] # removing unwanted attributes
    df_without_outliers = iqr_outlier_removal(df_log_transformed,attribute_list) # remove outliers

    data_pca, pca = apply_pca(df_without_outliers)

    return data_pca


if __name__ == "__main__":
    df = pd.read_csv('../data/CC_GENERAL.csv') # read data
    preprocessing(df)