import pandas as pd


from utility import iqr_outlier_removal, log_transform  




def etl (df):
    
    # Imputing missing values with mean
    df.loc[(df['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].mean()
    df.loc[(df['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=df['CREDIT_LIMIT'].mean()

    # log transform
    df_log_transformed = log_transform(df, list(df.columns))

    # list of attributes to be removed
    unwanted_ele = ['CUST_ID','BALANCE_FREQUENCY','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY']
    # 
    attribute_list = [ele for ele in list(df.columns) if ele not in unwanted_ele]

    # remove outliers
    df_without_outliers = iqr_outlier_removal(df,attribute_list)

    return df_without_outliers


if __name__ == "__main__":
    df = pd.read_csv('../data/CC_GENERAL.csv')
    etl(df)