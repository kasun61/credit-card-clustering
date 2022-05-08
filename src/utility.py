import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.decomposition import PCA

def parse_config(config_file):
    """
    Function to parse config file

    Args:
        config_file [str]: eg: "../config/config.yaml"

    Returns:
        config [dict]: dictionary of config file
    
    """
    with open(config_file,"rb") as f:
        config = yaml.safe_load(f)
    return config

def set_logger(log_path):
    """
    Function to set logging

    Args:
        log_path [str]: eg: "../log/train.log"

    Returns:
        logger [logger]: logger
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger

def impute_median(df):
    """
    Function to impute missing values with median

    Args:
        df (pandas dataframe): dataframe

    Returns:
        df (pandas dataframe): dataframe with missing values imputed
    
    """
    null_val = (df.isnull() # check for null values
                    .sum() # sum of null values
                    .sort_values(ascending=False) # sort values in descending order
                    .reset_index()) # reset index
    null_val.columns = ['attribute','count'] # rename columns

    # iterate through df to impute median
    for index,row in null_val.iterrows():
        if row['count'] > 0: # if null values exist
            print(f'{row["attribute"]} has {row["count"]} null values') # print attribute and count
            df.loc[(df[row['attribute']].isnull()==True),row['attribute']]=df[row['attribute']].median() # impute median
        else:
            continue
    return df

def log_transform(df, cols):
    """
    Function for Log transformation

    Args:
        df (pandas dataframe): dataframe
        cols (list): list of columns to be transformed

    Returns:
        df (pandas dataframe): dataframe with log transformed columns
    
    """
    for col in cols:
        try:
            df[col] = np.log(1 + df[col])
        except:
            print(f'{col} unsuccessful')
    
    return df


## Outlier Removal 
def iqr_outlier_removal(df,attribute_list):
    """
    Function for IQR outlier removal

    Args:
        df (pandas dataframe): dataframe
        attribute_list (list): list of columns to be transformed    

    Returns:
        df (pandas dataframe): dataframe with IQR outlier removed
    
    """
    def identify_outlier(df,column):
        """
        Function to identify outlers via IQR
        
        Args:
            df (pandas dataframe): dataframe
            column (str): column to be checked for outliers

        Returns:
            ls (list): list of indices for outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3-Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        ls = df.index[(df[column]< lower) | (df[column]> upper)] #returns list of indices for outliers
        return ls 

    def extract_outliers (df, attribute_list):
        """
        Function to extract identifed outliers
        
        Args:
            df (pandas dataframe): dataframe
            attribute_list (list): list of columns to be transformed

        Returns:
            outliers (list): list of outliers
        
        """
        outliers = [] #Initialise empty list

        for column in df[attribute_list].columns:
            outliers.extend(identify_outlier(df,column))        
        return outliers
        
    def remove_outliers(df,ls):
        """
        Function to remove outliers

        Args:
            df (pandas dataframe): dataframe
            ls (list): list of outliers

        Returns:
            outliers_removed (pandas dataframe): dataframe with outliers removed
        """
        ls = sorted(set(ls))
        df = df.drop(ls)
        return df

    outlier_index_list = extract_outliers(df,attribute_list)
    outliers_removed = remove_outliers(df,outlier_index_list)
    return outliers_removed



def apply_pca (data, n_components = 2):
    """
    Function to apply PCA to reduce dimensions
    
    Args:
        data [dataframe]: data file for user attributes
        n_components [int]: number of components in PCA, default as 2

    Returns:
        pca_df [dataframe]: data file with principal components
        pca_fit [dataframe]: fitted PCA
    """

    pca = PCA(n_components = n_components)     #PCA
    pca_fit = pca.fit_transform(data) #Reducing the dimensions of the data
    pca_df = pd.DataFrame(pca_fit,  #Dataframe for first 2 PCs
                            columns = ['PC1', 'PC2'],
                            index=data.index.copy())  # set yara_user_id as index
                            

    return pca_df, pca

def merge_cluster_labels (data, cluster_labels):
    """
    Function to merge cluster labels

    Args:
        data: dataframe with cluster labels
        cluster_labels: cluster labels

    Returns:
        data_merged: dataframe with merged cluster labels
        
    """
    # merge cluster labels
    data_merged = pd.merge(data, cluster_labels.drop(['PC1','PC2'],axis=1), left_index=True, right_index=True,how='left')

    data_merged['cluster'] = data_merged['cluster'].fillna(-1) # fill nan with 0
    
    return data_merged
