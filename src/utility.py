

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
        """Identify outlers via IQR"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3-Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        ls = df.index[(df[column]< lower) | (df[column]> upper)] #returns list of indices for outliers
        return ls 

    def extract_outliers (df, attribute_list):
        """Function to extract identifed outliers"""
        outliers = [] #Initialise empty list

        for column in df[attribute_list].columns:
            outliers.extend(identify_outlier(df,column))        
        return outliers
        
    def remove_outliers(df,ls):
        "Remove outliers"
        ls = sorted(set(ls))
        df = df.drop(ls)
        return df

    outlier_index_list = extract_outliers(df,attribute_list)
    outliers_removed = remove_outliers(df,outlier_index_list)
    return outliers_removed


def log_transform(df, cols):
    """
    Function for Log transformation

    Args:
        df (pandas dataframe): dataframe
        cols (list): list of columns to be transformed

    Returns:
        df (pandas dataframe): dataframe with log transformed columns
    
    """
    for col in cols[1:]: # skipping first column for index
        try:
            df[col] = np.log(1 + df[col])
        except:
            print(f'{col} unsuccessful')
    return df