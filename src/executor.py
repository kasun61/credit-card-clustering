import pandas as pd

from preprocessing import preprocessing

def executor(df):
    data_preprocessed = preprocessing(df)
    return data_preprocessed



if __name__ == "__main__":
    data = pd.read_csv('data/CC_GENERAL.csv') # read data
    executor(data)