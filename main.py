import pandas as pd

from src.executor import executor

def main(df):
    data_preprocessed = preprocessing(df)
    return data_preprocessed




if __name__ == "__main__":
    data = pd.read_csv('data/CC_GENERAL.csv') # read data
    main(data)