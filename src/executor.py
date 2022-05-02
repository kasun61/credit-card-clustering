### Executor contains the pipeline for the entire clustering flow, from preprocessing to modelling

import pandas as pd

from .preprocessing import preprocessing

def executor(df):
    data_preprocessed = preprocessing(df)
    return data_preprocessed



if __name__ == "__main__":
    executor()