### main file runs the execution script for the clustering pipeline. 

import pandas as pd

from src.executor import executor

def main(df):
    execute_clustering = executor(df) # execute clustering
    return execute_clustering


if __name__ == "__main__":
    data = pd.read_csv('data/CC_GENERAL.csv') # read data
    main(data)