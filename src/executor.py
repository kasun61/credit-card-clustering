### Executor contains the pipeline for the entire clustering flow, from preprocessing to modelling

import pandas as pd

from .preprocessing import preprocessing
from .dbscan import dbscan_model

def executor(df):
    data_preprocessed = preprocessing(df)

    # define min_points
    MinPts = len(data_preprocessed.columns)*2 # MinPts should follow attributes*2

    # define epsilon
    eps = 0.1
    dbscan  = dbscan_model(data_preprocessed)
    cluster_labels = dbscan.dbscan_model(eps, MinPts)
    optimal_eps = dbscan.search_optimal_minpts(MinPts)

    return optimal_eps



if __name__ == "__main__":
    executor()