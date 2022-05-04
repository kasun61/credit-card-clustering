### main file runs the execution script for the clustering pipeline. 

import pandas as pd
import click

from src.executor import executor

@click.command()
@click.option('--model', default='kmeans', help='Clustering Model to be used. Options: <kmeans> or <dbscan>.')
# @click.argument("config_file", type=str, default="config.yml")

def main(model):
    data = pd.read_csv('data/CC_GENERAL.csv') # read data
    execute_clustering = executor(data,model) # execute clustering
    return execute_clustering


if __name__ == "__main__":
    main()