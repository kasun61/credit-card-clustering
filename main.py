### main file runs the execution script for the clustering pipeline. 

from distutils.command.config import config
import pandas as pd
import click

from src.executor import executor
from src.utility import set_logger, parse_config

@click.command()
@click.argument("config_file", type=str, default="config.yml")
@click.option('--model', default='kmeans', help='Clustering Model to be used. Options: <kmeans> or <dbscan>.')

def main(model, config_file):

    ####--------------- Configuration ---------------#####
    click.echo(config_file)
    logger = set_logger("./log/main.log") # set logger

    # Load config file
    config = parse_config(config_file)

    # Load configs for data
    data_config = config["main"]["data"]  # load file
    logger.info(f"config:{config['main']}") 

    # Load data
    data = pd.read_csv(data_config).set_index(['CUST_ID']) # read data

    if model == 'kmeans':
        # Load configs for kmeans
        max_clusters = config["main"]["kmeans"]["max_clusters"]
        min_clusters = config["main"]["kmeans"]["min_clusters"]

        # Execute kmeans
        execute_clustering = executor(data,model, max_clusters = max_clusters, min_clusters = min_clusters) # execute clustering

    elif model == 'dbscan':
        # Load configs for dbscan
        eps = config["main"]["dbscan"]["eps"]

        # Execute dbscan
        execute_clustering = executor(data,model, eps = eps) # execute clustering


    return execute_clustering


if __name__ == "__main__":
    main()