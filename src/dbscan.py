import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from kneed import KneeLocator

class dbscan_model:
    def __init__ (self,data):
        self.data = data
        
    def dbscan_model(self, eps, min_samples):
        # X = self.data.drop(['CUST_ID'], axis=1)
        X = self.data.values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        self.labels = db.labels_
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))

        # adding cluster labels back to dataframe
        labelled_data = self.data.copy() # copy dataframe
        labelled_data['cluster'] = labels # add cluster labels back to standardised and pca df
        return labelled_data, self.labels

    def search_optimal_minpts (self, minpts):
        """
        Function to find optimal MinPts
        """       

        # Calculate average distance between each point in the data set and its nearest {MinPts} neighbours
        neigh = NearestNeighbors(n_neighbors=minpts) # using n_neighbours equivalent to minimum samples
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)

        # Sort distance values by ascending value and plot
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        i = np.arange(len(distances))

        # Find the optimal MinPts
        kn = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        kn.plot_knee();
        plt.title('K-distance Graph',fontsize=20)
        plt.xlabel('Data Points sorted by distance',fontsize=14)
        plt.ylabel('Epsilon',fontsize=14)

        print(f'Optimal MinPts: {kn.knee}')
        print(f'Optimal Epsilon: {distances[kn.knee]}')
        
        return distances[kn.knee]

    def plot_dbscan(self):
        plt.figure(figsize=(10,7))
        plt.scatter(self.data['PC1'], self.data['PC2'], c=self.labels, cmap='rainbow')
        plt.title('DBSCAN')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('data/data_preprocessed.csv').set_index('CUST_ID') # read data
    # Defining Parameters

    # define min_points
    MinPts = len(df.columns)*2 # MinPts should follow attributes*2

    # define epsilon
    eps = 0.1
    dbscan_model = dbscan_model(df)
    cluster_labels = dbscan_model.dbscan_model(eps, MinPts)
    optimal_eps = dbscan_model.search_optimal_minpts(MinPts)