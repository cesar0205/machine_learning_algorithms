from unsupervised_learning.clustering.agglomerative_clustering import AgglomerativeClustering
import numpy as np

def main():

    X = np.array([[2,4], [0,1], [1,1], [3,2], [4,0], [2,2]])

    clusterer = AgglomerativeClustering(linkage='single')
    # start the clustering procedure
    clusterer.fit(X)
    # plot the result as a dendrogram
    clusterer.dendrogram().draw(title=clusterer.linkage.__class__.__name__)

if __name__ == "__main__":
    main();