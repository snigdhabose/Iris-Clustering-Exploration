import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def update_centroids(X, current_clusters, n_clusters):
        centroids = np.zeros((n_clusters, 2))
        for index, cluster in enumerate(current_clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[index] = new_centroid
        return centroids
    
def calculateError(centroids,clusters,n_clusters):
    error=0
    for i in range(n_clusters):
        for j in clusters[i]:
            error+= np.sqrt(np.sum((X[j] - centroids[i]) ** 2))
    return error

def centroid_initialization(X , n_clusters):
        centroids = np.zeros((n_clusters, 2))
        for i in range(n_clusters):
            centroid_current_cluster = X[np.random.choice(range(1000))]
            centroids[i] = centroid_current_cluster
        return centroids
    
def cluster_assignment(X, centroids , n_clusters):
        current_clusters = [[] for i in range(n_clusters)]
        for index, data_point in enumerate(X):
            nearest_centroid = np.argmin(
                np.sqrt(np.sum((data_point - centroids) ** 2, axis=1)))
            current_clusters[nearest_centroid].append(index)
        return current_clusters
    

df = pd.read_csv( filepath_or_buffer='D:/SnigdhaDocs/iitm/sem2/ml/Dataset1.csv', header=None) 
df.columns=['Col1'] 
X=df.values
n_clusters =4
n_iter=1000
for i in range(5):
    centroids=centroid_initialization(X, n_clusters)
    errors=[]
    itrs=[]
    for itr in range(n_iter):
        clusters = cluster_assignment(X, centroids,n_clusters)
        previous_centroids = centroids.copy()
        centroids = update_centroids(X, clusters,n_clusters)
        diff = centroids - previous_centroids
        itrs.append(itr)
        errors.append(calculateError(centroids, clusters, n_clusters))
        if not diff.any():
            #print("K-Means Converged")
            break
    y_pred=np.zeros(1000)
    for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                y_pred[index] = cluster_id
    plt.figure(figsize=(10,8))
    plot1=plt.subplot2grid((6, 8),(i,0),rowspan=2,colspan=2)
    plt.title(str(i+1)+"th time")
    plot2=plt.subplot2grid((6, 8),(i,3),rowspan=2,colspan=3)
    plot2.set_xlabel("iteration")
    plot2.set_ylabel("error")
    plot1.scatter(X[:,0],X[:,0], c=y_pred, s=40, cmap= 'Accent')
    plot2.plot(itrs,errors)
    
    