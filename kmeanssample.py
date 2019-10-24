from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2,1, 4,1, 0],[10, 2,10, 4,10, 0],[10, 6,10, 14,14, 8],[10, 22,13, 44,1, 33]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print(kmeans.labels_) ###each x's cluster info.
kmeans.predict([[0,0,2,3,4,5]])
print(kmeans.cluster_centers_)
