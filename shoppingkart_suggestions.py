from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

#TF-IDF == Term Frequency * Inverse Document Frequency
documents = ["Milk Egg Oranges IceCream CornFlakes Chocolate LUX colgate brush","Milk Fan thread rope colgate brush","IceCream Corn Flakes Chocolate bresh brush","Rubber Oranges Corn Flakes notebook","Laptop Mobile Iron Box Corn Flakes Chocolate LUX brush","mobilecase pen pencil notebook CornFlakes Chocolate bresh colgate","pen notebook"]

#final output=KMeans.cluster_centers_
vectorizer = TfidfVectorizer()  #convert data to matrix form, #vectorizer = CountVectorizer() #this will get us counts of all unique tokens
'''v1 = CountVectorizer()
y1=v1.fit_transform(documents)
print(v1.get_feature_names())
print(y1.toarray())
'''
X = vectorizer.fit_transform(documents) #Learn the vocabulary dictionary and return term-document matrix.
#print(vectorizer.get_feature_names())
#print(X.shape)
#print(X.toarray())

no_of_clusters=5
#Method for initialization, defaults to ‘k-means++’:#‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
#n_init : int, default: 10 #Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
#max_iter : int, default: 300, #Maximum number of iterations of the k-means algorithm for a single run.

model = KMeans(n_clusters=no_of_clusters, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
#Returns the indices after the sort operation.##numpy.argsort(a, axis=-1, kind=None, order=None)##kind : {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional, Default==Quicksort
terms = vectorizer.get_feature_names()  #Array mapping from feature integer indices to feature name
'''
for a in range(5):
 print("Cluster %d:" % a),
 for ind in order_centroids[a, :10]:
  print(' %s' % terms[ind]),
 '''
Y = vectorizer.transform(["Milk Egg"])
prediction = model.predict(Y)
print("people who buy--Milk Egg --Likely to buy--")
for ind in order_centroids[prediction[0], :10]:
  print(' %s' % terms[ind])
Y = vectorizer.transform(["Mobile"])
prediction = model.predict(Y)
print("people who buy--Mobile--Likely to buy--")

for ind in order_centroids[prediction[0], :10]:
  print(' %s' % terms[ind])
