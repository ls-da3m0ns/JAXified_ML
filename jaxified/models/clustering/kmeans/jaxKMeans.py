import jax.numpy as jnp
from jax import random

class jaxKMeans():
    def __init__(self,n_clusters = 4, maxIter=100000, n_init = 10,init_idx = None):
        print("Initializing KMeans ....")
        self.key = random.PRNGKey(0)
        self.n_clusters = n_clusters
        self.maxIter = maxIter
        self.n_init = n_init
        self.init_idx = init_idx 

    def initializeMeans(self,X):
        n = X.shape[0]
        means = X[:self.n_clusters]
        idx = random.randint(self.key, (self.n_clusters,),0,n)

        #for i in range(idx.shape[0]):
        #    means[i] = X[i]

        return means

    def euclideanDistance(self,X,means):
        temp = (X ** 2).sum(-1)[:,None] + (means ** 2).sum(-1)[None,:]
        temp = temp - 2 * X @ means.T
        return temp
    
    def classify(self,X,means):
        distances = self.euclideanDistance(X,means)
        clusters = jnp.argmin(distances,axis=1)
        return clusters
    
    def updateMeans(self,X,clusters,means):
        clusters = clusters.reshape(clusters.shape[0],1)
        n = X.shape[1]
        X = jnp.hstack((X,clusters))
        X = X[X[:,n].argsort()]

        spilited = jnp.split(X[:,:n], jnp.unique(X[:,n],
                                                 return_index=True)[1][1:])
        
        temp = [0 for j in range(len(spilited))] #jnp.zeros((len(spilited),n))
        for i in range(len(spilited)):
            temp[i] = jnp.mean(spilited[i],axis=0)
        
        temp = jnp.array(temp)
        newmean = (means + temp)/2

        return newmean

    def calculateMeans(self,X):
        means = self.initializeMeans(X)
        belongsTo = jnp.zeros((X.shape[0]))

        for iteration in range(self.maxIter):
            noChange = True
            clusters = self.classify(X,means)
            means = self.updateMeans(X,clusters,means)

            if(clusters != belongsTo).any():
                noChange = False

            belongsTo = clusters 
            if noChange:
                break

        return means

    def calculateInertia(self,X,means):
        dis = self.euclideanDistance(X,means)
        return jnp.sum(dis.reshape(-1))

    def fit(self,X):
        bestMeans = self.calculateMeans(X)
        bestInertia = self.calculateInertia(X,bestMeans)

        for i in range(self.n_init-1):
            means = self.calculateMeans(X)
            currInertia = self.calculateInertia(X,means)
            if currInertia < bestInertia:
                bestMeans = means
                bestInertia = currInertia

        self.means = bestMeans
        return self

    def predict(self,X):
        return  self.classify(X,self.means)

    def fit_predict(self,X):
        return self.fit(X).predict(X)

    def getMeans(self):
        return self.means

    
