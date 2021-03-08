from jaxKMeans import *
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

data = load_iris(as_frame=True)
jx = jaxKMeans(n_clusters=3)

df=data['data']

print(df)

#means = jx.calculateMeans(df.values)
#print(means)

pred = jx.fit_predict(df.values)
df["pred"] = pred

print(data["target"])
print(pred)

print(df["pred"].value_counts())

#km = KMeans(n_clusters = 3)
#km.fit(df)
#pre = km.predict(df)
#df["pre"] = pre
#print(df[["pred","pre"]].value_counts())

