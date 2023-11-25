from jaxified import jaxKMeans
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
jx = jaxKMeans(n_clusters=3)

df=data['data']
pred = jx.fit_predict(df.values)
df["pred"] = pred
df["target"] = data['target']

print(f"Data Sample: {df.head()}")
print(f"Cluster Counts: {df['pred'].value_counts()}")

## print accuracy of clustering
print(
    df[["target", "pred"]].value_counts()
)



