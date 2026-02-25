import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("data.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
df.drop("CUST_ID", axis=1, inplace=True)

df.isnull().sum()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df["BALANCE"], bins=30, kde=True, color="blue")
plt.title("Customer Balance Distribution")

plt.subplot(1, 2, 2)
sns.histplot(df['PURCHASES'], bins=30, kde=True, color="green")
plt.title("Customer Purchases Distribution")
plt.tight_layout()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heat Map Between Variables")
plt.show()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled.head()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker="o",color="red")
plt.title("Elbow Method")
plt.ylabel("WCSS")
plt.xticks(range(1,11))
plt.show()

kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df["Cluster"] = clusters
df.head()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(data=pca_data, columns=["PCA1", "PCA2"])
df_pca["Cluster"] = clusters

plt.figure(figsize=(10, 8))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df_pca, palette="Set2")
plt.title("Customer Segmentation")
plt.show()