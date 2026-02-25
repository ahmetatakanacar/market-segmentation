import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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