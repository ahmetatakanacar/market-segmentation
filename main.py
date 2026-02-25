import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"c:\Users\ACAR\market-segmentation\data.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())