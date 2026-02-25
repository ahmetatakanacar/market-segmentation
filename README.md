# Market Segmentation
This project is an unsupervised learning study that aims to divide a customer base into subgroups (segments) based on their shared financial characteristics and spending habits.

🔗 **Dataset:** [Kaggle - Market Segmentation in Insurance](https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised/data/code)

## 🧠 Algorithms Used
* **K-Means Clustering:** Used to segment customers into the most appropriate clusters based on their financial behavior. The ideal number of clusters (K) was determined using the *Elbow Method*.
* **PCA (Principal Component Analysis):** Applied for dimensionality reduction to make high-dimensional customer data more interpretable and to visualize it on a 2D plane.

## 🛠️ Technologies Used
* **Language:** Python 3
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`KMeans`, `PCA`, `StandardScaler`)
* **Data Visualization:** Matplotlib, Seaborn

## 📈 Data Analysis & Visualizations

The main outputs and clustering graphs obtained from the project are as follows:

### 1. Balance and Purchases Distribution
![Balance and Purchases](balance_purchases.png)

### 2. Feature Correlation Matrix
![Correlation Matrix](corr.png)

### 3. Elbow Method
![Elbow Method](elbow_method.png)

### 4. Customer Segmentation with PCA
![PCA Clusters](pca1_pca2.png)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py
```

