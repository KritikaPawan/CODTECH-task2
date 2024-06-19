#TASK 2:CUSTOMER SEGMENTATION AND ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from math import pi

data = pd.read_csv('/users/kritikap/desktop/CUSTSEG/data.csv', encoding='ISO-8859-1')
print(data.head())

#Handling Missing Values
data = data.dropna(subset=['CustomerID'])

#Handling Duplicates
data = data.drop_duplicates()

#Correcting StockCode Anomalies
data = data[data['StockCode'].str.isnumeric()]

#Checking Cancelled Transactions
data = data[~data['InvoiceNo'].astype(str).str.contains('C')]

data['Description'] = data['Description'].str.strip()

#Treating Zero Unit Prices
data = data[data['UnitPrice'] > 0]

#Outlier removal
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

data = remove_outliers(data, 'Quantity')
data = remove_outliers(data, 'UnitPrice')


#Recency (R)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
latest_date = data['InvoiceDate'].max()
recency = data.groupby('CustomerID')['InvoiceDate'].apply(lambda x: (latest_date - x.max()).days)

#Frequency (F)
frequency = data.groupby('CustomerID')['InvoiceNo'].nunique()

#Monetary (M)
data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
monetary = data.groupby('CustomerID')['TotalAmount'].sum()

product_diversity = data.groupby('CustomerID')['StockCode'].nunique()

#Behavioral Features
average_basket_size = data.groupby('CustomerID')['Quantity'].mean()

#Geographic Features
country = data.groupby('CustomerID')['Country'].first()

#Cancellation Insights
cancellations = data[data['InvoiceNo'].astype(str).str.contains('C')]
cancellation_rate = cancellations.groupby('CustomerID')['InvoiceNo'].nunique() / frequency

#Seasonality & Trends
data['Month'] = data['InvoiceDate'].dt.month
monthly_sales = data.groupby('Month')['TotalAmount'].sum()

#Correlation Analysis
features = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary, 
                         'ProductDiversity': product_diversity, 'AvgBasketSize': average_basket_size, 
                         'CancellationRate': cancellation_rate})

correlation_matrix = features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix', fontsize=15)
plt.show()

#Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.fillna(0))

#Dimensionality Reduction
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_features)

#K-Means Clustering
optimal_clusters = 4  # Chosen based on previous elbow plot analysis
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(principal_components)
features['Cluster'] = clusters

#3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], 
           c=features['Cluster'], cmap='viridis', s=50, alpha=0.6)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D Visualization of Clusters', fontsize=15)
plt.show()

sns.pairplot(features, hue='Cluster', palette='viridis', diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Cluster Distribution', y=1.02, fontsize=15)
plt.show()

#Evaluation Metrics
score = silhouette_score(principal_components, clusters)
print(f'Silhouette Score: {score}')

#Radar Chart Approach
def plot_radar_chart(data, title):
    categories = list(data.index)
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, color='grey', size=8)

    values = data.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title(title, size=15, color='blue', y=1.1)
    plt.show()

for cluster in features['Cluster'].unique():
    data = features[features['Cluster'] == cluster].mean()
    plot_radar_chart(data, f'Cluster {cluster}')

#Histogram Chart Approach
features.hist(column=['Recency', 'Frequency', 'Monetary', 'ProductDiversity', 'AvgBasketSize', 'CancellationRate'], 
              by='Cluster', figsize=(15, 10), bins=20, layout=(3, 2), sharey=True, alpha=0.6, color='blue', edgecolor='black')
plt.suptitle('Distribution of Features by Cluster', fontsize=15)
plt.show()
