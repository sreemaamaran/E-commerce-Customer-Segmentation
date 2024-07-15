import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Load data from Excel file
data = pd.read_excel('cust_data.xlsx')

# Handle missing values by replacing with median for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Prepare data for clustering
X = data.drop(columns=['Cust_ID', 'Gender'])  # Exclude non-numeric and identifier columns

# Standardize the features by scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate silhouette scores for different values of k
silhouette_scores = []
inertia_values = []  # To store the inertia (within-cluster sum of squares) values for the elbow method
K = range(2, 11)  # Test k from 2 to 10 clusters
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"Silhouette Score: {score}")
    inertia_values.append(kmeans.inertia_)  # Sum of squared distances of samples to their closest cluster center

# Plot silhouette scores to find optimal k
plt.figure(figsize=(12, 6))

# Plot 1: Silhouette Scores
plt.subplot(1, 2, 1)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters, k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Values of k')


plt.tight_layout()
plt.savefig('cluster_evaluation.png')  # Save the plot as a PNG file

# Determine the optimal k based on the highest silhouette score
optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2

# Perform clustering with optimal k (using silhouette method) and assign clusters to data
kmeans_silhouette = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
data['Cluster_silhouette'] = kmeans_silhouette.fit_predict(X_scaled)


# Save clustered data to CSV files without index columns
data.to_csv('clustered_cust_data.csv', index=False)

# Display some statistics and results for silhouette method clustering
print(f"Number of clusters identified (Silhouette method): {optimal_k_silhouette}")
cluster_counts_silhouette = data['Cluster_silhouette'].value_counts()
print("\nCluster counts (Silhouette method):")
print(cluster_counts_silhouette)

# Visualize clusters (if the number of features allows)
if X.shape[1] <= 3:
    # Plot clusters based on silhouette method
    plt.figure(figsize=(10, 6))
    if X.shape[1] == 2:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data['Cluster_silhouette'], cmap='viridis')
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title('Cluster Visualization (Silhouette Method)')
        plt.colorbar(label='Cluster')
        plt.savefig('cluster_visualization_silhouette.png')  # Save the plot as a PNG file
    elif X.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=data['Cluster_silhouette'], cmap='viridis')
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_zlabel(X.columns[2])
        ax.set_title('Cluster Visualization (Silhouette Method)')
        plt.savefig('cluster_visualization_silhouette.png')  # Save the plot as a PNG file

print("\nClustering and analysis completed successfully.")