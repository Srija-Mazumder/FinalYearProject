import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from io import StringIO
import os

# Function to find the closest gene to each cluster center
def find_closest_gene(center, data_points, gene_ids):
    distances = np.linalg.norm(data_points - center, axis=1)
    closest_index = np.argmin(distances)
    return gene_ids.iloc[closest_index], data_points[closest_index]

# Function to find the closest gene to each cluster center
def find_closest_genes_to_centers(cluster_centers, data_points, gene_names):
    closest_genes = []
    for center in cluster_centers:
        closest_gene, _ = find_closest_gene(center, data_points, gene_names)
        closest_genes.append(closest_gene)
    return closest_genes

# Function for plotting clusters
def plot_clusters(features_pca, labels, centers_pca, k):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
    plt.title(f'K-means Clustering Results (k={k})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for cluster centers
    for i, (x, y) in enumerate(centers_pca):
        plt.annotate(f'Center {i + 1}', (x, y), xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()

    # Save the plot
    file_path = os.path.join(folder_path, f'cluster_plot_k{k}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Cluster plot for k={k} saved to '{file_path}'.")

    plt.show()

# Initialize lists to store metadata and data
metadata = []
data_lines = []

# Open the file and read it line by line
with open(r'C:\Users\hp\OneDrive\Desktop\ARM SIR ML\GSE48350_series_matrix.txt', 'r') as file:
    is_table = False  # Flag to detect when the table begins

    for line in file:
        if line.startswith("!series_matrix_table_begin"):
            is_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            is_table = False
            break

        if is_table:
            data_lines.append(line)
        else:
            metadata.append(line)

print("Metadata:\n", metadata[:10])  # Printing the first few lines of metadata

# Convert data into a pandas dataframe
if data_lines:
    data = pd.read_csv(StringIO(''.join(data_lines)), delimiter='\t')
    print("Data Preview:\n", data.head())  # Check the first few rows
else:
    print("No data found after '!series_matrix_table_begin'.")
    exit()

# Extract the gene names (ID_REF column) before performing any operations
gene_names = data['ID_REF']

# Store the original data before normalization
original_data = data.copy()

# Check if data is normalized
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Get the minimum and maximum for each column
min_values = numerical_data.min()
max_values = numerical_data.max()

print("Min values:\n", min_values)
print("\nMax values:\n", max_values)

# Identify columns that are not normalized
non_normalized_cols = [col for col in numerical_data.columns if min_values[col] != 0 or max_values[col] != 1]

# Create a copy of the data to avoid modifying the original dataframe
data_normalized = data.copy()
scaler = MinMaxScaler()
data_normalized[non_normalized_cols] = scaler.fit_transform(data_normalized[non_normalized_cols])

print("Normalization completed.")

# Define the path to the folder where you want to save the normalized data
folder_path = r'C:\Users\hp\OneDrive\Desktop\ARM SIR ML'
file_path = os.path.join(folder_path, 'normalized_data.csv')
data_normalized.to_csv(file_path, index=False)
print(f"Normalized data saved to '{file_path}'.")

# Prepare data for PCA: Drop non-numeric columns (like 'ID_REF')
features = data_normalized.drop(columns=['ID_REF'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply K-Means clustering with different values of k
for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(features_scaled)
    cluster_centers = kmeans.cluster_centers_

    # Project cluster centers to PCA space for visualization
    centers_pca = pca.transform(cluster_centers)

    # Find genes closest to cluster centers
    center_genes = find_closest_genes_to_centers(cluster_centers, features_scaled, gene_names)

    # Create DataFrame with only the center genes
    result_df = pd.DataFrame({
        'ID_REF': center_genes,
        'Cluster': range(1, k+1)
    })

    # Add original expression values for center genes
    for i, gene_id in enumerate(center_genes):
        original_values = original_data.loc[original_data['ID_REF'] == gene_id].iloc[:, 1:].values[0]
        result_df.loc[i, original_data.columns[1:]] = original_values

    # Save the result to CSV
    file_path = os.path.join(folder_path, f'cluster_centers_k{k}.csv')
    result_df.to_csv(file_path, index=False)
    print(f"Cluster centers for k={k} saved to '{file_path}'.")

    # Optional: Display the results
    print(f"\nCluster centers for k={k}:")
    print(result_df.to_string(index=False))

    # Plot the clusters
    plot_clusters(features_pca, cluster_labels, centers_pca, k)

    # Print silhouette scores for each k value
    score = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette Score for k={k}: {score:.4f}")
