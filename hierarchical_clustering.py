import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Function to find the closest gene to each cluster center
def find_closest_gene(center, data_points, gene_ids):
    distances = np.linalg.norm(data_points - center, axis=1)
    closest_index = np.argmin(distances)
    return gene_ids.iloc[closest_index], data_points[closest_index]

# Initialize lists to store data
data_lines = []

# Open the file and read it line by line
with open(r'C:\Users\hp\OneDrive\Desktop\ARM SIR ML\GSE48350_series_matrix.txt', 'r') as file:
    is_table = False  # Flag to detect when the table begins
    for line in file:
        if line.startswith("!series_matrix_table_begin"):
            is_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            break
        if is_table:
            data_lines.append(line)

# Convert data into a pandas dataframe
data = pd.read_csv(StringIO(''.join(data_lines)), delimiter='\t')
print("Data Preview:\n", data.head())

# Extract the gene names (ID_REF column) before performing any operations
gene_names = data['ID_REF']

# Store the original data
original_data = data.copy()

# Prepare data for clustering: Drop non-numeric columns (like 'ID_REF')
features = data.drop(columns=['ID_REF'])

# Shuffle the data to ensure randomness and reduce data size by selecting a subset of rows
features, gene_names = shuffle(features, gene_names, random_state=42)
features = features.head(10000)  # Reduce to first 10,000 rows for memory efficiency
gene_names = gene_names.head(10000)

# Also reduce the original_data to match the selected subset of rows
original_data = original_data.head(10000)

# Scale the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5)  # Reduce to 5 dimensions to save memory
features_pca = pca.fit_transform(features_scaled)

# Perform hierarchical clustering using 'ward' linkage method
Z = linkage(features_pca, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel('Gene Index')
plt.ylabel('Euclidean Distance')
dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=gene_names.values)
plt.show()

# Define the number of clusters based on the dendrogram
max_d = 50  # Adjust this value based on where you want to cut the dendrogram
cluster_labels = fcluster(Z, max_d, criterion='distance')

# Add the cluster labels to the reduced original data
original_data['Cluster'] = cluster_labels

# Save the clustered genes to a CSV file
folder_path = r'C:\Users\hp\OneDrive\Desktop\ARM SIR ML'
output_file = os.path.join(folder_path, 'hierarchical_clustering_result.csv')
original_data.to_csv(output_file, index=False)

print(f"Hierarchical clustering result saved to '{output_file}'.")

# Optional: Visualize clusters if necessary
plt.figure(figsize=(8, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cluster_labels, cmap='viridis', s=10)
plt.title('Hierarchical Clustering (PCA Reduced Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
