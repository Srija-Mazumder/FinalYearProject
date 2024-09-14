import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from io import StringIO
import os

# Initialize lists to store metadata and data
metadata = []
data_lines = []

# Open the file and read it line by line
with open(r'C:\Users\hp\OneDrive\Desktop\ARM SIR ML\GSE48350_series_matrix.txt', 'r') as file:
    is_table = False  # Flag to detect when the table begins

    for line in file:
        if line.startswith("!series_matrix_table_begin"):
            # Change the flag when we detect the table begins
            is_table = True
            continue
        elif line.startswith("!series_matrix_table_end"):
            # End flag for the table (if it exists), useful for extra safety
            is_table = False
            break

        if is_table:
            # Append lines to data if inside the table
            data_lines.append(line)
        else:
            # Append lines to metadata if before the table starts
            metadata.append(line)

# Print or store metadata for later use
print("Metadata:\n", metadata[:10])  # Printing the first few lines of metadata

# Now, handle the data part
# Convert data into a pandas dataframe
if data_lines:
    # Convert the list of lines into a DataFrame
    data = pd.read_csv(StringIO(''.join(data_lines)), delimiter='\t')
    print("Data Preview:\n", data.head())  # Check the first few rows
else:
    print("No data found after '!series_matrix_table_begin'.")

# Check if data is normalized (assuming `data` is your DataFrame)
# Select only numerical columns to test normalization
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Get the minimum and maximum for each column
min_values = numerical_data.min()
max_values = numerical_data.max()

# Display the min and max values for each column
print("Min values:\n", min_values)
print("\nMax values:\n", max_values)

# Identify columns that are not normalized (i.e., those with min != 0 or max != 1)
non_normalized_cols = [col for col in numerical_data.columns if min_values[col] != 0 or max_values[col] != 1]

# Create a copy of the data to avoid modifying the original dataframe
data_normalized = data.copy()

# Apply Min-Max scaling to the non-normalized columns
scaler = MinMaxScaler()
data_normalized[non_normalized_cols] = scaler.fit_transform(data_normalized[non_normalized_cols])

print("Normalization completed.")

# Define the path to the folder where you want to save the normalized data
folder_path = r'C:\Users\hp\OneDrive\Desktop\ARM SIR ML'
# Define the full path for the CSV file
file_path = os.path.join(folder_path, 'normalized_data.csv')

# Save the normalized data to the specified path
data_normalized.to_csv(file_path, index=False)
print(f"Normalized data saved to '{file_path}'.")

# Prepare data for PCA: Drop non-numeric columns
features = data_normalized.drop(columns=['ID_REF'])

# Standardize features before applying PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)  # Reduce to 2 components for 2D visualization
features_pca = pca.fit_transform(features_scaled)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
k_values = range(1, 11)  # Testing k from 1 to 10

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
cluster_results = {}

plt.figure(figsize=(16, 12))

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(features_pca)
    cluster_centers = kmeans.cluster_centers_

    # Store results
    cluster_results[k] = {
        'labels': cluster_labels,
        'centers': cluster_centers,
        'silhouette_score': silhouette_score(features_pca, cluster_labels)
    }

    # Plotting
    plt.subplot(2, 2, [2, 3, 4, 5].index(k) + 1)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o',
                          alpha=0.6)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title(f'K-Means Clustering (k={k})\nSilhouette Score: {cluster_results[k]["silhouette_score"]:.2f}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster Label')

# Display all plots
plt.tight_layout()
plt.show()

print("K-Means clustering visualizations completed.")

# Print cluster centers and silhouette scores for each k
for k in [2, 3, 4, 5]:
    print(f"\nCluster Centers for k={k}:")
    print(cluster_results[k]['centers'])
    print(f"Silhouette Score for k={k}: {cluster_results[k]['silhouette_score']:.2f}")
