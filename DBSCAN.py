import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: File path to the normalized data
file_path = r'C:\Users\hp\PycharmProjects\ARM_SIR_ML_FINAL_YEAR\normalized_data.csv'

# Step 2: Check if the file exists
print(f"Checking file at: {file_path}")

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"File found: {file_path}")

    # Step 3: Load the data
    try:
        data = pd.read_csv(file_path)
        print("Data successfully loaded!")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # Step 4: Normalize the data
    try:
        # Assuming the first column is non-numerical (e.g., ID), exclude it
        numerical_data = data.iloc[:, 1:]  # Modify this if necessary

        # Scale the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(numerical_data)
        print("Data successfully normalized!")
    except Exception as e:
        print(f"Error normalizing data: {e}")
        exit()

    # Step 5: Apply PCA
    try:
        pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
        pca_data = pca.fit_transform(normalized_data)
        print("PCA applied successfully!")
        print(f"PCA Data Shape: {pca_data.shape}")
    except Exception as e:
        print(f"Error during PCA: {e}")
        exit()

    # Step 6: Perform DBSCAN clustering
    try:
        # Debugging: Check for NaN or Inf values
        if pd.DataFrame(pca_data).isnull().values.any():
            print("PCA Data contains NaN values. Please check your input.")
            exit()

        # Adjust DBSCAN parameters as needed
        dbscan = DBSCAN(eps=0.5, min_samples=5)  # Modify eps and min_samples for your dataset
        labels = dbscan.fit_predict(pca_data)

        # Debugging: Output cluster labels
        print(f"DBSCAN Labels: {labels}")

        # Add the cluster labels to the dataset
        data['Cluster'] = labels
        print("DBSCAN clustering completed!")
    except Exception as e:
        print(f"Error during clustering: {e}")
        exit()

    # Step 7: Save clustered data
    try:
        # Ensure the output folder exists
        output_dir = r'C:\Users\hp\Desktop'  # Change path if necessary
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Specify the output file
        output_file = os.path.join(output_dir, 'clustered_data.csv')

        # Save the clustered data to a new CSV file
        data.to_csv(output_file, index=False)
        print(f"Clustering complete. Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving clustered data: {e}")
        exit()

    # Step 8: Plot DBSCAN results
    try:
        plt.figure(figsize=(10, 6))

        # Plot clusters using the PCA-reduced data
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.6)

        # Highlight noise points (label = -1)
        noise_points = (labels == -1)
        plt.scatter(pca_data[noise_points, 0], pca_data[noise_points, 1], c='red', marker='x', label='Noise', s=100)

        plt.title('DBSCAN Clustering Results with PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
