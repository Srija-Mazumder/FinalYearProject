**Alzheimer's Disease Influencing Genes Identification using Machine Learning**

Project Overview
This project aims to identify influencing genes in Alzheimer's disease by leveraging unsupervised machine learning techniques. The analysis is performed on two gene expression datasets:

GSE48350
GSE11882

The goal is to identify a common set of genes across both datasets, which may have a significant role in Alzheimer's disease, by following a structured pipeline that includes data preprocessing, dimensionality reduction, and clustering.

Project Structure
This repository includes the following:
Data Preprocessing: Normalization and redundancy removal to prepare the datasets for analysis.
Dimensionality Reduction: PCA (Principal Component Analysis) to reduce the number of features.
Clustering: Various unsupervised clustering algorithms (K-Means, Hierarchical Clustering, and DBSCAN) to group genes based on their expression patterns.
Gene Network Analysis: Identification of the intersection of clusters from both datasets to find a common set of genes potentially influencing Alzheimer's disease.

Methods and Techniques
1. Normalization
Normalization ensures that the gene expression data from both datasets are scaled to a similar range, making them comparable for further analysis. Standardization (Z-score normalization) is applied to both datasets.

2. Redundancy Removal
To reduce computational complexity and remove irrelevant features:
Low-variance genes are filtered out, as they carry little information across samples.
Highly correlated genes (correlation coefficient > 0.9) are removed to avoid redundancy.

4. Principal Component Analysis (PCA)
PCA is used to reduce the dimensionality of the gene expression data. It helps in retaining the most significant features (principal components) that contribute to the variance in the data, making the clustering process more efficient.

5. Clustering
Three unsupervised clustering algorithms are applied to group genes based on their expression profiles:

K-Means Clustering
Hierarchical Clustering
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Each clustering technique is applied separately to both datasets to understand the common and unique gene groupings.

5. Gene Network and Influencing Gene Identification
After clustering, the intersection of the genes identified by each clustering algorithm from both datasets is analyzed. This helps to identify a set of genes that are common to both datasets and might be crucial in the development of Alzheimer's disease.
