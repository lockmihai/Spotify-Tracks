import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ==========================================
# CUSTOM HUNGARIAN BALANCED K-MEANS
# ==========================================

class HungarianBalancedKMeans:
    """
    Hungarian Balanced KMeans.
    Enforces equal cluster sizes by mapping the clustering problem 
    to a Bipartite Matching problem solved via the Hungarian algorithm.
    """
    def __init__(self, n_clusters=5, max_iter=20, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        K = self.n_clusters

        # Step 1: Initialize centroids using standard KMeans
        print(f"Initializing centroids using standard KMeans (K={K})...")
        kmeans = KMeans(n_clusters=K, random_state=self.random_state, n_init=5)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_.copy()

        # Calculate balanced cluster sizes
        sizes = [n_samples // K + (1 if i < n_samples % K else 0) for i in range(K)]
        
        # Target cluster mapping for the columns of the cost matrix
        cluster_mapping = []
        for i in range(K):
            cluster_mapping.extend([i] * sizes[i])
        cluster_mapping = np.array(cluster_mapping)

        labels = -np.ones(n_samples, dtype=int)

        print("Running Hungarian assignment loop...")
        for iteration in range(self.max_iter):
            t0 = time.time()
            
            # Step 2: Bipartite matching / assignment
            # Construct repeated centroids matching the sizes
            repeated_centroids = []
            for i in range(K):
                repeated_centroids.extend([centroids[i]] * sizes[i])
            repeated_centroids = np.array(repeated_centroids)

            # Compute Euclidean distance matrix (N x N)
            C = cdist(X, repeated_centroids, metric='euclidean')

            # Run Hungarian assignment algorithm
            row_ind, col_ind = linear_sum_assignment(C)

            # Map assignments back to cluster IDs
            new_labels = np.zeros(n_samples, dtype=int)
            new_labels[row_ind] = cluster_mapping[col_ind]

            # Check for convergence
            changes = np.sum(new_labels != labels)
            t_iter = time.time() - t0
            print(f"  Iteration {iteration + 1}/{self.max_iter}: {changes} label changes in {t_iter:.3f}s")
            
            if np.array_equal(new_labels, labels):
                print(f"Hungarian Balanced KMeans converged at iteration {iteration + 1}")
                break

            labels = new_labels

            # Step 3: Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(K):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = X[np.random.choice(n_samples)]
            centroids = new_centroids

        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def run_balanced_clustering():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading spotify_dataset.csv...")
    if not os.path.exists('spotify_dataset.csv'):
        print("Error: spotify_dataset.csv not found in the current directory.")
        sys.exit(1)
        
    df = pd.read_csv('spotify_dataset.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Cleaning data
    df = df.dropna(subset=['track_id', 'track_genre'])
    df = df.drop_duplicates(subset=['track_id'])
    print(f"After dropping duplicates & missing values: {df.shape}")
    
    # Stratified sampling based on track_genre (same as run_clustering.py)
    sample_size = 3000
    genres = df['track_genre'].unique()
    samples_per_genre = max(1, int(sample_size / len(genres)))
    
    sampled_indices = []
    for g, group in df.groupby('track_genre'):
        sampled_indices.extend(group.sample(min(len(group), samples_per_genre), random_state=42).index)
    df_sampled = df.loc[sampled_indices].copy()
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Stratified sample shape: {df_sampled.shape}")
    
    # Features selection
    df_sampled['explicit'] = df_sampled['explicit'].astype(float)
    features = [
        'popularity', 'duration_ms', 'danceability', 'energy', 'key', 
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'time_signature', 'explicit'
    ]
    
    # 1. Scaling comparison (Standard vs Quantile)
    X = df_sampled[features].values
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    quantile_transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    X_q = quantile_transformer.fit_transform(X)
    
    # Plot feature distributions comparison
    # Choose 6 key audio features to visualize
    visual_features = ['loudness', 'energy', 'acousticness', 'instrumentalness', 'tempo', 'speechiness']
    visual_indices = [features.index(f) for f in visual_features]
    
    fig, axes = plt.subplots(6, 2, figsize=(12, 16))
    for i, idx in enumerate(visual_indices):
        feat_name = features[idx]
        
        # Standard scaled distribution
        sns.histplot(X_std[:, idx], ax=axes[i, 0], kde=True, color='skyblue')
        axes[i, 0].set_title(f'{feat_name} (StandardScaler)')
        axes[i, 0].set_xlabel('Standardized Value')
        axes[i, 0].set_ylabel('Count')
        
        # Quantile Transformed (normal) distribution
        sns.histplot(X_q[:, idx], ax=axes[i, 1], kde=True, color='salmon')
        axes[i, 1].set_title(f'{feat_name} (QuantileTransformer - Normal)')
        axes[i, 1].set_xlabel('Quantile Normal Value')
        axes[i, 1].set_ylabel('Count')
        
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, 'feature_distributions_comparison.png')
    plt.savefig(dist_plot_path, dpi=150)
    print(f"Distribution comparison plot saved to: {dist_plot_path}")
    plt.close()
    
    # 2. RUN CLUSTERING (KMeans vs HungarianBalancedKMeans)
    n_clusters = 5
    print(f"\n--- Running Standard KMeans on Quantile-Transformed features (K={n_clusters}) ---")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_q)
    
    print(f"\n--- Running Hungarian Balanced KMeans on Quantile-Transformed features (K={n_clusters}) ---")
    hb_kmeans = HungarianBalancedKMeans(n_clusters=n_clusters, max_iter=20, random_state=42)
    hb_kmeans.fit(X_q)
    hb_labels = hb_kmeans.labels_
    
    # Calculate sizes and metrics
    km_sizes = np.bincount(kmeans_labels, minlength=n_clusters)
    hb_sizes = np.bincount(hb_labels, minlength=n_clusters)
    
    print("\nCluster Sizes comparison:")
    for i in range(n_clusters):
        print(f"  Cluster {i}: Standard KMeans = {km_sizes[i]} | Hungarian Balanced KMeans = {hb_sizes[i]}")
        
    # Validation metrics
    km_sil = silhouette_score(X_q, kmeans_labels)
    km_ch = calinski_harabasz_score(X_q, kmeans_labels)
    km_db = davies_bouldin_score(X_q, kmeans_labels)
    
    hb_sil = silhouette_score(X_q, hb_labels)
    hb_ch = calinski_harabasz_score(X_q, hb_labels)
    hb_db = davies_bouldin_score(X_q, hb_labels)
    
    print("\nClustering Validation Metrics (Computed on Quantile-Transformed Space):")
    print("Standard KMeans:")
    print(f"  Silhouette: {km_sil:.4f}")
    print(f"  Calinski-Harabasz: {km_ch:.2f}")
    print(f"  Davies-Bouldin: {km_db:.4f}")
    print("Hungarian Balanced KMeans:")
    print(f"  Silhouette: {hb_sil:.4f}")
    print(f"  Calinski-Harabasz: {hb_ch:.2f}")
    print(f"  Davies-Bouldin: {hb_db:.4f}")
    
    # 3. SAVE PCA AND COMPARE
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_q)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.get_cmap('tab10', n_clusters)
    
    # Plot Standard KMeans
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=[colors(l) for l in kmeans_labels], s=15, alpha=0.8, edgecolors='none')
    axes[0].set_title(f'Standard KMeans (K={n_clusters}, Unbalanced)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Hungarian Balanced KMeans
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=[colors(l) for l in hb_labels], s=15, alpha=0.8, edgecolors='none')
    axes[1].set_title(f'Hungarian Balanced KMeans (K={n_clusters}, Perfect Equal Sizes)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, 'balanced_vs_standard_clusters.png')
    plt.savefig(pca_plot_path, dpi=150)
    print(f"PCA comparison plot saved to: {pca_plot_path}")
    plt.close()
    
    # 4. EXPORT LABELED DATASET
    df_sampled['standard_kmeans_cluster'] = kmeans_labels
    df_sampled['hungarian_balanced_cluster'] = hb_labels
    
    csv_path = os.path.join(output_dir, 'spotify_balanced_clustered.csv')
    df_sampled.to_csv(csv_path, index=False)
    print(f"Clustered dataset exported to: {csv_path}")
    
    # 5. PRINT CLUSTER CHARACTERISTICS (PROFILES)
    print("\n--- Audio Profiles of Hungarian Balanced Clusters (Original Scale Means) ---")
    profiles = df_sampled.groupby('hungarian_balanced_cluster')[features].mean()
    print(profiles.to_string())
    
    # Return metrics for report generation
    return {
        'km_sizes': km_sizes,
        'hb_sizes': hb_sizes,
        'km_sil': km_sil,
        'km_ch': km_ch,
        'km_db': km_db,
        'hb_sil': hb_sil,
        'hb_ch': hb_ch,
        'hb_db': hb_db,
        'profiles': profiles
    }

if __name__ == '__main__':
    run_balanced_clustering()
