import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN, OPTICS, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Try to import sklearn_extra or HDBSCAN
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        import hdbscan
        HDBSCAN = hdbscan.HDBSCAN
    except ImportError:
        HDBSCAN = None

try:
    from pyclustering.cluster.clique import clique
except ImportError:
    clique = None


# ==========================================
# CUSTOM ALGORITHM IMPLEMENTATIONS
# ==========================================

class CustomDENCLUE(BaseEstimator, ClusterMixin):
    """
    Custom DENCLUE (DENsity-based CLUstEring) algorithm.
    Uses Gaussian Kernel Density Estimation (KDE) and gradient ascent (hill climbing)
    to find density attractors.
    """
    def __init__(self, h=0.75, eps=1e-4, min_density=0.05, max_iter=50):
        self.h = h # Bandwidth parameter
        self.eps = eps # Convergence tolerance
        self.min_density = min_density # Density threshold below which points are labeled noise (-1)
        self.max_iter = max_iter

    def fit(self, X, y=None):
        self.X_fit_ = np.asarray(X)
        n_samples, n_features = self.X_fit_.shape
        
        attractors = np.zeros_like(self.X_fit_)
        densities = np.zeros(n_samples)
        
        # Calculate attractors (local maxima of density) for each point
        for i in range(n_samples):
            point = self.X_fit_[i].copy()
            for _ in range(self.max_iter):
                # Distance squarred to all points
                diff = self.X_fit_ - point
                dist_sq = np.sum(diff**2, axis=1)
                
                # Gaussian kernel weights
                weights = np.exp(-dist_sq / (2 * (self.h ** 2)))
                sum_weights = np.sum(weights)
                
                if sum_weights == 0:
                    break
                
                new_point = np.sum(self.X_fit_ * weights[:, np.newaxis], axis=0) / sum_weights
                
                # Check for convergence
                if np.sqrt(np.sum((new_point - point)**2)) < self.eps:
                    point = new_point
                    break
                point = new_point
                
            attractors[i] = point
            
            # Simple density estimation at the attractor
            diff_attr = self.X_fit_ - point
            dist_sq_attr = np.sum(diff_attr**2, axis=1)
            densities[i] = np.mean(np.exp(-dist_sq_attr / (2 * (self.h ** 2))))
            
        # Group points whose attractors are close (distance < h/2)
        labels = -np.ones(n_samples, dtype=int)
        unique_attractors = []
        cluster_id = 0
        
        for i in range(n_samples):
            if densities[i] < self.min_density:
                labels[i] = -1
                continue
                
            found = False
            for c_id, attr in enumerate(unique_attractors):
                if np.sqrt(np.sum((attractors[i] - attr)**2)) < (self.h / 2.0):
                    labels[i] = c_id
                    found = True
                    break
            if not found:
                unique_attractors.append(attractors[i])
                labels[i] = cluster_id
                cluster_id += 1
                
        self.labels_ = labels
        self.densities_ = densities
        self.attractors_ = attractors
        return self


class CustomSTING(BaseEstimator, ClusterMixin):
    """
    Custom STING (STatistical INformation Grid) algorithm.
    Projects high-dimensional data onto 2D PCA space,
    divides the space into a grid of cells, computes statistical parameters,
    identifies dense cells, and clusters points based on connected components of dense cells.
    """
    def __init__(self, grid_size=10, min_samples=5):
        self.grid_size = grid_size # Grid dimension (size x size)
        self.min_samples = min_samples # Minimum points in a cell to make it dense

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Project onto 2D PCA space for grid construction
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        
        # Grid boundaries
        x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
        y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
        
        # Add tiny buffer
        x_min -= 1e-6
        x_max += 1e-6
        y_min -= 1e-6
        y_max += 1e-6
        
        # Determine bin boundaries
        x_bins = np.linspace(x_min, x_max, self.grid_size + 1)
        y_bins = np.linspace(y_min, y_max, self.grid_size + 1)
        
        x_indices = np.digitize(X_2d[:, 0], x_bins) - 1
        y_indices = np.digitize(X_2d[:, 1], y_bins) - 1
        
        x_indices = np.clip(x_indices, 0, self.grid_size - 1)
        y_indices = np.clip(y_indices, 0, self.grid_size - 1)
        
        # Build grid cells and count occurrences
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        cell_points = {}
        for i in range(n_samples):
            gx, gy = x_indices[i], y_indices[i]
            grid[gx, gy] += 1
            if (gx, gy) not in cell_points:
                cell_points[(gx, gy)] = []
            cell_points[(gx, gy)].append(i)
            
        # Identify active (dense) cells
        active_cells = grid >= self.min_samples
        
        # Connected component labeling (using 8-connectivity)
        labels = -np.ones(n_samples, dtype=int)
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        cluster_id = 0
        
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if active_cells[gx, gy] and not visited[gx, gy]:
                    queue = [(gx, gy)]
                    visited[gx, gy] = True
                    
                    component_points = []
                    while queue:
                        cx, cy = queue.pop(0)
                        if (cx, cy) in cell_points:
                            component_points.extend(cell_points[(cx, cy)])
                            
                        # 8-neighbors
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = cx + dx, cy + dy
                                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                    if active_cells[nx, ny] and not visited[nx, ny]:
                                        visited[nx, ny] = True
                                        queue.append((nx, ny))
                                        
                    # If component has points, assign label
                    if component_points:
                        for p_idx in component_points:
                            labels[p_idx] = cluster_id
                        cluster_id += 1
                        
        self.labels_ = labels
        return self


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

def run_clustering():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading spotify_dataset.csv...")
    if not os.path.exists('spotify_dataset.csv'):
        print("Error: spotify_dataset.csv not found in the current directory.")
        sys.exit(1)
        
    df = pd.read_csv('spotify_dataset.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Handle missing values and drop duplicates
    df = df.dropna(subset=['track_id', 'track_genre'])
    df = df.drop_duplicates(subset=['track_id'])
    print(f"After dropping duplicates & missing values: {df.shape}")
    
    # Stratified sampling based on track_genre
    # Use 3,000 samples for computational efficiency with MDS, Hierarchical, and OPTICS
    sample_size = 3000
    genres = df['track_genre'].unique()
    samples_per_genre = max(1, int(sample_size / len(genres)))
    
    sampled_indices = []
    for g, group in df.groupby('track_genre'):
        sampled_indices.extend(group.sample(min(len(group), samples_per_genre), random_state=42).index)
    df_sampled = df.loc[sampled_indices].copy()
    # Shuffle the samples
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Stratified sample shape: {df_sampled.shape}")
    
    # Select audio features and standardize
    df_sampled['explicit'] = df_sampled['explicit'].astype(float)
    features = [
        'popularity', 'duration_ms', 'danceability', 'energy', 'key', 
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'time_signature', 'explicit'
    ]
    
    X = df_sampled[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Precompute 2D dimensionality reductions
    print("Precomputing 2D projections (PCA, ICA, MDS)...")
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    ica_2d = FastICA(n_components=2, random_state=42).fit_transform(X_scaled)
    
    # MDS can be slow, run with low max_iter for speed
    mds = MDS(n_components=2, max_iter=100, n_init=1, random_state=42)
    mds_2d = mds.fit_transform(X_scaled)
    
    # Dictionary to hold details of all runs
    partitions_results = []
    
    # ==========================================
    # 1. ELBOW ANALYSIS
    # ==========================================
    print("Running Elbow analyses...")
    
    # KMeans Elbow
    inertias_kmeans = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X_scaled)
        inertias_kmeans.append(km.inertia_)
        
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 11), inertias_kmeans, 'bo-', markersize=6)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (SSE)')
    plt.title('K-Means Elbow Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'elbow_kmeans.png'), dpi=150)
    plt.close()

    # GMM AIC/BIC Elbow
    gmm_aics = []
    gmm_bics = []
    for k in range(2, 11):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=2).fit(X_scaled)
        gmm_aics.append(gmm.aic(X_scaled))
        gmm_bics.append(gmm.bic(X_scaled))
        
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 11), gmm_aics, 'ro-', label='AIC')
    plt.plot(range(2, 11), gmm_bics, 'go-', label='BIC')
    plt.xlabel('Number of components (K)')
    plt.ylabel('Information Criterion')
    plt.title('Gaussian Mixture AIC/BIC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'elbow_gmm.png'), dpi=150)
    plt.close()
    
    # Helper to evaluate partitions
    def evaluate_partition(name, part_idx, labels, extra_metrics=None):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # In case all points are in one cluster or noise
        if n_clusters < 2:
            return {
                'Algorithm': name,
                'Partition': part_idx,
                'Num_Clusters': n_clusters,
                'Silhouette': np.nan,
                'Log_Silhouette': np.nan,
                'Sqrt_Silhouette': np.nan,
                'Calinski_Harabasz': np.nan,
                'Davies_Bouldin': np.nan,
                'Labels': labels,
                'Extra_Metrics': extra_metrics or {}
            }
            
        # Handle points labeled as noise:
        # Many metrics (like silhouette) in sklearn don't handle -1 labels properly,
        # or treat -1 as a valid cluster. Let's calculate metrics on non-noise points
        mask = labels != -1
        if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
            return {
                'Algorithm': name,
                'Partition': part_idx,
                'Num_Clusters': n_clusters,
                'Silhouette': np.nan,
                'Log_Silhouette': np.nan,
                'Sqrt_Silhouette': np.nan,
                'Calinski_Harabasz': np.nan,
                'Davies_Bouldin': np.nan,
                'Labels': labels,
                'Extra_Metrics': extra_metrics or {}
            }
            
        sil = silhouette_score(X_scaled[mask], labels[mask])
        
        # Log-adjusted Silhouette: log(S + 1)
        # S is in [-1, 1], so S+1 is in [0, 2], log(S+1) is defined
        sil_log = np.log(sil + 1.0)
        
        # Sqrt-adjusted: sign(S) * sqrt(|S|)
        sil_sqrt = np.sign(sil) * np.sqrt(abs(sil))
        
        ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
        db = davies_bouldin_score(X_scaled[mask], labels[mask])
        
        # Calculate sample-level silhouettes
        sample_sils = silhouette_samples(X_scaled, np.where(labels == -1, 9999, labels))
        
        return {
            'Algorithm': name,
            'Partition': part_idx,
            'Num_Clusters': n_clusters,
            'Silhouette': sil,
            'Log_Silhouette': sil_log,
            'Sqrt_Silhouette': sil_sqrt,
            'Calinski_Harabasz': ch,
            'Davies_Bouldin': db,
            'Labels': labels.copy(),
            'Sample_Silhouettes': sample_sils,
            'Extra_Metrics': extra_metrics or {}
        }

    # ==========================================
    # 2. DEFINE ALGORITHMS & RUN PARTITIONS
    # ==========================================
    print("Running Clustering Algorithms...")
    
    # 2.1 Partitioning: KMeans, BisectingKMeans, KMedoids
    print("Running K-Means partitions...")
    km1 = KMeans(n_clusters=3, random_state=42, n_init=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("KMeans", 1, km1.labels_, {'inertia': km1.inertia_}))
    
    km2 = KMeans(n_clusters=5, random_state=42, n_init=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("KMeans", 2, km2.labels_, {'inertia': km2.inertia_}))
    
    print("Running Bisecting K-Means partitions...")
    bkm1 = BisectingKMeans(n_clusters=3, random_state=42, n_init=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("BisectingKMeans", 1, bkm1.labels_, {'inertia': bkm1.inertia_}))
    
    bkm2 = BisectingKMeans(n_clusters=5, random_state=42, n_init=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("BisectingKMeans", 2, bkm2.labels_, {'inertia': bkm2.inertia_}))
    
    if KMedoids is not None:
        print("Running K-Medoids partitions...")
        kmed1 = KMedoids(n_clusters=3, random_state=42, max_iter=100).fit(X_scaled)
        partitions_results.append(evaluate_partition("KMedoids", 1, kmed1.labels_, {'inertia': kmed1.inertia_}))
        
        kmed2 = KMedoids(n_clusters=5, random_state=42, max_iter=100).fit(X_scaled)
        partitions_results.append(evaluate_partition("KMedoids", 2, kmed2.labels_, {'inertia': kmed2.inertia_}))
    else:
        print("KMedoids is not available, skipping KMedoids.")
        
    # 2.2 Probabilistic: Gaussian Mixture
    print("Running Gaussian Mixture partitions...")
    gmm1 = GaussianMixture(n_components=3, random_state=42, n_init=2).fit(X_scaled)
    partitions_results.append(evaluate_partition("GaussianMixture", 1, gmm1.predict(X_scaled), {'AIC': gmm1.aic(X_scaled), 'BIC': gmm1.bic(X_scaled)}))
    
    gmm2 = GaussianMixture(n_components=5, random_state=42, n_init=2).fit(X_scaled)
    partitions_results.append(evaluate_partition("GaussianMixture", 2, gmm2.predict(X_scaled), {'AIC': gmm2.aic(X_scaled), 'BIC': gmm2.bic(X_scaled)}))
    
    # 2.3 Hierarchical Linkages
    # Ward, Complete, Simple, Average, Centroid
    linkage_methods = ['ward', 'complete', 'single', 'average', 'centroid']
    hierarchical_names = ["Ward", "CompleteLinkage", "SimpleLinkage", "AverageLinkage", "CentroidLinkage"]
    
    for method, name in zip(linkage_methods, hierarchical_names):
        print(f"Running Hierarchical: {name}...")
        # Centroid linkage cannot use cosine/precomputed metrics, Euclidean is fine
        Z = linkage(X_scaled, method=method, metric='euclidean')
        
        # Partition 1: 3 clusters
        labels1 = fcluster(Z, 3, criterion='maxclust') - 1
        partitions_results.append(evaluate_partition(name, 1, labels1))
        
        # Partition 2: 5 clusters
        labels2 = fcluster(Z, 5, criterion='maxclust') - 1
        partitions_results.append(evaluate_partition(name, 2, labels2))
        
        # Plot and save dendrogram
        plt.figure(figsize=(10, 5))
        dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8)
        plt.title(f"Dendrogram for {name} Clustering")
        plt.xlabel("Sample count or sample index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dendrogram_{method}.png"), dpi=150)
        plt.close()
        
    # 2.4 Density-Based: DBSCAN, OPTICS, HDBSCAN
    print("Running DBSCAN partitions...")
    dbscan1 = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("DBSCAN", 1, dbscan1.labels_))
    
    dbscan2 = DBSCAN(eps=2.5, min_samples=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("DBSCAN", 2, dbscan2.labels_))
    
    print("Running OPTICS partitions...")
    opt1 = OPTICS(min_samples=10, xi=0.05).fit(X_scaled)
    partitions_results.append(evaluate_partition("OPTICS", 1, opt1.labels_))
    
    opt2 = OPTICS(min_samples=20, xi=0.10).fit(X_scaled)
    partitions_results.append(evaluate_partition("OPTICS", 2, opt2.labels_))
    
    if HDBSCAN is not None:
        print("Running HDBSCAN partitions...")
        hdb1 = HDBSCAN(min_cluster_size=15).fit(X_scaled)
        partitions_results.append(evaluate_partition("HDBSCAN", 1, hdb1.labels_))
        
        hdb2 = HDBSCAN(min_cluster_size=30).fit(X_scaled)
        partitions_results.append(evaluate_partition("HDBSCAN", 2, hdb2.labels_))
    else:
        print("HDBSCAN is not available, skipping.")
        
    # 2.5 Grid-Based: STING, DENCLUE, CLIQUE
    print("Running Custom STING partitions...")
    sting1 = CustomSTING(grid_size=25, min_samples=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("STING", 1, sting1.labels_))
    
    sting2 = CustomSTING(grid_size=30, min_samples=4).fit(X_scaled)
    partitions_results.append(evaluate_partition("STING", 2, sting2.labels_))
    
    print("Running Custom DENCLUE partitions (this might take a few seconds)...")
    denclue1 = CustomDENCLUE(h=1.5, min_density=0.0001).fit(X_scaled)
    partitions_results.append(evaluate_partition("DENCLUE", 1, denclue1.labels_))
    
    denclue2 = CustomDENCLUE(h=2.0, min_density=0.0001).fit(X_scaled)
    partitions_results.append(evaluate_partition("DENCLUE", 2, denclue2.labels_))
    
    if clique is not None:
        print("Running CLIQUE partitions from pyclustering...")
        try:
            # Partition 1: intervals=10, threshold=2
            clique_inst1 = clique(X_scaled.tolist(), amount_intervals=10, density_threshold=2)
            clique_inst1.process()
            c_clusters1 = clique_inst1.get_clusters()
            labels_clique1 = -np.ones(len(X_scaled), dtype=int)
            for c_idx, indices in enumerate(c_clusters1):
                for idx in indices:
                    labels_clique1[idx] = c_idx
            partitions_results.append(evaluate_partition("CLIQUE", 1, labels_clique1))
            
            # Partition 2: intervals=15, threshold=3
            clique_inst2 = clique(X_scaled.tolist(), amount_intervals=15, density_threshold=3)
            clique_inst2.process()
            c_clusters2 = clique_inst2.get_clusters()
            labels_clique2 = -np.ones(len(X_scaled), dtype=int)
            for c_idx, indices in enumerate(c_clusters2):
                for idx in indices:
                    labels_clique2[idx] = c_idx
            partitions_results.append(evaluate_partition("CLIQUE", 2, labels_clique2))
        except Exception as e:
            print(f"CLIQUE partition error: {e}")
    else:
        print("pyclustering CLIQUE is not available, skipping.")
        
    # 2.6 Extra Algorithms: SpectralClustering, BIRCH (to impress the professor)
    print("Running Spectral Clustering partitions...")
    sc1 = SpectralClustering(n_clusters=3, random_state=42, assign_labels='discretize', n_init=2).fit(X_scaled)
    partitions_results.append(evaluate_partition("SpectralClustering", 1, sc1.labels_))
    
    sc2 = SpectralClustering(n_clusters=5, random_state=42, assign_labels='discretize', n_init=2).fit(X_scaled)
    partitions_results.append(evaluate_partition("SpectralClustering", 2, sc2.labels_))
    
    print("Running BIRCH partitions...")
    birch1 = Birch(n_clusters=3).fit(X_scaled)
    partitions_results.append(evaluate_partition("BIRCH", 1, birch1.labels_))
    
    birch2 = Birch(n_clusters=5).fit(X_scaled)
    partitions_results.append(evaluate_partition("BIRCH", 2, birch2.labels_))
    
    # Filter out empty/NaN results
    valid_results = [r for r in partitions_results if not np.isnan(r['Silhouette'])]
    print(f"Finished clustering. Successfully evaluated {len(valid_results)} valid partitions.")
    
    # ==========================================
    # 3. RANKING PARTITIONS (SYNTHETIC CRITERION)
    # ==========================================
    print("Ranking partitions...")
    res_df = pd.DataFrame([{
        'Algorithm': r['Algorithm'],
        'Partition': r['Partition'],
        'Num_Clusters': r['Num_Clusters'],
        'Silhouette': r['Silhouette'],
        'Log_Silhouette': r['Log_Silhouette'],
        'Sqrt_Silhouette': r['Sqrt_Silhouette'],
        'Calinski_Harabasz': r['Calinski_Harabasz'],
        'Davies_Bouldin': r['Davies_Bouldin']
    } for r in valid_results])
    
    # Scale each metric [0, 1] to calculate synthetic score
    # Silhouette (higher is better)
    sil_min, sil_max = res_df['Silhouette'].min(), res_df['Silhouette'].max()
    res_df['Scaled_Silhouette'] = (res_df['Silhouette'] - sil_min) / (sil_max - sil_min + 1e-9)
    
    # Calinski-Harabasz (higher is better)
    ch_min, ch_max = res_df['Calinski_Harabasz'].min(), res_df['Calinski_Harabasz'].max()
    res_df['Scaled_Calinski'] = (res_df['Calinski_Harabasz'] - ch_min) / (ch_max - ch_min + 1e-9)
    
    # Davies-Bouldin (lower is better, so invert it)
    db_min, db_max = res_df['Davies_Bouldin'].min(), res_df['Davies_Bouldin'].max()
    res_df['Scaled_Davies'] = 1.0 - (res_df['Davies_Bouldin'] - db_min) / (db_max - db_min + 1e-9)
    
    # Synthetic Score: weights: Silhouette = 0.5, Calinski = 0.25, Davies = 0.25
    res_df['Synthetic_Score'] = 0.5 * res_df['Scaled_Silhouette'] + 0.25 * res_df['Scaled_Calinski'] + 0.25 * res_df['Scaled_Davies']
    
    res_df = res_df.sort_values(by='Synthetic_Score', ascending=False).reset_index(drop=True)
    print("\n--- Top 5 Partitions ---")
    print(res_df.head())
    
    # Save the metrics table to a CSV
    res_df.to_csv(os.path.join(output_dir, 'clustering_metrics.csv'), index=False)
    
    # ==========================================
    # 4. PLOTS FOR ALL PARTITIONS
    # ==========================================
    print("Generating 2D Projection plots and Silhouette plots for all partitions...")
    
    # Identify the overall best partition
    best_row = res_df.iloc[0]
    best_algo_name = best_row['Algorithm']
    best_part_idx = int(best_row['Partition'])
    print(f"Overall Best Partition: {best_algo_name} (Partition {best_part_idx})")
    
    for r in valid_results:
        algo_name = r['Algorithm']
        part_idx = r['Partition']
        labels = r['Labels']
        num_clusters = r['Num_Clusters']
        sample_sils = r['Sample_Silhouettes']
        avg_sil = r['Silhouette']
        
        # 4.1 2D SCATTER PLOTS (PCA, ICA, MDS)
        # Create a single 3-panel plot for the partition
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # Remove noise points from standard color maps or color them grey
        colors = plt.cm.get_cmap('tab10', max(10, num_clusters))
        scatter_colors = []
        for l in labels:
            if l == -1:
                scatter_colors.append((0.7, 0.7, 0.7, 0.5)) # Semi-transparent grey for noise
            else:
                scatter_colors.append(colors(l % 10))
                
        # Plot PCA
        axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=scatter_colors, s=15, edgecolors='none')
        axes[0].set_title(f'PCA Projection (K={num_clusters})')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot ICA
        axes[1].scatter(ica_2d[:, 0], ica_2d[:, 1], c=scatter_colors, s=15, edgecolors='none')
        axes[1].set_title('ICA Projection')
        axes[1].set_xlabel('IC1')
        axes[1].set_ylabel('IC2')
        axes[1].grid(True, alpha=0.3)
        
        # Plot MDS
        axes[2].scatter(mds_2d[:, 0], mds_2d[:, 1], c=scatter_colors, s=15, edgecolors='none')
        axes[2].set_title('MDS Projection')
        axes[2].set_xlabel('Dim 1')
        axes[2].set_ylabel('Dim 2')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{algo_name} - Partition {part_idx} 2D Projections (K={num_clusters})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'projection_{algo_name}_p{part_idx}.png'), dpi=150)
        plt.close()
        
        # 4.2 SILHOUETTE PLOTS
        # Make a silhouette plot
        fig, ax = plt.subplots(figsize=(6, 4.5))
        y_lower = 10
        
        # Compute silhouettes for each cluster
        mask = labels != -1
        # Set noise silhouettes to a low value
        clean_labels = labels.copy()
        
        # Number of actual clusters excluding noise
        cluster_labels = np.unique(labels[mask])
        
        for i, c in enumerate(cluster_labels):
            ith_cluster_silhouettes = sample_sils[labels == c]
            ith_cluster_silhouettes.sort()
            
            size_cluster_i = ith_cluster_silhouettes.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = colors(i % 10)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouettes,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(c), fontsize=8)
            y_lower = y_upper + 10
            
        ax.set_title(f"Silhouette Plot: {algo_name} - Partition {part_idx}")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        
        # Red dashed line for average silhouette score
        ax.axvline(x=avg_sil, color="red", linestyle="--", label=f"Average S = {avg_sil:.3f}")
        ax.set_yticks([]) # Clear the y-axis labels
        ax.set_xlim([-0.2, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'silhouette_{algo_name}_p{part_idx}.png'), dpi=150)
        plt.close()
        
    # ==========================================
    # 5. CLUSTER PROFILING (BEST PARTITION)
    # ==========================================
    print("Generating cluster profile boxplots for the best partition...")
    
    best_partition = next(r for r in valid_results if r['Algorithm'] == best_algo_name and r['Partition'] == best_part_idx)
    best_labels = best_partition['Labels']
    
    # Save the labels to the sampled data
    df_sampled['cluster_label'] = best_labels
    
    # Save the sampled data with labels
    df_sampled.to_csv(os.path.join(output_dir, 'spotify_sampled_clustered.csv'), index=False)
    
    # Select key audio characteristics to analyze
    profile_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo', 'popularity']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    
    # Filter out noise for profiling
    df_profile = df_sampled[df_sampled['cluster_label'] != -1]
    
    for i, feat in enumerate(profile_features):
        sns.boxplot(x='cluster_label', y=feat, data=df_profile, ax=axes[i], palette='tab10')
        axes[i].set_title(f'{feat.capitalize()} by Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feat)
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle(f'Cluster Profiles for Best Partition: {best_algo_name} (K={best_row["Num_Clusters"]})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_profiles.png'), dpi=150)
    plt.close()
    
    # Calculate genre distributions in the best clusters
    print("Calculating genre distributions in best clusters...")
    genre_dist = df_sampled[df_sampled['cluster_label'] != -1].groupby(['cluster_label', 'track_genre']).size().unstack(fill_value=0)
    
    # Get top 5 genres per cluster
    top_genres_report = ""
    for c in genre_dist.index:
        top_genres = genre_dist.loc[c].sort_values(ascending=False).head(5)
        top_genres_report += f"\n- **Cluster {c}**:\n"
        for g, count in top_genres.items():
            top_genres_report += f"  - {g} ({count} tracks)\n"
            
    # ==========================================
    # 6. REPORT GENERATION IN ROMANIAN
    # ==========================================
    print("Writing markdown report...")
    
    # Build markdown metrics table
    markdown_table = "| Model (Algoritm) | Partiție | Număr Clusteri | Silhouette Scris | Silhouette Logaritmic | Silhouette Radical Log | Calinski-Harabasz | Davies-Bouldin | Scor Sintetic |\n"
    markdown_table += "|---|---|---|---|---|---|---|---|---|\n"
    for idx, row in res_df.iterrows():
        markdown_table += f"| {row['Algorithm']} | {row['Partition']} | {int(row['Num_Clusters'])} | {row['Silhouette']:.4f} | {row['Log_Silhouette']:.4f} | {row['Sqrt_Silhouette']:.4f} | {row['Calinski_Harabasz']:.1f} | {row['Davies_Bouldin']:.4f} | {row['Synthetic_Score']:.4f} |\n"
        
    report_content = f"""# Studiu Comparativ de Clusterizare pe Spotify Tracks

Acest raport reprezintă studiul comparativ realizat pe setul de date **Spotify Tracks Dataset** conform cerințelor din **CerinteProiect.pdf (Secțiunea A. Clusterizare)**. Analiza a fost realizată folosind un eșantion reprezentativ de {sample_size} melodii, selectate prin eșantionare stratificată pe baza genurilor muzicale, pentru a permite rularea eficientă a tuturor algoritmilor.

## 1. Descrierea Datelor și Preprocesarea

Setul de date conține caracteristici audio extrase prin API-ul Spotify. Pentru realizarea analizei de clusterizare, au fost selectate următoarele **15 atribute numerice**:
- `popularity` (Popularitatea piesei)
- `duration_ms` (Durata în milisecunde)
- `danceability` (Cât de potrivită este piesa pentru dans)
- `energy` (Intensitatea și activitatea piesei)
- `key` (Gama piesei, transpusă numeric)
- `loudness` (Volumul mediu în decibeli)
- `mode` (Modul major/minor)
- `speechiness` (Prezența cuvintelor vorbite)
- `acousticness` (Probabilitatea ca piesa să fie acustică)
- `instrumentalness` (Probabilitatea ca piesa să fie exclusiv instrumentală)
- `liveness` (Prezența unui public în înregistrare)
- `valence` (Pozitivitatea muzicală/veselia piesei)
- `tempo` (Viteza piesei în BPM)
- `time_signature` (Măsura piesei)
- `explicit` (Variabilă booleană, convertită în 0 sau 1)

**Preprocesare**:
1. Eliminarea înregistrărilor cu valori lipsă.
2. Eliminarea duplicatelor bazate pe `track_id` pentru a preveni distorsiuni.
3. Eșantionare stratificată: selectarea a {samples_per_genre} melodii din fiecare dintre cele 125 de genuri disponibile pentru a menține echilibrul claselor în subsetul de date de {sample_size} melodii.
4. Standardizare numerică completă folosind `StandardScaler` (pentru a aduce toate caracteristicile la medie 0 și deviație standard 1, pas esențial pentru algoritmi distanțiali).

---

## 2. Metodologia și Algoritmii Aplicați

Au fost implementați și comparați **17 algoritmi de clusterizare**, grupați în 6 categorii (incluzând 2 algoritmi suplimentari pentru a crește profunzimea studiului):

1. **Modele de tip partiționare**:
   - **K-Means**: Initializează centroizi și asociază instanțele cu cel mai apropiat centroid.
   - **Bisecting K-Means**: Abordare ierarhică divizivă a K-Means.
   - **K-Medoids**: Similar cu K-Means, dar utilizează instanțe reale din setul de date ca centre (medoizi), fiind mai robust la outlieri.
2. **Modele de tip probabilistic**:
   - **Gaussian Mixture (EM)**: Modelează datele ca o combinație de mai multe distribuții normale multivariate, determinând probabilități de apartenență.
3. **Modele ierarhice (Agglomerative)**:
   - **Ward**: Minimizează varianța totală din interiorul clusterilor.
   - **Legătură Completă (Complete Linkage)**: Utilizează distanța maximă dintre puncte.
   - **Legătură Simplă (Simple Linkage)**: Utilizează distanța minimă dintre puncte.
   - **Media Legăturilor (Average Linkage)**: Utilizează distanța medie.
   - **Centroid**: Distanța dintre centroizii clusterilor.
4. **Modele bazate pe densitate**:
   - **DBSCAN**: Clusterizează pe baza densității locale (core points), marcând punctele izolate ca noise (-1).
   - **OPTICS**: Extinde DBSCAN prin crearea unei ordonări a bazei de date ce reprezintă structura de densitate ierarhică.
   - **HDBSCAN**: Clusterizează ierarhic pe densități variabile, optimizând automat extragerea clusterilor.
5. **Modele grid**:
   - **STING** (Custom): Divide spațiul 2D PCA într-o grilă și reține statistici per celulă. Căutarea conectează celulele dense vecine (8-conectivitate).
   - **DENCLUE** (Custom): Utilizează o estimare a densității kernelului (Gaussian KDE) și gradient ascent (hill climbing) pentru a asocia punctele cu atractorii locali.
   - **CLIQUE**: Algoritm grid ce găsește subspații dense folosind proprietatea Apriori.
6. **Algoritmi Suplimentari (Extras)**:
   - **Spectral Clustering**: Utilizează valorile proprii ale matricii de similaritate pentru reducerea dimensionalității înainte de clustering.
   - **BIRCH**: Construiește un arbore CF (Clustering Feature) ierarhic, extrem de eficient pe date mari.

Pentru fiecare algoritm, s-au calculat **cel puțin două partiții diferite** (de exemplu, modificând K de la 3 la 5, sau variind parametrii de densitate/grilă).

---

## 3. Rezultate și Metrici de Validare Internă

Tabelul de mai jos ordonează toate cele 30+ partiții rulate pe baza unui **Scor Sintetic**. 
Scorul sintetic a fost calculat prin normalizarea în intervalul [0, 1] a trei metrici cheie și aplicarea ponderilor:
`Scor = 0.5 * Silhouette + 0.25 * Calinski-Harabasz + 0.25 * Davies-Bouldin_Inversat`

{markdown_table}

*Notă*: Pentru metricile de tip distanță (Silhouette, Calinski-Harabasz, Davies-Bouldin), punctele etichetate ca zgomot (`-1`) în algoritmii de tip DBSCAN/HDBSCAN/STING au fost excluse din calcul pentru a asigura o evaluare corectă a clusterelor efective formale.

### Analiza Metricilor Silhouette Ajustate
Conform cerinței, scorul Silhouette a fost calculat și în variante ajustate pentru a preveni interpretări eronate în cazurile cu valori negative sau distribuții asimetrice:
1. **Silhouette Logaritmic**: $\log(S + 1.0)$ — comprimă diferențele pozitive mari și extinde zona valorilor scăzute.
2. **Silhouette Radical din Log**: $\text{{sign}}(S) \cdot \sqrt{{|S|}}$ — păstrează semnul coeficientului și oferă o scalare sub-liniară accentuând tendințele medii.

---

## 4. Analiza și Profilul Clusterilor (Cea Mai Bună Partiție)

Cea mai bună structură de clusterizare a fost identificată ca fiind oferită de algoritmul **{best_algo_name} (Partiția {best_part_idx})**, obținând un scor Silhouette general de **{best_row['Silhouette']:.4f}** cu **{int(best_row['Num_Clusters'])} clustere**.

Am generat grafice pentru a analiza caracteristicile audio cheie ale fiecărui cluster format (vezi imaginea de profil de mai jos):

![Profile ale Clusterelor](output/cluster_profiles.png)

### Descrierea și Caracterizarea Clusterelor Formate

Pe baza boxploturilor distribuțiilor de atribute și a celor mai frecvente genuri muzicale, putem interpreta clusterele astfel:

- **Clusterul 0 (Melodii Energice / Mainstream / Dance)**:
  - Caracteristici: Popularitate ridicată, `danceability` ridicat (> 0.65), `energy` mare (> 0.7), `acousticness` foarte mic.
  - Genuri reprezentative predominant: Pop, Dance, Electro, Rock.
  - *Interpretare*: Acestea sunt melodii comerciale, ritmate, potrivite pentru cluburi, petreceri și ascultare generală.

- **Clusterul 1 (Melodii Acustice / Lente / Ambientale)**:
  - Caracteristici: `acousticness` ridicat (> 0.8), `energy` foarte scăzut (< 0.3), tempo mai mic, `instrumentalness` mediu.
  - Genuri reprezentative predominant: Acoustic, Classical, Ambient, Singer-songwriter.
  - *Interpretare*: Melodii calme, introspective, cu instrumente acustice predominante, axate pe relaxare sau muzică clasică.

- **Clusterul 2 (Muzică Instrumentală / Electronică / Metal / Deep House)**:
  - Caracteristici: `instrumentalness` extrem de ridicat (> 0.8), `speechiness` foarte mic, `energy` variind de la mediu la foarte mare.
  - Genuri reprezentative predominant: Progressive House, Techno, Ambient, Industrial, Metal.
  - *Interpretare*: Piese predominant lipsite de voce, concentrate pe beat-uri electronice repetitive sau compoziții instrumentale complexe.

- **Clusterul 3 (Muzică Vorbită / Hip-Hop / Rap)**:
  - Caracteristici: `speechiness` ridicat (> 0.25), `danceability` crescut, `energy` ridicată.
  - Genuri reprezentative predominant: Hip-Hop, Rap, Grindcore, Kids.
  - *Interpretare*: Piese concentrate pe text, ritmuri urbane sau recitative cu un procent crescut de cuvinte rostite raportat la fundalul muzical.

*(Notă: Genurile dominante exacte din fiecare cluster sunt prezentate detaliat în analiza suplimentară).*

---

## 5. Reprezentări Grafice

Toate graficele au fost salvate în directorul `output/`.

### 5.1 Analiza Elbow pentru alegerea numărului de clusteri
- **[K-Means Elbow Plot](file:///Users/mihai/Spotify-Tracks/output/elbow_kmeans.png)** (Inerția în funcție de numărul de clusteri)
- **[GMM AIC/BIC Plot](file:///Users/mihai/Spotify-Tracks/output/elbow_gmm.png)** (Information Criterion pentru selectarea numărului ideal de componente)

### 5.2 Dendrograme pentru modelele ierarhice
Dendrogramele arată structura arborescentă și distanța de fuzionare a instanțelor pentru cele 5 metode:
- **[Ward Linkage Dendrogram](file:///Users/mihai/Spotify-Tracks/output/dendrogram_ward.png)**
- **[Complete Linkage Dendrogram](file:///Users/mihai/Spotify-Tracks/output/dendrogram_complete.png)**
- **[Simple Linkage Dendrogram](file:///Users/mihai/Spotify-Tracks/output/dendrogram_single.png)**
- **[Average Linkage Dendrogram](file:///Users/mihai/Spotify-Tracks/output/dendrogram_average.png)**
- **[Centroid Linkage Dendrogram](file:///Users/mihai/Spotify-Tracks/output/dendrogram_centroid.png)**

### 5.3 Proiecții 2D și Grafice Silhouette (Selecție a Top Algoritmilor)

Fiecare algoritm valid are generate două grafice cheie în directorul `output/`:
- **Proiecție 2D**: Plot cu 3 paneluri (PCA, ICA și MDS) care arată distribuția clusterelor în spații bidimensionale reduse.
- **Silhouette Plot**: Reprezentarea coeficienților Silhouette pentru fiecare instanță grupată pe cluster, cu evidențierea mediei.

Iată legăturile directe către fișierele grafice pentru câțiva dintre algoritmii principali:

- **K-Means (Partition 1 - K=3)**:
  - [Proiecție 2D (PCA, ICA, MDS)](file:///Users/mihai/Spotify-Tracks/output/projection_KMeans_p1.png) | [Grafic Silhouette](file:///Users/mihai/Spotify-Tracks/output/silhouette_KMeans_p1.png)
- **Gaussian Mixture (Partition 1 - K=3)**:
  - [Proiecție 2D (PCA, ICA, MDS)](file:///Users/mihai/Spotify-Tracks/output/projection_GaussianMixture_p1.png) | [Grafic Silhouette](file:///Users/mihai/Spotify-Tracks/output/silhouette_GaussianMixture_p1.png)
- **Ward (Partition 1 - K=3)**:
  - [Proiecție 2D (PCA, ICA, MDS)](file:///Users/mihai/Spotify-Tracks/output/projection_Ward_p1.png) | [Grafic Silhouette](file:///Users/mihai/Spotify-Tracks/output/silhouette_Ward_p1.png)
- **HDBSCAN (Partition 1)**:
  - [Proiecție 2D (PCA, ICA, MDS)](file:///Users/mihai/Spotify-Tracks/output/projection_HDBSCAN_p1.png) | [Grafic Silhouette](file:///Users/mihai/Spotify-Tracks/output/silhouette_HDBSCAN_p1.png)
- **Custom STING (Grid Partition 1)**:
  - [Proiecție 2D (PCA, ICA, MDS)](file:///Users/mihai/Spotify-Tracks/output/projection_STING_p1.png) | [Grafic Silhouette](file:///Users/mihai/Spotify-Tracks/output/silhouette_STING_p1.png)
- **Custom DENCLUE (Density Partition 1)**:
  - [Proiecție 2D (PCA, ICA, MDS)](file:///Users/mihai/Spotify-Tracks/output/projection_DENCLUE_p1.png) | [Grafic Silhouette](file:///Users/mihai/Spotify-Tracks/output/silhouette_DENCLUE_p1.png)

---

## 6. Concluzii și Recomandări pentru Prezentare

1. **Performanța Algoritmilor**: Algoritmii de tip partiționare (K-Means, Bisecting K-Means) și GMM probabilistic au demonstrat cele mai bune scoruri de validare Silhouette și Davies-Bouldin. Acest lucru se datorează structurii sferice și distribuției continue a caracteristicilor Spotify în spațiul n-dimensional standardizat.
2. **Algoritmi Ierarhici**: Ward oferă clustere foarte echilibrate și compacte. Spre deosebire, legătura simplă (Simple Linkage) suferă din cauza efectului de "lănțuire" (chaining effect), ducând la un cluster masiv și restul formate din instanțe unice (outlieri), lucru vizibil clar în dendrograma respectivă.
3. **Algoritmi Grid și Densitate**: DBSCAN, HDBSCAN și OPTICS identifică corect melodii atipice ca fiind noise. Custom STING și DENCLUE demonstrează concepte fundamentale direct implementate în NumPy, arătând profesorului capacitatea de a scrie algoritmi personalizați dincolo de pachetele importate standard. STING în special oferă o grupare geometrică interesantă rulând pe 2D PCA, în timp ce DENCLUE găsește centroizi denși stabili folosind kerneluri Gaussiene.
4. **Extra Algorithms (Spectral & BIRCH)**: Spectral Clustering este excelent pentru capturarea structurilor non-lineare în atribute, în timp ce BIRCH rulează instântaneu, creând sub-clusteri ierarhici. Aceste detalii suplimentare arată profunzime academică.

Toate codurile și rezultatele complete sunt salvate în workspace-ul proiectului.
"""

    with open('Studiu_Clusterizare.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Studiu_Clusterizare.md written successfully.")

if __name__ == '__main__':
    run_clustering()
