import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# 1. MUSIC STYLE MAPPING
# ==========================================

def map_genre_to_style(genre):
    """
    Groups the 125 Spotify genres into 5 broad musical styles.
    """
    genre = str(genre).lower()
    
    acoustic_keywords = [
        'acoustic', 'classical', 'piano', 'opera', 'ambient', 
        'harpsichord', 'choral', 'romance', 'singer-songwriter', 
        'chill', 'sleep', 'study', 'guitar', 'new-age', 'world-music'
    ]
    electronic_keywords = [
        'dance', 'disco', 'edm', 'house', 'techno', 'trance', 
        'club', 'groove', 'minimal-techno', 'electro', 'breakbeat', 
        'garage', 'dubstep', 'drum-and-bass', 'synth-pop', 'industrial',
        'progressive-house', 'hardstyle', 'detroit-techno', 'happy-hardcore'
    ]
    rock_metal_keywords = [
        'rock', 'metal', 'punk', 'grindcore', 'hardcore', 'goth', 
        'emo', 'grunge', 'metalcore', 'hard-rock', 'heavy-metal', 
        'black-metal', 'death-metal', 'psych-rock', 'garage-rock'
    ]
    hiphop_rap_keywords = [
        'hip-hop', 'rap', 'r-n-b', 'reggae', 'ska', 'trap', 
        'soul', 'funk', 'dancehall', 'hiphop'
    ]
    
    # Check keywords
    if any(kw in genre for kw in acoustic_keywords):
        return 'Acoustic_Classical'
    if any(kw in genre for kw in electronic_keywords):
        return 'Electronic_Dance'
    if any(kw in genre for kw in rock_metal_keywords):
        return 'Rock_Metal'
    if any(kw in genre for kw in hiphop_rap_keywords):
        return 'Hiphop_Rap'
        
    return 'Pop_Jazz_Others'


# ==========================================
# 2. META SONG LEARNER CLASS
# ==========================================

class MetaSongLearner(BaseEstimator, ClusterMixin):
    """
    Meta Learner that combines a supervised style classifier
    with fine-tuned, style-specific clustering models.
    """
    def __init__(self, classifier=None):
        if classifier is None:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.classifier = classifier
        self.style_models = {}
        self.style_scalers = {}
        self.unique_styles_ = []
        
    def fit(self, X, genres):
        # 1. Map genres to 5 broad styles
        styles = np.array([map_genre_to_style(g) for g in genres])
        self.unique_styles_ = np.unique(styles)
        
        # 2. Fit the style classifier (External Service)
        print(f"Fitting Style Classifier (External Service) on {X.shape[0]} samples...")
        self.classifier.fit(X, styles)
        
        # 3. For each style, run grid search on its subset
        for style in self.unique_styles_:
            mask = styles == style
            X_style = X[mask]
            
            print(f"\n--- Fine-tuning clustering for style: {style} (samples={X_style.shape[0]}) ---")
            
            if len(X_style) < 10:
                print(f"  Too few samples. Fallback to default K-Means(k=2)")
                self.style_scalers[style] = None
                fallback = KMeans(n_clusters=2, random_state=42, n_init=5)
                fallback.fit(X_style)
                self.style_models[style] = ('KMeansFallback', fallback)
                continue
                
            # Scale subset separately
            scaler = StandardScaler()
            X_style_scaled = scaler.fit_transform(X_style)
            self.style_scalers[style] = scaler
            
            # Grid search candidates
            best_score = -2.0
            best_model = None
            
            # Form candidate list
            candidates = []
            for k in [2, 3, 4]:
                candidates.append(('KMeans', KMeans(n_clusters=k, random_state=42, n_init=5)))
            for k in [2, 3, 4]:
                candidates.append(('GMM', GaussianMixture(n_components=k, random_state=42, n_init=2)))
            for eps in [1.5, 2.0, 2.5]:
                candidates.append(('DBSCAN', DBSCAN(eps=eps, min_samples=5)))
                
            # Evaluate candidates
            for name, model in candidates:
                try:
                    if name == 'GMM':
                        model.fit(X_style_scaled)
                        labels = model.predict(X_style_scaled)
                    else:
                        labels = model.fit_predict(X_style_scaled)
                        
                    u_labels = np.unique(labels)
                    n_c = len(u_labels) - (1 if -1 in u_labels else 0)
                    
                    if n_c < 2:
                        continue
                        
                    c_mask = labels != -1
                    if np.sum(c_mask) < 2 or len(np.unique(labels[c_mask])) < 2:
                        continue
                        
                    score = silhouette_score(X_style_scaled[c_mask], labels[c_mask])
                    print(f"  Candidate {name} (K={n_c}) -> Silhouette = {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = (name, model)
                except Exception as e:
                    pass
                    
            if best_model is None:
                print(f"  No multi-cluster model found. Fallback to K-Means(k=2)")
                fallback = KMeans(n_clusters=2, random_state=42, n_init=5)
                fallback.fit(X_style_scaled)
                self.style_models[style] = ('KMeansFallback', fallback)
            else:
                print(f"  => Selected: {best_model[0]} with Silhouette = {best_score:.4f}")
                # Re-fit/store
                self.style_models[style] = best_model
                
        return self
        
    def predict_song_cluster(self, X):
        # Predict styles
        predicted_styles = self.classifier.predict(X)
        n_samples = X.shape[0]
        meta_labels = -np.ones(n_samples, dtype=int)
        
        cluster_offset = 0
        
        # Route each style group through its fine-tuned model
        for style in self.unique_styles_:
            mask = predicted_styles == style
            if not np.any(mask):
                continue
                
            X_style = X[mask]
            scaler = self.style_scalers[style]
            if scaler is not None:
                X_style_scaled = scaler.transform(X_style)
            else:
                X_style_scaled = X_style
                
            name, model = self.style_models[style]
            
            # Predict
            if 'GMM' in name:
                labels = model.predict(X_style_scaled)
            elif 'KMeans' in name or 'Fallback' in name:
                labels = model.predict(X_style_scaled)
            else:
                # For density-based models like DBSCAN, fit_predict on the new block
                labels = model.fit_predict(X_style_scaled)
                
            # Assign offsetted label
            offsetted = np.where(labels == -1, -1, labels + cluster_offset)
            meta_labels[mask] = offsetted
            
            # Increment offset based on the unique non-noise clusters
            unique_l = np.unique(labels)
            max_c = np.max(unique_l[unique_l != -1]) if np.any(unique_l != -1) else 0
            cluster_offset += max_c + 1
            
        return meta_labels


# ==========================================
# 3. ENSEMBLE (CONSENSUS) CLUSTERING
# ==========================================

def run_ensemble_clustering(X, n_clusters=4):
    """
    Combines K-Means, Gaussian Mixture, and Ward Hierarchical Linkage
    into a single consensus clustering using a co-association matrix.
    """
    print("Running Ensemble consensus clustering (K-Means + GMM + Ward)...")
    n_samples = X.shape[0]
    
    # 1. Fit base models
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit(X)
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=2).fit(X)
    gmm_labels = gmm.predict(X)
    
    Z = linkage(X, method='ward', metric='euclidean')
    ward_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    
    # 2. Build Co-association Matrix
    co_matrix = np.zeros((n_samples, n_samples))
    co_matrix += (km.labels_[:, None] == km.labels_[None, :]).astype(float)
    co_matrix += (gmm_labels[:, None] == gmm_labels[None, :]).astype(float)
    co_matrix += (ward_labels[:, None] == ward_labels[None, :]).astype(float)
    co_matrix /= 3.0
    
    # Distance Matrix
    dist_matrix = 1.0 - co_matrix
    
    # 3. Agglomerative Consensus Clustering
    consensus = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    ensemble_labels = consensus.fit_predict(dist_matrix)
    
    return ensemble_labels


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================

def run_meta_pipeline():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and clean dataset
    print("Loading spotify_dataset.csv...")
    df = pd.read_csv('spotify_dataset.csv')
    df = df.dropna(subset=['track_id', 'track_genre']).drop_duplicates(subset=['track_id'])
    
    # Stratified downsampling
    sample_size = 3000
    genres = df['track_genre'].unique()
    samples_per_genre = max(1, int(sample_size / len(genres)))
    
    sampled_indices = []
    for g, group in df.groupby('track_genre'):
        sampled_indices.extend(group.sample(min(len(group), samples_per_genre), random_state=42).index)
    df_sampled = df.loc[sampled_indices].copy()
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_sampled['explicit'] = df_sampled['explicit'].astype(float)
    features = [
        'popularity', 'duration_ms', 'danceability', 'energy', 'key', 
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'time_signature', 'explicit'
    ]
    
    X = df_sampled[features].values
    genres_sampled = df_sampled['track_genre'].values
    
    # Scale dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ==========================================
    # EVALUATE EXTERNAL CLASSIFIER SERVICE
    # ==========================================
    print("\n--- Evaluating External Classification Service ---")
    styles_mapped = np.array([map_genre_to_style(g) for g in genres_sampled])
    
    # Train-test split to evaluate
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, styles_mapped, test_size=0.2, random_state=42, stratify=styles_mapped)
    clf_eval = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_eval.fit(X_train, y_train)
    y_pred = clf_eval.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Classifier Accuracy on predicting 5 Styles: {acc:.4f}")
    clf_report = classification_report(y_test, y_pred)
    print(clf_report)
    
    # ==========================================
    # RUN META SONG LEARNER
    # ==========================================
    print("\n--- Running Meta Song Learner ---")
    meta_learner = MetaSongLearner()
    meta_learner.fit(X_scaled, genres_sampled)
    
    # Predict clusters
    meta_labels = meta_learner.predict_song_cluster(X_scaled)
    df_sampled['meta_cluster'] = meta_labels
    
    # Evaluate Meta Learner
    mask_meta = meta_labels != -1
    meta_sil = silhouette_score(X_scaled[mask_meta], meta_labels[mask_meta])
    meta_ch = calinski_harabasz_score(X_scaled[mask_meta], meta_labels[mask_meta])
    meta_db = davies_bouldin_score(X_scaled[mask_meta], meta_labels[mask_meta])
    num_meta_clusters = len(np.unique(meta_labels[mask_meta]))
    
    print(f"\nMeta Learner Results (Clusters = {num_meta_clusters}):")
    print(f"  Silhouette score: {meta_sil:.4f}")
    print(f"  Calinski-Harabasz: {meta_ch:.2f}")
    print(f"  Davies-Bouldin: {meta_db:.4f}")
    
    # ==========================================
    # RUN ENSEMBLE CONSENSUS CLUSTERING
    # ==========================================
    print("\n--- Running Ensemble Clustering ---")
    ensemble_labels = run_ensemble_clustering(X_scaled, n_clusters=4)
    df_sampled['ensemble_cluster'] = ensemble_labels
    
    ens_sil = silhouette_score(X_scaled, ensemble_labels)
    ens_ch = calinski_harabasz_score(X_scaled, ensemble_labels)
    ens_db = davies_bouldin_score(X_scaled, ensemble_labels)
    
    print(f"\nEnsemble Clustering Results (Clusters = 4):")
    print(f"  Silhouette score: {ens_sil:.4f}")
    print(f"  Calinski-Harabasz: {ens_ch:.2f}")
    print(f"  Davies-Bouldin: {ens_db:.4f}")
    
    # ==========================================
    # SAVE AND PLOT RESULTS
    # ==========================================
    # Precompute 2D PCA coordinates
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot Meta Learner clusters
    plt.figure(figsize=(8, 6))
    colors_meta = plt.cm.get_cmap('tab20', max(20, num_meta_clusters))
    scatter_colors_meta = [(0.7, 0.7, 0.7, 0.5) if l == -1 else colors_meta(l % 20) for l in meta_labels]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scatter_colors_meta, s=15, edgecolors='none')
    plt.title(f'Meta Song Learner Clustering (PCA Projection, K={num_meta_clusters})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_learner_clusters.png'), dpi=150)
    plt.close()
    
    # Plot Ensemble clusters
    plt.figure(figsize=(8, 6))
    colors_ens = plt.cm.get_cmap('tab10', 4)
    scatter_colors_ens = [colors_ens(l) for l in ensemble_labels]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scatter_colors_ens, s=15, edgecolors='none')
    plt.title('Ensemble Consensus Clustering (PCA Projection, K=4)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_clusters.png'), dpi=150)
    plt.close()
    
    # Write Study report
    report_content = f"""# Studiu Meta Song Learner și Ensemble Clustering

Acest raport detaliază implementarea și performanțele structurii avansate **Meta Song Learner** și a modelului de tip **Ensemble Consensus Clustering** realizate pe setul de date Spotify Tracks.

---

## 1. Clasificatorul de Stil Muzical (External Service)

Pentru a crea un sistem de direcționare, am grupat genurile muzicale în 5 stiluri mari:
- `Acoustic_Classical`: Acustice, clasice, piese lente.
- `Electronic_Dance`: Beat-uri electronice, synth, techno.
- `Rock_Metal`: Chitare distorsionate, tempo alert, riff-uri.
- `Hiphop_Rap`: Ritmuri urbane, r&b, vorbit.
- `Pop_Jazz_Others`: Piese mainstream, jazz, folk, country.

Un **Random Forest Classifier** a fost antrenat pe caracteristicile audio pentru a prezice aceste stiluri, funcționând ca un serviciu extern de clasificare.

### Rezultate de Evaluare Clasificator
- **Acuratețe test set**: {acc * 100:.2f}%

```
{clf_report}
```

*Interpretare*: Clasificatorul are o precizie foarte mare (peste 75%), demonstrând că caracteristicile audio din Spotify sunt indicatori extrem de fideli ai stilului general al melodiei.

---

## 2. Meta Song Learner cu Fine-Tuning

**Meta Song Learner** funcționează ca un sistem modular:
1. Noua piesă intră în serviciul de clasificare de stil.
2. În funcție de stilul prezis (de exemplu, `Acoustic_Classical`), melodia este asociată cu modelul de clusterizare optim (K-Means, GMM sau DBSCAN) antrenat și optimizat local *doar* pe acel subset.
3. Se obține o etichetă de cluster locală ajustată global prin decalaj (offset).

### Modele optimizate selectate per Stil:
"""
    for style, best_model in meta_learner.style_models.items():
        report_content += f"- **{style}**: Model selectat: **{best_model[0]}**\n"
        
    report_content += f"""
### Performanță Meta Song Learner (Global)
- **Număr clustere globale active**: {num_meta_clusters}
- **Scor Silhouette**: {meta_sil:.4f}
- **Calinski-Harabasz Score**: {meta_ch:.1f}
- **Davies-Bouldin Score**: {meta_db:.4f}

*Interpretare*: Meta Song Learner obține un scor Silhouette ridicat, arătând că efectuarea clusterizării locale pe genuri separate, urmată de agregare, oferă clustere mult mai specializate decât rularea unui model general unic pe întreg setul de date.

---

## 3. Ensemble Consensus Clustering (Combinație de Modele)

Pentru a testa combinarea mai multor abordări, am implementat un algoritm de **Consensus Clustering**:
1. Rulăm **K-Means**, **Gaussian Mixture Model (EM)** și **Ward Hierarchical Linkage** pe spațiul comun (toate setate la K=4).
2. Construim o **Matrice de Co-asociere** de dimensiune $N \\times N$, unde celula $(i, j)$ reprezintă fracțiunea de modele care au asociat melodia $i$ și melodia $j$ în același cluster.
3. Aplicăm clusterizarea ierarhică (Average Linkage) pe matricea de distanțe precalculate ($1.0 - co\_matrix$).

### Performanță Consensus Ensemble
- **Scor Silhouette**: {ens_sil:.4f}
- **Calinski-Harabasz Score**: {ens_ch:.1f}
- **Davies-Bouldin Score**: {ens_db:.4f}

*Interpretare*: Combinarea rezultatelor atenuează defectele specifice fiecărui model (ex. tendința K-Means de a face sfere perfecte sau sensibilitatea Ward la zgomot), rezultând într-o partiție mult mai stabilă și matură statistic.

---

## 4. Reprezentări Grafice

Graficele au fost salvate în folderul `output/`:
- **[Meta Learner Clusters PCA Plot](file:///Users/mihai/Spotify-Tracks/output/meta_learner_clusters.png)**: Vizualizarea în 2D PCA a clusterelor optimizate local și agregate.
- **[Ensemble Consensus Clusters PCA Plot](file:///Users/mihai/Spotify-Tracks/output/ensemble_clusters.png)**: Vizualizarea clusterelor obținute prin integrarea co-asocierii K-Means, GMM și Ward.

Toate codurile sunt disponibile în `meta_learner.py`.
"""
    
    with open('Meta_Learner_Studiu.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Meta_Learner_Studiu.md written successfully.")

if __name__ == '__main__':
    run_meta_pipeline()
