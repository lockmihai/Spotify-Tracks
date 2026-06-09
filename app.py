import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN, OPTICS, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import Parallel, delayed

# Import custom algorithms
from run_clustering import CustomSTING, CustomDENCLUE
from run_balanced_clustering import HungarianBalancedKMeans
from meta_learner import MetaSongLearner, map_genre_to_style

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Spotify Tracks Clustering Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #111216;
        color: #fafafa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1f29;
        border-radius: 4px;
        color: #fafafa;
        font-weight: 600;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2e303f;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1db954;
        color: #111216;
    }
    h1, h2, h3 {
        color: #1db954 !important;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_style=True)

# ==========================================
# CACHED DATA LOADING & SAMPLING
# ==========================================
@st.cache_data
def load_base_data():
    dataset_path = "spotify_dataset.csv"
    if not os.path.exists(dataset_path):
        # Fallback for parent directories
        dataset_path = "../spotify_dataset.csv"
    if not os.path.exists(dataset_path):
        return pd.DataFrame()
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['track_id', 'track_genre'])
    df = df.drop_duplicates(subset=['track_id'])
    return df

@st.cache_data
def get_stratified_sample(sample_size):
    df = load_base_data()
    if df.empty:
        return df
    genres = df['track_genre'].unique()
    samples_per_genre = max(1, int(sample_size / len(genres)))
    
    sampled_indices = []
    for g, group in df.groupby('track_genre'):
        sampled_indices.extend(group.sample(min(len(group), samples_per_genre), random_state=42).index)
    df_sampled = df.loc[sampled_indices].copy()
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    df_sampled['explicit'] = df_sampled['explicit'].astype(float)
    return df_sampled

# Load data once
df_raw = load_base_data()
if df_raw.empty:
    st.error("Nu s-a găsit fișierul spotify_dataset.csv în workspace. Rulați descărcarea sau asigurați-vă că fișierul se află în directorul rădăcină.")
    st.stop()

# ==========================================
# HEADER
# ==========================================
st.title("🎵 Spotify Tracks Clustering & Cloud Tuning Studio")
st.write("O platformă de analiză interactivă și optimizare paralelă a algoritmilor de clusterizare pe caracteristici audio Spotify.")

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=80)
st.sidebar.title("Navigare Studio")
app_tab = st.sidebar.radio(
    "Alegeți Secțiunea:",
    ["Dashboard Comparativ", "Unificare & Algoritm Ungar", "Meta Song Learner", "Cloud Parameter Tuning Sandbox"]
)

features = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'key', 
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'time_signature', 'explicit'
]

# ==========================================
# TAB 1: COMPARATIVE DASHBOARD
# ==========================================
if app_tab == "Dashboard Comparativ":
    st.header("📊 Dashboard Comparativ de Clusterizare")
    st.write("Comparați performanțele a 10+ algoritmi cu opțiuni interactive de scalare și eșantionare.")
    
    col_settings, col_metrics = st.columns([1, 3])
    
    with col_settings:
        st.subheader("Configurație")
        sample_size = st.slider("Dimensiune Eșantion Stratificat:", 500, 5000, 2000, step=100)
        scaling_method = st.selectbox("Metodă Scalare:", ["StandardScaler", "QuantileTransformer (Normal)"])
        
        algorithm = st.selectbox(
            "Algoritm:",
            ["KMeans", "BisectingKMeans", "Gaussian Mixture (EM)", "DBSCAN", "OPTICS", "BIRCH", "Custom DENCLUE", "Custom STING"]
        )
        
        # Hyperparameters depending on algorithm
        st.write("---")
        st.write("**Hiperparametri Algoritm:**")
        if algorithm in ["KMeans", "BisectingKMeans", "Gaussian Mixture (EM)"]:
            n_clusters = st.slider("Număr Clustere (K):", 2, 10, 5)
        elif algorithm == "DBSCAN":
            eps = st.slider("Epsilon (eps):", 0.1, 5.0, 1.5, step=0.1)
            min_samples = st.slider("Min Samples:", 1, 20, 5)
        elif algorithm == "OPTICS":
            min_samples = st.slider("Min Samples:", 2, 20, 5)
        elif algorithm == "BIRCH":
            n_clusters = st.slider("Număr Clustere (K):", 2, 10, 5)
            threshold = st.slider("Threshold:", 0.1, 2.0, 0.5, step=0.1)
        elif algorithm == "Custom DENCLUE":
            h = st.slider("Bandwidth (h):", 0.1, 2.0, 0.75, step=0.05)
            min_density = st.slider("Min Density Threshold:", 0.0001, 0.1, 0.001, step=0.001, format="%.4f")
        elif algorithm == "Custom STING":
            grid_size = st.slider("Grid Size (N x N):", 5, 20, 10)
            min_samples = st.slider("Min Samples per Cell:", 2, 15, 5)
            
    # Load and scale data
    df_sampled = get_stratified_sample(sample_size)
    X = df_sampled[features].values
    
    if scaling_method == "StandardScaler":
        X_scaled = StandardScaler().fit_transform(X)
    else:
        X_scaled = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42).fit_transform(X)
        
    # Run selected algorithm
    labels = None
    t0 = time.time()
    
    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "BisectingKMeans":
        model = BisectingKMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "Gaussian Mixture (EM)":
        model = GaussianMixture(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "OPTICS":
        model = OPTICS(min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "BIRCH":
        model = Birch(n_clusters=n_clusters, threshold=threshold)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "Custom DENCLUE":
        model = CustomDENCLUE(h=h, min_density=min_density, max_iter=30)
        model.fit(X_scaled)
        labels = model.labels_
    elif algorithm == "Custom STING":
        model = CustomSTING(grid_size=grid_size, min_samples=min_samples)
        model.fit(X_scaled)
        labels = model.labels_
        
    execution_time = time.time() - t0
    
    # Calculate metrics
    # Filter noise for metrics calculation if density-based
    valid_mask = labels != -1
    num_valid_clusters = len(np.unique(labels[valid_mask])) if np.any(valid_mask) else 0
    
    # Plot results
    with col_metrics:
        tab_plot, tab_dist = st.tabs(["Proiecție 2D Interactivă (PCA)", "Distribuție Clustere și Statistici"])
        
        with tab_plot:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            df_plot = df_sampled.copy()
            df_plot['PC1'] = X_pca[:, 0]
            df_plot['PC2'] = X_pca[:, 1]
            df_plot['Cluster'] = [f"Cluster {l}" if l != -1 else "Noise/Zgomot" for l in labels]
            
            fig = px.scatter(
                df_plot, x='PC1', y='PC2', color='Cluster',
                hover_data=['track_name', 'artists', 'track_genre', 'popularity', 'tempo'],
                title=f"Proiecție PCA 2D - {algorithm} (Timp Execuție: {execution_time:.3f}s)",
                color_discrete_sequence=px.colors.qualitative.Dark24,
                opacity=0.75
            )
            fig.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_dist:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                # Count sizes
                unique, counts = np.unique(labels, return_counts=True)
                df_counts = pd.DataFrame({'Cluster': [f"Cluster {u}" if u != -1 else "Zgomot" for u in unique], 'Dimensiune': counts})
                fig_bar = px.bar(df_counts, x='Cluster', y='Dimensiune', color='Cluster', title="Dimensiunea Clusterelor")
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_d2:
                st.write("**Metrici Validare Geometrice:**")
                if num_valid_clusters >= 2:
                    sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
                    ch = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
                    db = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
                    
                    st.metric("Scor Silhouette (Coeziune)", f"{sil:.4f}")
                    st.metric("Calinski-Harabasz (Separare)", f"{ch:.2f}")
                    st.metric("Davies-Bouldin (Ideal mic)", f"{db:.4f}")
                else:
                    st.warning("Metricile nu pot fi calculate: mai puțin de 2 clustere valide identificate (excluzând zgomotul).")

# ==========================================
# TAB 2: HUNGARIAN BALANCED CLUSTERING
# ==========================================
elif app_tab == "Unificare & Algoritm Ungar":
    st.header("⚖️ Echilibrarea Clusterelor & Unificarea Undelor Sonore")
    st.write("Această secțiune prezintă aplicarea algoritmului Ungar pentru a crea clustere de dimensiuni perfect egale și efectul transformării cuantile.")
    
    sample_size_bal = st.sidebar.slider("Dimensiune Eșantion Bipartit:", 1000, 3000, 2000, step=500)
    K_bal = st.sidebar.slider("Număr Clustere Echilibrate (K):", 3, 6, 5)
    
    st.write("### 1. Distribuția Undelor Sonore: StandardScaler vs QuantileTransformer")
    st.info("Observați cum transformarea cuantilă elimină formele extrem de asimetrice ale distribuțiilor audio, forțând o curbă normală care aduce toate caracteristicile în aceeași zonă valorică.")
    
    df_bal = get_stratified_sample(sample_size_bal)
    X_bal = df_bal[features].values
    
    X_std = StandardScaler().fit_transform(X_bal)
    X_q = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42).fit_transform(X_bal)
    
    feat_to_plot = st.selectbox("Alegeți Caracteristica Audio pentru Vizualizare:", ['loudness', 'instrumentalness', 'acousticness', 'energy', 'tempo'])
    feat_idx = features.index(feat_to_plot)
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        fig_std = px.histogram(pd.DataFrame({feat_to_plot: X_std[:, feat_idx]}), x=feat_to_plot, kde=True, title=f"{feat_to_plot} (StandardScaler)")
        fig_std.update_layout(template="plotly_dark")
        st.plotly_chart(fig_std, use_container_width=True)
    with col_plot2:
        fig_q = px.histogram(pd.DataFrame({feat_to_plot: X_q[:, feat_idx]}), x=feat_to_plot, kde=True, title=f"{feat_to_plot} (QuantileTransformer - Normal)", color_discrete_sequence=['salmon'])
        fig_q.update_layout(template="plotly_dark")
        st.plotly_chart(fig_q, use_container_width=True)
        
    st.write("---")
    st.write("### 2. Comparație Clusterizare: Standard KMeans vs. Hungarian Balanced KMeans")
    
    if st.button("Rulează Clusterizarea Echilibrată (Se rezolvă Bipartite Matching live)"):
        with st.spinner("Se rulează algoritmul Ungar..."):
            # Fit KMeans standard
            km = KMeans(n_clusters=K_bal, random_state=42, n_init=5)
            km_labels = km.fit_predict(X_q)
            
            # Fit Hungarian Balanced KMeans
            hb = HungarianBalancedKMeans(n_clusters=K_bal, max_iter=15, random_state=42)
            hb.fit(X_q)
            hb_labels = hb.labels_
            
            # Counts
            km_counts = np.bincount(km_labels, minlength=K_bal)
            hb_counts = np.bincount(hb_labels, minlength=K_bal)
            
            # Plot comparisons
            col_b1, col_b2 = st.columns(2)
            
            pca_bal = PCA(n_components=2, random_state=42)
            X_pca_bal = pca_bal.fit_transform(X_q)
            
            with col_b1:
                df_km = df_bal.copy()
                df_km['PC1'] = X_pca_bal[:, 0]
                df_km['PC2'] = X_pca_bal[:, 1]
                df_km['Cluster'] = [f"Cluster {l} ({km_counts[l]} piese)" for l in km_labels]
                fig_km = px.scatter(df_km, x='PC1', y='PC2', color='Cluster', title="KMeans Standard (Dezechilibrat)", hover_data=['track_name', 'artists'])
                fig_km.update_layout(template="plotly_dark")
                st.plotly_chart(fig_km, use_container_width=True)
                
            with col_b2:
                df_hb = df_bal.copy()
                df_hb['PC1'] = X_pca_bal[:, 0]
                df_hb['PC2'] = X_pca_bal[:, 1]
                df_hb['Cluster'] = [f"Cluster {l} ({hb_counts[l]} piese)" for l in hb_labels]
                fig_hb = px.scatter(df_hb, x='PC1', y='PC2', color='Cluster', title="Hungarian Balanced KMeans (Perfect Egal)", hover_data=['track_name', 'artists'])
                fig_hb.update_layout(template="plotly_dark")
                st.plotly_chart(fig_hb, use_container_width=True)
                
            # Metrics comparison table
            st.subheader("Analiză Comparativă a Metricilor")
            metrics_data = {
                "Model/Algoritm": ["Standard KMeans (Dezechilibrat)", "Hungarian Balanced KMeans (Echilibrat)"],
                "Dimensiuni Clustere": [str(list(km_counts)), str(list(hb_counts))],
                "Scor Silhouette": [f"{silhouette_score(X_q, km_labels):.4f}", f"{silhouette_score(X_q, hb_labels):.4f}"],
                "Davies-Bouldin (Mic=Ideal)": [f"{davies_bouldin_score(X_q, km_labels):.4f}", f"{davies_bouldin_score(X_q, hb_labels):.4f}"]
            }
            st.table(pd.DataFrame(metrics_data))
            st.write("> **Explicație Academică**: Scorul geometric Silhouette este mai scăzut pentru modelul Hungarian deoarece adăugarea constrângerii de dimensiune egală restrânge spațiul soluțiilor optime (unele piese de la graniță sunt forțate să treacă în alte clustere pentru a păstra balanța). În schimb, obținem o utilitate practică mult mai mare în crearea playlist-urilor.")

# ==========================================
# TAB 3: META SONG LEARNER
# ==========================================
elif app_tab == "Meta Song Learner":
    st.header("🧠 Simulator de Rutare: Meta Song Learner")
    st.write("Acest simulator prezintă logica din `meta_learner.py` antrenată live. O melodie este clasificată supervizat într-un stil mare, iar apoi este atribuită clusterului din modelul optim antrenat local pentru acel stil.")
    
    # Train Meta Learner on the sample
    with st.spinner("Se antrenează serviciul extern de clasificare și modelele locale de Grid Search (poate dura câteva secunde)..."):
        # Load representative sample
        df_meta_sample = get_stratified_sample(2500)
        X_meta = StandardScaler().fit_transform(df_meta_sample[features].values)
        genres_meta = df_meta_sample['track_genre'].values
        
        meta_learner = MetaSongLearner()
        meta_learner.fit(X_meta, genres_meta)
        
    st.success("Meta Song Learner este antrenat și pregătit pentru rutare!")
    
    st.write("### Testare Cântec Custom")
    col_input, col_routing = st.columns([1, 2])
    
    with col_input:
        st.write("**Alegeți Caracteristici Audio Cântec:**")
        # Generate sliders for prediction
        in_popularity = st.slider("Popularitate:", 0.0, 100.0, 50.0)
        in_duration = st.slider("Durată (ms):", 50000.0, 500000.0, 200000.0, step=10000.0)
        in_dance = st.slider("Dansabilitate:", 0.0, 1.0, 0.6)
        in_energy = st.slider("Energie:", 0.0, 1.0, 0.6)
        in_loudness = st.slider("Loudness (dB):", -40.0, 0.0, -8.0)
        in_speech = st.slider("Speechiness:", 0.0, 1.0, 0.05)
        in_acoustic = st.slider("Acousticness:", 0.0, 1.0, 0.2)
        in_instrumental = st.slider("Instrumentalness:", 0.0, 1.0, 0.0)
        in_liveness = st.slider("Liveness:", 0.0, 1.0, 0.15)
        in_valence = st.slider("Valence (Veselie):", 0.0, 1.0, 0.5)
        in_tempo = st.slider("Tempo (BPM):", 50.0, 220.0, 120.0)
        
        custom_song = np.array([
            in_popularity, in_duration, in_dance, in_energy, 5.0, # key = 5
            in_loudness, 1.0, in_speech, in_acoustic, in_instrumental, # mode = 1
            in_liveness, in_valence, in_tempo, 4.0, 0.0 # time_signature = 4, explicit = 0
        ])
        
    with col_routing:
        st.subheader("Vizualizare Rutare în Timp Real")
        
        # Scale input using fit parameters
        custom_song_scaled = (custom_song - meta_learner.scaler.mean_) / np.sqrt(meta_learner.scaler.var_)
        custom_song_scaled = custom_song_scaled.reshape(1, -1)
        
        # 1. Prediction via external style classifier
        predicted_style = meta_learner.style_classifier.predict(custom_song_scaled)[0]
        style_prob = meta_learner.style_classifier.predict_proba(custom_song_scaled)[0]
        style_idx = list(meta_learner.style_classifier.classes_).index(predicted_style)
        
        st.write(f"#### Pasul 1: Clasificatorul Supervizat de Stil (External Service)")
        st.write(f"Stil prezis: **{predicted_style}** (Confidență: {style_prob[style_idx]*100:.1f}%)")
        
        # Display probabilities bar chart
        df_prob = pd.DataFrame({
            'Stil Muzical': meta_learner.style_classifier.classes_,
            'Probabilitate': style_prob
        })
        fig_prob = px.bar(df_prob, x='Probabilitate', y='Stil Muzical', orientation='h', color='Probabilitate')
        fig_prob.update_layout(template="plotly_dark", height=200)
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # 2. Prediction via specialized clustering
        st.write(f"#### Pasul 2: Redirecționare către Clusterizator Local")
        best_model_name, best_model = meta_learner.style_models[predicted_style]
        st.write(f"Modelul local optim selectat prin Grid Search pentru stilul **{predicted_style}** este: `{best_model_name}`.")
        
        # Find cluster
        if best_model_name == "DBSCAN":
            # Find nearest point if DBSCAN
            labels_all = best_model.labels_
            mask = labels_all != -1
            if np.any(mask):
                sub_X = meta_learner.X_styles[predicted_style]
                distances = np.linalg.norm(sub_X - custom_song_scaled, axis=1)
                best_idx = np.argmin(distances)
                predicted_cluster = labels_all[best_idx]
            else:
                predicted_cluster = -1
        else:
            predicted_cluster = best_model.predict(custom_song_scaled)[0]
            
        st.markdown(f"""
        <div style="background-color:#1e2d24; border-left:6px solid #1db954; padding:15px; border-radius:4px;">
            <h4 style="margin:0; color:#1db954;">Rezultat Final Rutare:</h4>
            <p style="margin:5px 0 0 0; font-size:18px;">Piesa a fost atribuită la: <strong>Clusterul {predicted_cluster}</strong> din sub-grupul <strong>{predicted_style}</strong></p>
        </div>
        """, unsafe_style=True)

# ==========================================
# TAB 4: CLOUD TUNING SANDBOX
# ==========================================
elif app_tab == "Cloud Parameter Tuning Sandbox":
    st.header("⚙️ Cloud Parameter Fine-Tuning Sandbox")
    st.write("Utilizați resursele cloud multi-core pentru a executa un Grid Search masiv pe toți algoritmii selectați în paralel.")
    
    st.sidebar.subheader("Setări Grid Search")
    grid_sample_size = st.sidebar.slider("Piese pentru Fine-Tuning:", 1000, 4000, 2000, step=500)
    n_jobs = st.sidebar.slider("Core-uri Paralele (Jobs):", 1, 8, 4)
    
    selected_algs = st.multiselect(
        "Selectați algoritmii pentru Fine-Tuning în paralel:",
        ["KMeans", "Gaussian Mixture"],
        default=["KMeans", "Gaussian Mixture"]
    )
    
    st.write("### Configurare Spațiu Hiperparametri")
    k_range = st.slider("Interval pentru K (Număr Clustere):", 2, 12, (3, 8))
    
    if st.button("Lansează Fine-Tuning Paralel pe Google Cloud"):
        with st.spinner("Se execută Grid Search-ul paralel..."):
            # Prepare data
            df_grid = get_stratified_sample(grid_sample_size)
            X_grid = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42).fit_transform(df_grid[features].values)
            
            # Configurations
            configs = []
            for alg in selected_algs:
                for k in range(k_range[0], k_range[1] + 1):
                    configs.append((alg, k))
                    
            # Define function to evaluate a config
            def evaluate_config(alg, k, X_data):
                try:
                    if alg == "KMeans":
                        m = KMeans(n_clusters=k, random_state=42, n_init=3)
                        lbls = m.fit_predict(X_data)
                    elif alg == "Gaussian Mixture":
                        m = GaussianMixture(n_clusters=k, random_state=42, n_init=1)
                        lbls = m.fit_predict(X_data)
                    
                    # Compute score
                    sil = silhouette_score(X_data, lbls)
                    db = davies_bouldin_score(X_data, lbls)
                    return {
                        'Algoritm': alg,
                        'K': k,
                        'Silhouette': sil,
                        'Davies-Bouldin': db,
                        'Status': 'Succes'
                    }
                except Exception as e:
                    return {
                        'Algoritm': alg,
                        'K': k,
                        'Silhouette': -1.0,
                        'Davies-Bouldin': 99.0,
                        'Status': f'Eroare: {str(e)}'
                    }
            
            # Run parallel jobs
            t_grid_0 = time.time()
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_config)(alg, k, X_grid) for alg, k in configs
            )
            t_grid_total = time.time() - t_grid_0
            
            df_res = pd.DataFrame(results)
            st.success(f"Grid Search-ul paralel a finalizat evaluarea a {len(configs)} configurații în {t_grid_total:.2f} secunde!")
            
            # Best model selection
            df_success = df_res[df_res['Status'] == 'Succes']
            if not df_success.empty:
                best_config = df_success.loc[df_success['Silhouette'].idxmax()]
                st.markdown(f"""
                <div style="background-color:#1e2d24; border-left:6px solid #1db954; padding:15px; border-radius:4px; margin-bottom:20px;">
                    <h4 style="margin:0; color:#1db954;">Cea mai bună configurație găsită:</h4>
                    <p style="margin:5px 0 0 0; font-size:18px;">Algoritm: <strong>{best_config['Algoritm']}</strong> | K: <strong>{best_config['K']}</strong> | Scor Silhouette: <strong>{best_config['Silhouette']:.4f}</strong></p>
                </div>
                """, unsafe_style=True)
            
            # Display results leaderboard
            st.subheader("Leaderboard Performanță Hiperparametri")
            st.dataframe(df_res.sort_values(by='Silhouette', ascending=False).reset_index(drop=True), use_container_width=True)
            
            # Plot interactive comparison line chart
            fig_line = px.line(
                df_success, x='K', y='Silhouette', color='Algoritm', markers=True,
                title="Evoluția Scorului Silhouette în funcție de K și Algoritm"
            )
            fig_line.update_layout(template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)
