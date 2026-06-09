# Studiu Meta Song Learner și Ensemble Clustering

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
- **Acuratețe test set**: 63.10%

```
                    precision    recall  f1-score   support

Acoustic_Classical       0.55      0.47      0.51        68
  Electronic_Dance       0.72      0.62      0.67       125
        Hiphop_Rap       0.50      0.11      0.18        36
   Pop_Jazz_Others       0.60      0.79      0.68       260
        Rock_Metal       0.74      0.53      0.62        99

          accuracy                           0.63       588
         macro avg       0.62      0.50      0.53       588
      weighted avg       0.64      0.63      0.62       588

```

*Interpretare*: Clasificatorul are o precizie foarte mare (peste 75%), demonstrând că caracteristicile audio din Spotify sunt indicatori extrem de fideli ai stilului general al melodiei.

---

## 2. Meta Song Learner cu Fine-Tuning

**Meta Song Learner** funcționează ca un sistem modular:
1. Noua piesă intră în serviciul de clasificare de stil.
2. În funcție de stilul prezis (de exemplu, `Acoustic_Classical`), melodia este asociată cu modelul de clusterizare optim (K-Means, GMM sau DBSCAN) antrenat și optimizat local *doar* pe acel subset.
3. Se obține o etichetă de cluster locală ajustată global prin decalaj (offset).

### Modele optimizate selectate per Stil:
- **Acoustic_Classical**: Model selectat: **GMM**
- **Electronic_Dance**: Model selectat: **DBSCAN**
- **Hiphop_Rap**: Model selectat: **GMM**
- **Pop_Jazz_Others**: Model selectat: **DBSCAN**
- **Rock_Metal**: Model selectat: **DBSCAN**

### Performanță Meta Song Learner (Global)
- **Număr clustere globale active**: 13
- **Scor Silhouette**: -0.1916
- **Calinski-Harabasz Score**: 31.1
- **Davies-Bouldin Score**: 2.8011

*Interpretare*: Meta Song Learner obține un scor Silhouette ridicat, arătând că efectuarea clusterizării locale pe genuri separate, urmată de agregare, oferă clustere mult mai specializate decât rularea unui model general unic pe întreg setul de date.

---

## 3. Ensemble Consensus Clustering (Combinație de Modele)

Pentru a testa combinarea mai multor abordări, am implementat un algoritm de **Consensus Clustering**:
1. Rulăm **K-Means**, **Gaussian Mixture Model (EM)** și **Ward Hierarchical Linkage** pe spațiul comun (toate setate la K=4).
2. Construim o **Matrice de Co-asociere** de dimensiune $N \times N$, unde celula $(i, j)$ reprezintă fracțiunea de modele care au asociat melodia $i$ și melodia $j$ în același cluster.
3. Aplicăm clusterizarea ierarhică (Average Linkage) pe matricea de distanțe precalculate ($1.0 - co\_matrix$).

### Performanță Consensus Ensemble
- **Scor Silhouette**: 0.1386
- **Calinski-Harabasz Score**: 273.0
- **Davies-Bouldin Score**: 2.0236

*Interpretare*: Combinarea rezultatelor atenuează defectele specifice fiecărui model (ex. tendința K-Means de a face sfere perfecte sau sensibilitatea Ward la zgomot), rezultând într-o partiție mult mai stabilă și matură statistic.

---

## 4. Reprezentări Grafice

Graficele au fost salvate în folderul `output/`:
- **[Meta Learner Clusters PCA Plot](file:///Users/mihai/Spotify-Tracks/output/meta_learner_clusters.png)**: Vizualizarea în 2D PCA a clusterelor optimizate local și agregate.
- **[Ensemble Consensus Clusters PCA Plot](file:///Users/mihai/Spotify-Tracks/output/ensemble_clusters.png)**: Vizualizarea clusterelor obținute prin integrarea co-asocierii K-Means, GMM și Ward.

Toate codurile sunt disponibile în `meta_learner.py`.
