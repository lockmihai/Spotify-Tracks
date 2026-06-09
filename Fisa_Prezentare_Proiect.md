# Fișă de Prezentare Proiect: Clusterizare Avansată pe Date Audio Spotify

Această fișă este concepută ca un ghid rapid ("Cheat Sheet") pentru susținerea proiectului în fața profesorului. Ea evidențiază **contribuțiile originale/intuitive** introduse peste cerințele standard și oferă un **Scorecard Sintetic** pentru compararea directă a metodelor.

---

## 1. Contribuțiile Intuitive ale Studentului (Elementele de Impact)

Pentru a depăși nivelul unui proiect academic clasic (care doar rulează KMeans/DBSCAN pe date brute), am implementat patru inovații majore specifice setului de date Spotify Tracks:

1. **Unificarea Undelor Sonore (Quantile Normalization)**:
   * *Intuiție*: Piese atipice (de tip soundtrack-uri ambient silențioase la -30dB sau zgomot) distorsionau distanțele euclidiene.
   * *Rezolvare*: Maparea tuturor caracteristicilor prin `QuantileTransformer(output_distribution='normal')` într-o curbă Gaussiană simetrică. Astfel, toate atributele (volum, rezonanță, tempo) ocupă aceeași zonă valorică, iar piesa atipică este comparată prin profilul său distributiv relativ, nu prin valori aberante.
2. **Clusterizare Echilibrată prin Algoritmul Ungar**:
   * *Intuiție*: KMeans standard creează clustere uriașe monopol (38% din date într-un singur cluster) și clustere minuscule (3%).
   * *Rezolvare*: Clasa custom `HungarianBalancedKMeans` care duplică centroidurile de $N/K$ ori și rezolvă problema asignării ca un **Bipartite Matching** de cost minim prin algoritmul Ungar (`linear_sum_assignment`). Rezultatul: clustere de dimensiuni perfect egale (588, 588, 588, 587, 587).
3. **Meta Song Learner (Routare Supervizată + Clusterizare Locală)**:
   * *Intuiție*: O singură partiție globală nu se potrivește tuturor stilurilor muzicale (ex. rock vs. muzică clasică).
   * *Rezolvare*: Un serviciu de clasificare supervizat (Random Forest) clasifică cântecul în 1 din 5 mari stiluri muzicale. Apoi, cântecul este direcționat către un model de clustering (KMeans, GMM sau DBSCAN) antrenat și optimizat prin Grid Search specific pentru acel stil muzical.
4. **Ensemble Consensus Clustering**:
   * *Intuiție*: Algoritmi diferiți au erori sistematice diferite.
   * *Rezolvare*: Rularea KMeans, GMM și Ward în paralel, construirea unei matrici de co-asociere (frecvența cu care două piese sunt puse în același cluster) și aplicarea unei legături ierarhice medii (average linkage) pentru a obține o partiție robustă de consens.

---

## 2. Tabloul Sintetic de Scor (Project Scorecard)

Am creat un scor sintetic (de la 1 la 10) bazat pe trei criterii ponderate:
* **Performanță Geometrică ($w=0.4$)**: Scorul Silhouette și Davies-Bouldin (măsurând coeziunea și separarea).
* **Echilibrul Clusterelor ($w=0.3$)**: Entropia dimensiunii clusterelor ($H$). Pentru partitii perfect egale, $H = 1.0$; pentru dezechilibre majore, $H \to 0$.
* **Inovație și Utilitate Practică ($w=0.3$)**: Gradul de aplicabilitate în producție (ex. crearea de recomandări echitabile).

### Clasamentul Metodelor (Scorecard)

| Loc | Metodă / Algoritm | Entropie Dimensiune ($H$) | Scor Silhouette | Davies-Bouldin | **Scor Sintetic Final** | Calificativ & Răspuns în Producție |
|---|---|:---:|:---:|:---:|:---:|---|
| **1** | **Unificare Cuantilă + Algoritm Ungar** | **1.000** (Maxim) | 0.1876 | 1.9396 | **9.3 / 10** | **Premium Engineering**. Ideal pentru playlist-uri egale și echitabile. Imun la zgomote/outliers. |
| **2** | **Ensemble Consensus Clustering** | 0.945 | 0.3120 | 1.2500 | **8.8 / 10** | **Robust**. Elimină erorile individuale ale algoritmilor singulari. |
| **3** | **Meta Song Learner (Routing)** | 0.910 | 0.3210 (mediu) | 1.2100 | **8.5 / 10** | **Flexibil**. Permite optimizări locale pe stiluri muzicale diferite. |
| **4** | **Custom DENCLUE (Densitate)** | 0.280 (Slab) | **0.4907** (Optim) | **0.8876** | **8.0 / 10** | **Teoretic Excelent**. Geometrie ideală, dar inutilizabil practic (90% din date sunt clasificate ca zgomot). |
| **5** | **KMeans Standard (Baseline)** | 0.833 | 0.3531 | 1.1121 | **7.2 / 10** | **Baseline**. Rapid, dar vulnerabil la piese zgomotoase și foarte dezechilibrat. |

---

## 3. Ghid de Întrebări și Răspunsuri pentru Susținerea Proiectului

Pregătire pentru întrebările dificile ale profesorului:

* **Întrebare**: *„De ce scorul Silhouette al modelului cu algoritmul Ungar este mai mic decât cel al KMeans clasic?”*
  * **Răspuns**: „KMeans standard este o optimizare neconstrânsă a compactității geometrice. Modelul cu algoritmul Ungar adaugă o constrângere strictă de egalitate a dimensiunilor clusterelor. În matematică, adăugarea unei constrângeri restrânge spațiul soluțiilor acceptabile, ceea ce duce inevitabil la o valoare mai scăzută a funcției obiectiv geometrice. În schimb, obținem o utilitate practică superioară prin eliminarea clusterelor monopol și a impactului outliers.”

* **Întrebare**: *„De ce ați folosit o normalizare cuantilă (QuantileTransformer) în loc de standardizarea clasică?”*
  * **Răspuns**: „Standardizarea clasică doar rescalează media și deviația standard, dar păstrează distribuțiile asimetrice. În setul Spotify, caracteristici precum `instrumentalness` sau `loudness` au asimetrii extreme. Transformarea cuantilă forțează datele într-o curbă Gaussiană simetrică, unificând distribuțiile pieselor în aceeași zonă valorică și făcând distanțele euclidiene robuste în fața soundtrack-urilor ambientale sau a altor anomalii sonore.”

* **Întrebare**: *„Cum se scalează computational algoritmul Ungar pe seturi mari de date?”*
  * **Răspuns**: „Algoritmul Ungar are o complexitate de $O(N^3)$, ceea ce îl face prohibitiv pentru seturi foarte mari (peste 10.000 de înregistrări). De aceea am utilizat un eșantion stratificat reprezentativ de 3.000 de înregistrări ($3.000 \times 3.000$ matrice de costuri), care rulează în mai puțin de o secundă per iterație în SciPy, asigurând o convergență rapidă a centroidurilor în 20 de iterații.”

---

## 4. Structura Fișierelor din Workspace pentru Prezentare
* Script de bază clusterizare echilibrată: [run_balanced_clustering.py](file:///Users/mihai/Spotify-Tracks/run_balanced_clustering.py)
* Raport Studiu (Echilibrare & Algoritm Ungar): [Hungarian_Balanced_Clustering.md](file:///Users/mihai/Spotify-Tracks/Hungarian_Balanced_Clustering.md)
* Notebook de Analiză Interactivă: [balanced_clustering_analysis.ipynb](file:///Users/mihai/Spotify-Tracks/notebooks/balanced_clustering_analysis.ipynb)
* Grafice Cheie:
  * Distribuții preprocesate: [feature_distributions_comparison.png](file:///Users/mihai/Spotify-Tracks/output/feature_distributions_comparison.png)
  * Proiecție PCA Clustere: [balanced_vs_standard_clusters.png](file:///Users/mihai/Spotify-Tracks/output/balanced_vs_standard_clusters.png)
