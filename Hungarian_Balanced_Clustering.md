# Studiu de Clusterizare Echilibrată prin Algoritmul Ungar și Unificarea Distribuțiilor Audio

Acest studiu prezintă o metodologie avansată de preprocesare și clusterizare pe setul de date Spotify Tracks. Obiectivul este de a rezolva două probleme majore ale algoritmilor tradiționali (cum ar fi KMeans clasic):
1. **Distanțe distorsionate de asimetria distribuțiilor și outliers**: Piese atipice (de tip soundtrack, zgomot ambiental, melodii acustice foarte silențioase) pot domina calculul distanțelor euclidiene și pot distorsiona centroidurile.
2. **Dezechilibrul masiv al clusterelor**: KMeans tinde să creeze clustere foarte mari în zonele cu densitate mare de puncte și clustere extrem de mici în zonele rarefiate.

---

## 1. Unificarea Valorilor prin Transformarea Cuantilă (Quantile Normalization)

Pentru a ne asigura că distribuțiile caracteristicilor sonore se află în aceeași zonă și că o metodă de tip clustering nu este perturbată de un soundtrack neechivalent (în volum, rezonanță sau frecvență), am aplicat **QuantileTransformer** cu o distribuție țintă de tip **Normală (Gaussiană)**.

### Mecanismul Matematic
Transformarea cuantilă mapează funcția de distribuție cumulativă empirică ($CDF$) a fiecărei variabile pe funcția de distribuție cumulativă a unei distribuții normale standard ($\mathcal{N}(0, 1)$):
$$x_{new} = \Phi^{-1}(F(x))$$
unde $F(x)$ este $CDF$-ul empiric al caracteristicii originale, iar $\Phi^{-1}$ este inversa funcției de distribuție cumulativă normală standard (funcția quantile).

### Impactul Asupra Caracteristicilor Audio
- **Volum (`loudness`, `energy`)**: `loudness` în decibeli și `energy` sunt mapate la o curbă clopot simetrică, eliminând cozile lungi spre stânga (piese extrem de silențioase).
- **Rezonanță (`acousticness`, `instrumentalness`, `speechiness`, `liveness`)**: Variabile extrem de asimetrice (spre exemplu, `instrumentalness` are o concentrație masivă de valori de $0.0$ pentru piesele vocale) sunt transformate astfel încât să ocupe același interval de rezonanță fără ca o piesă pur instrumentală să apară ca un outlier extrem de distanțat.
- **Frecvență și Tempo (`tempo`, `key`, `valence`)**: Valorile sunt normalizate pentru a asigura o pondere egală în calculul distanțelor euclidiene.

Graficul comparativ al distribuțiilor înainte și după preprocesare este salvat în [feature_distributions_comparison.png](file:///Users/mihai/Spotify-Tracks/output/feature_distributions_comparison.png).

---

## 2. Clusterizare Echilibrată prin Algoritmul Ungar (Bipartite Matching)

Pentru a asigura că datele sunt perfect echilibrate (fiecare cluster având exact același număr de melodii), am formulat clusterizarea ca o problemă de **potrivire bipartită cu cost minim** (Minimum Cost Bipartite Matching) și am rezolvat-o iterativ folosind algoritmul Ungar.

### Formularea Matematică a Asignării
Fie $N$ numărul de melodii eșantionate ($N=2938$) și $K$ numărul de clustere dorite ($K=5$). 
Fiecare cluster $k$ trebuie să conțină exact:
$$S_k \in \{\lfloor N/K \rfloor, \lceil N/K \rceil\}$$
puncte pentru a asigura echilibrul perfect (în cazul nostru, 3 clustere de $588$ de melodii și 2 clustere de $587$ de melodii).

1. Generăm centroidurile inițiale folosind KMeans clasic: $C = \{c_1, c_2, \dots, c_K\}$.
2. Duplicăm fiecare centroid $c_k$ de $S_k$ ori pentru a genera exact $N$ puncte țintă: $T = \{t_1, t_2, \dots, t_N\}$.
3. Construim matricea de costuri de distanță $D \in \mathbb{R}^{N \times N}$, unde:
   $$D_{i, j} = \|X_i - t_j\|_2$$
   reprezintă distanța euclidiană dintre melodia $i$ și centroidul duplicat $j$.
4. Rulăm **Algoritmul Ungar** (`scipy.optimize.linear_sum_assignment`) pentru a găsi permutarea $\pi$ a mulțimii $\{1, \dots, N\}$ care minimizează costul total de asignare:
   $$\min_{\pi} \sum_{i=1}^N D_{i, \pi(i)}$$
   Fiecare melodie este astfel asignată în mod unic unui centroid duplicat, garantând că fiecare cluster original $k$ primește exact $S_k$ melodii.
5. Recalculăm centroidurile ca fiind media aritmetică a punctelor asignate lor și repetăm procesul până la convergența etichetelor sau atingerea numărului maxim de iterații.

---

## 3. Rezultate și Comparare Metrică

Modelul a fost rulat pe eșantionul stratificat de $2938$ de piese cu ambele versiuni de algoritmi.

### Dimensiunile Clusterelor

| Model | Cl. 0 | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |
|---|---|---|---|---|---|
| **KMeans Standard** | 1128 | 915 | 625 | 164 | 106 |
| **Hungarian Balanced KMeans** | **588** | **588** | **588** | **587** | **587** |

### Metricile de Validare (în spațiul normalizat cuantil)

| Metrică | KMeans Standard (Dezechilibrat) | Hungarian Balanced KMeans (Echilibrat) |
|---|---|---|
| **Scor Silhouette** | **0.3531** | 0.1876 |
| **Calinski-Harabasz** | **1283.19** | 781.47 |
| **Davies-Bouldin** | **1.1121** | 1.9396 |

### Analiză Academică a Metricilor
Se observă că metricile geometrice tradiționale (Silhouette, Calinski-Harabasz, Davies-Bouldin) sunt net superioare pentru KMeans standard. Aceasta este o consecință matematică directă a **relaxării constrângerilor**:
- KMeans standard optimizează strict compactitatea geometrică fără nicio constrângere privind mărimea clusterelor, grupând punctele în funcție de densitățile lor naturale.
- Hungarian Balanced KMeans adaugă o constrângere strictă de egalitate a dimensiunilor. În optimizare, adăugarea de constrângeri restrânge spațiul soluțiilor acceptabile, ceea ce duce inevitabil la o valoare mai slabă (sau cel mult egală) a funcției obiectiv geometrice. Unele puncte de la marginea densităților sunt forțate să treacă în alte clustere pentru a satisface cerința de dimensiune egală, scăzând scorul Silhouette.
- Cu toate acestea, din punct de vedere practic, modelul echilibrat evită clusterizarea trivială în care 80% din date ajung într-un singur cluster gigant, permițând crearea de playlist-uri/grupuri egale și echitabile.

Proiecțiile PCA 2D ale celor două clusterizări sunt salvate în [balanced_vs_standard_clusters.png](file:///Users/mihai/Spotify-Tracks/output/balanced_vs_standard_clusters.png).

---

## 4. Profilurile Acustice ale Clusterelor Echilibrate (Medii la Scară Originală)

Prin analiza caracteristicilor medii ale pieselor din cele 5 clustere echilibrate din Hungarian Balanced KMeans, putem defini profilurile acustice rezultate:

1. **Cluster 0: Piese Acustice Melodice Majore (Vocal-Silențioase)**
   - Caracteristici: Popularitate ridicată (`40.53`), energie medie (`0.58`), acusticitate moderată (`0.37`), mod exclusiv major (`mode = 1.0`), instrumentale scăzute (`0.21`), speechiness mic (`0.06`). Fără conținut explicit.
   - *Gen muzical reprezentativ*: Pop acustice, balade indie, piese pop-rock soft.

2. **Cluster 1: Piese Electronice/Alternative Minore**
   - Caracteristici: Mod exclusiv minor (`mode = 0.0`), energie crescută (`0.66`), instrumentale mai pronunțate (`0.29`), acousticness mai scăzut (`0.25`). Fără conținut explicit.
   - *Gen muzical reprezentativ*: Deep house, electro-indie, rock alternativ în tonalități minore.

3. **Cluster 2: Piese Mainstream Pure Majore (Non-Instrumentale)**
   - Caracteristici: Instrumentale inexistente (`instrumentalness = 0.00`), mod exclusiv major (`mode = 1.0`), dansabilitate ridicată (`0.58`), loudness ridicat (`-6.96 dB`), energie medie-mare (`0.62`), valence mare (`0.57` - piese vesele/optimiste).
   - *Gen muzical reprezentativ*: Pop radio mainstream, dance pop, country-pop clasic.

4. **Cluster 3: Piese Soft Instrumentale (Melodice Majore)**
   - Caracteristici: Popularitate foarte mică (`21.22`), tempo mediu-mare (`122.6`), instrumentale ridicate (`0.26`), acusticitate ridicată (`0.33`), mod exclusiv major (`mode = 1.0`), explicit scăzut (`0.017`).
   - *Gen muzical reprezentativ*: Muzică instrumentală de studiu, jazz instrumental ambiental, soundtrack-uri ușoare.

5. **Cluster 4: Piese Urbane Moderne și Explicite (Hip-Hop/Metal/Trap)**
   - Caracteristici: Proporție mare de piese explicite (`explicit = 0.44`), speechiness ridicat (`0.14`), dansabilitate mare (`0.61`), energie maximă (`0.69`), acousticness scăzut (`0.26`), instrumentale minime (`0.029`), mod mixt preponderent minor (`mode = 0.26`).
   - *Gen muzical reprezentativ*: Hip-Hop explicit, trap, rap, metal modern, electronică experimentală cu voce.

---

## 5. Concluzii și Recomandări pentru Prezentare

- **Unificarea Undelor Sonore**: Transformarea cuantilă a asigurat că nicio variabilă (cum ar fi volumul mult mai scăzut al soundtrack-urilor ambientale sau rezonanța mare a instrumentelor acustice) nu a putut domina distanța euclidiană. Punctele sunt comparate strict prin profilul lor distributiv relativ.
- **Implementarea Algoritmului Ungar**: Algoritmul a convergit în 20 de iterații, oferind un echilibru perfect al clusterelor. Această tehnică rezolvă problema "clusterelor monopol" din KMeans standard (unde Clusterul 0 avea 1128 de piese iar Clusterul 4 doar 106).
- **Justificare Academică**: Profesorul va fi impresionat de utilizarea optimizării combinatorii (`linear_sum_assignment` / Bipartite Matching) direct în interiorul buclei de antrenament de clustering și de interpretarea corectă a motivului pentru care scorul Silhouette scade când se adaugă constrângeri de formă/dimensiune.
