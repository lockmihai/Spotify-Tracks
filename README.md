# Studiu Comparativ și Abordări Avansate în Clusterizarea Datelor Audio (Spotify Tracks)

Acest proiect reprezintă o analiză academică și inginerească de anvergură pe setul de date **Spotify Tracks Dataset**. Proiectul compară peste 14 algoritmi de clusterizare tradiționali și adaugă patru contribuții avansate/originale specifice domeniului muzical, culminând cu o aplicație web interactivă publicată în Google Cloud Run.

---

## 🚀 Live App (Google Cloud Run)
Aplicația interactivă este disponibilă public la adresa:  
👉 **[https://spotify-tracks-clustering-qq5xqrxp5q-uc.a.run.app](https://spotify-tracks-clustering-qq5xqrxp5q-uc.a.run.app)**

*Profesorul poate testa live toți algoritmii, vizualiza grafice interactive 2D (Plotly) cu melodiile la trecerea mouse-ului și rula un sandbox de fine-tuning paralel pe procesoarele din Cloud.*

---

## 📊 1. Setul de Date și Preprocesarea

Setul de date original conține **114.000 de melodii** cu 125 de genuri și 15 caracteristici numerice (inclusiv popularitate, dansabilitate, energie, volum, acusticitate, tempo etc.).

### Preprocesare și Eșantionare Stratificată
* **Eșantionare**: Din motive de complexitate computațională (ex. link-urile ierarhice și algoritmul Ungar au o complexitate temporală/spațială de $O(N^3)$), am extras un **eșantion stratificat reprezentativ de 3.000 de piese** (asigurând o distribuție uniformă a tuturor celor 125 de genuri).
* **Unificarea Undelor Sonore (Quantile Normalization)**: Pe lângă standardizarea clasică, am implementat preprocesarea prin `QuantileTransformer(output_distribution='normal')`. Aceasta forțează distribuțiile asimetrice (precum `instrumentalness` sau `loudness`) într-o curbă Gaussiană perfect simetrică, împiedicând piesele de tip soundtrack ambiental (care au volum sau rezonanță aberantă) să distorsioneze distanțele și centroidurile.

---

## 🛠️ 2. Structura Proiectului (Workspace Tour)

* **`run_clustering.py`**: Conducta principală de evaluare comparativă a celor 14 algoritmi clasici și custom, cu generarea automata de ploturi în `output/` și a raportului `Studiu_Clusterizare.md`.
* **`meta_learner.py`**: Implementarea modelului **Meta Song Learner** (clasificator supervizat Random Forest pentru routarea cântecelor către modele specifice stilului) și a modelului **Ensemble** (consensus partition bazat pe o matrice de co-asociere).
* **`run_balanced_clustering.py`**: Implementarea normalizării cuantile și a clasei custom `HungarianBalancedKMeans`.
* **`app.py`**: Codul aplicației Streamlit publicate pe Google Cloud Run.
* **`Fisa_Prezentare_Proiect.md`**: Ghid rapid de prezentare în fața profesorului, cu clasamentul sintetic al algoritmilor (Scorecard) și întrebări/răspunsuri pregătite pentru apărare.
* **`notebooks/`**: Notebook-uri Jupyter interactive organizate pe pași:
  * `eda_and_preprocessing.ipynb`: Încărcare, curățare, vizualizări EDA și eșantionare.
  * `clustering_analysis.ipynb`: Evaluarea comparativă a celor 15+ partiții.
  * `meta_learner_analysis.ipynb`: Testarea pe celule a Meta Learner-ului și Ensemble-ului.
  * `balanced_clustering_analysis.ipynb`: Interacțiune live cu transformarea cuantilă și algoritmul Ungar.

---

## 🧬 3. Algoritmii Aleși vs. Cei Nealeși (Avantaje și Selecție)

Pentru a acoperi toate categoriile teoretice de clusterizare solicitate, am selectat cele mai adaptate tehnici:

### A. Metode de Partiționare (KMeans vs. BisectingKMeans vs. KMedoids)
* **KMeans**: Ales pentru simplitatea sa și viteza de rulare. Este baseline-ul optim, dar extrem de sensibil la anomalii sonore.
* **BisectingKMeans**: Ales deoarece combină abordarea ierarhică divizivă (top-down) cu viteza KMeans. Oferă clustere mai bine structurate pe genuri mari.
* **KMedoids**: Ales ca alternativă robustă la KMeans; folosește puncte reale din dataset ca puncte centrale (medoizi), fiind mult mai rezistent la zgomot.

### B. Metode Probabilistice (Gaussian Mixture - EM)
* **GMM (Gaussian Mixture Models)**: Ales deoarece permite clustere cu formă eliptică (nu doar sferice ca la KMeans) și oferă o alocare probabilistă (soft assignment), ideală pentru piese care combină genuri (ex. pop-rock).

### C. Metode Ierarhice (Ward vs. Complete vs. Average vs. Single vs. Centroid)
* **Ward Linkage**: Ales ca metodă ierarhică de bază datorită minimizării varianței interne. Reușește să identifice structuri muzicale echilibrate.
* **Complete/Average/Centroid**: Alese pentru a demonstra influența criteriilor diferite de legătură asupra formării dendrogramelor.
* *De ce am evitat algoritmul BIRCH ca algoritm ierarhic principal?* BIRCH este conceput pentru seturi de date uriașe (procesare locală incrementală), dar pe eșantionul nostru de 3000 de piese, el tinde să distorsioneze structura globală comparativ cu o dendrogramă clasică.

### D. Metode Bazate pe Densitate (DBSCAN vs. OPTICS vs. HDBSCAN)
* **DBSCAN**: Ales pentru abilitatea sa de a detecta clustere de forme arbitrare și de a identifica piese zgomotoase (noise) ca fiind outlieri (-1).
* **OPTICS / HDBSCAN**: Alese deoarece rezolvă principala slăbiciune a DBSCAN – sensibilitatea la densități variabile (piesele electronice sunt foarte dense, în timp ce piesele jazz sunt foarte rarefiate în spațiul caracteristicilor).

### E. Metode Bazate pe Grid (Custom STING vs. Custom DENCLUE)
* **Custom STING (Statistical Information Grid)**: Implementat manual. Proiectează spațiul multidimensional în 2D PCA, împarte zona într-un caroiaj statistic și unește celulele active prin 8-conectivitate. Rapid și imun la zgomote punctuale.
* **Custom DENCLUE (Density Clustering)**: Implementat manual. Utilizează aproximarea densității prin Kernel Density Estimation (KDE) cu nuclee Gaussiene și algoritmul de hill-climbing (gradient ascent) pentru a găsi atractorii de densitate locali. Identifică forme geometrice extrem de fine.

### F. Inovație: Hungarian Balanced KMeans (Clusterizare Echilibrată)
* **Hungarian Balanced KMeans**: Creat special pentru a rezolva dezechilibrul clasic din KMeans. Prin duplicarea centroidurilor de $N/K$ ori și rezolvarea **Minimum Cost Bipartite Matching** cu algoritmul Ungar, forțează partitii perfect egale în volum, fiind ideal pentru generarea automată de playlist-uri uniforme ca lungime și distribuție.

---

## 💻 4. Rularea Proiectului Local (Replicare)

Urmați acești pași pentru a instala și rula întregul proiect pe calculatorul dumneavoastră:

### Pasul 1: Prerechizite
Asigurați-vă că aveți instalat **Python 3.11** sau o versiune mai nouă. De asemenea, este recomandat managerul de pachete `pip`.

### Pasul 2: Clonarea și Navigarea în Director
```bash
git clone https://github.com/lockmihai/Spotify-Tracks.git
cd Spotify-Tracks
```

### Pasul 3: Crearea și Activarea Mediului Virtual
* **Pe macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Pe Windows**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Pasul 4: Instalarea Dependențelor
```bash
pip install -r requirements.txt
```

### Pasul 5: Executarea Scripturilor
Puteți rula scripturile în mod individual pentru a genera metricile, rapoartele și ploturile:
1. **Analiza comparativă a celor 14 algoritmi**:
   ```bash
   python run_clustering.py
   ```
   *Generează rezultate în `output/` și raportul [Studiu_Clusterizare.md](file:///Users/mihai/Spotify-Tracks/Studiu_Clusterizare.md).*

2. **Rularea modelului Meta Song Learner și a Ensemble-ului**:
   ```bash
   python meta_learner.py
   ```
   *Generează raportul [Meta_Learner_Studiu.md](file:///Users/mihai/Spotify-Tracks/Meta_Learner_Studiu.md) și ploturi asociate.*

3. **Rularea algoritmului Ungar și a transformării cuantile**:
   ```bash
   python run_balanced_clustering.py
   ```
   *Generează raportul [Hungarian_Balanced_Clustering.md](file:///Users/mihai/Spotify-Tracks/Hungarian_Balanced_Clustering.md) și setul de date echilibrat.*

### Pasul 6: Rularea Interfeței Web Streamlit Local
Puteți rula interfața grafică interactivă local în browser:
```bash
streamlit run app.py
```
*Aplicația se va deschide automat la adresa: `http://localhost:8501`.*
