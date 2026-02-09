# 📊 Data Mining Project - UCI Adult Dataset

Un proiect complet de **Data Mining** care analizează dataset-ul Adult (Census Income) de la UCI Machine Learning Repository. Acest README explică fiecare concept și metodă folosită, presupunând că cititorul nu are experiență anterioară în data mining.

---

## 📖 Ce este Data Mining?

**Data Mining** (sau "mineritul datelor") este procesul de descoperire a patternurilor, corelațiilor și informațiilor utile din seturi mari de date. Gândește-te la el ca la un detectiv digital care caută indicii ascunse în munți de date.

### De ce este important?
- **Business**: Predicția comportamentului clienților
- **Medicină**: Detectarea bolilor din simptome
- **Finanțe**: Detectarea fraudelor
- **Marketing**: Segmentarea clienților

---

## 📁 Dataset-ul Adult (Census Income)

### Ce conține?
Dataset-ul conține informații din recensământul SUA din 1994, cu scopul de a prezice dacă o persoană câștigă **mai mult de $50,000/an** sau nu.

| Atribut | Tip | Descriere |
|---------|-----|-----------|
| `age` | Numeric | Vârsta persoanei |
| `workclass` | Categoric | Tipul angajatorului (Private, Gov, Self-employed, etc.) |
| `fnlwgt` | Numeric | Pondere finală (factor de reprezentativitate) |
| `education` | Categoric | Nivel de educație (Bachelors, Masters, etc.) |
| `education-num` | Numeric | Număr de ani de educație |
| `marital-status` | Categoric | Stare civilă |
| `occupation` | Categoric | Ocupație |
| `relationship` | Categoric | Relație în familie |
| `race` | Categoric | Rasă |
| `sex` | Categoric | Sex |
| `capital-gain` | Numeric | Câștiguri din capital |
| `capital-loss` | Numeric | Pierderi din capital |
| `hours-per-week` | Numeric | Ore lucrate pe săptămână |
| `native-country` | Categoric | Țara de origine |
| `income` | **Target** | `>50K` sau `<=50K` (ce vrem să prezicem) |

### Statistici
- **Total instanțe**: 48,842
- **Training**: 32,561 instanțe
- **Test**: 16,281 instanțe
- **Missing values**: Marcate cu `?`

---

## 🏗️ Structura Proiectului

```
DataMining/
├── data/
│   ├── adult.data          # Date de antrenament
│   └── adult.test          # Date de test
├── src/
│   ├── __init__.py         # Inițializare pachet Python
│   ├── preprocessing.py    # Pregătirea datelor
│   ├── classification.py   # Task 1: Clasificare
│   ├── outlier_detection.py# Task 2: Detectare outlieri
│   ├── clustering.py       # Task 3: Clustering
│   ├── association_rules.py# Task 4: Reguli de asociere
│   ├── feature_selection.py# Task 5: Selecția features
│   └── utils.py            # Funcții utilitare
├── results/
│   ├── models/             # Modele antrenate salvate
│   ├── plots/              # Grafice generate
│   └── reports/            # Rapoarte CSV
├── main.py                 # Script principal
├── requirements.txt        # Dependențe Python
└── README.md               # Acest fișier
```

---

## 🔧 Preprocesarea Datelor (`preprocessing.py`)

Înainte de a aplica orice algoritm, datele trebuie **pregătite**. Datele "brute" rareori sunt gata de utilizare.

### 1. Tratarea Valorilor Lipsă
```
Problemă: Unele câmpuri au valoarea "?" (nu se știe)
Soluție: Înlocuim cu cea mai frecventă valoare (modul)
```

**De ce?** Algoritmii nu pot procesa valori lipsă. Modalul (valoarea cea mai frecventă) este o aproximare sigură.

### 2. Encoding-ul Variabilelor Categorice

**Problemă**: Algoritmii înțeleg doar numere, nu text ca "Private" sau "Bachelors".

**Soluție**: **One-Hot Encoding**
```
occupation=Tech-support  →  [1, 0, 0, 0, ...]
occupation=Sales         →  [0, 1, 0, 0, ...]
occupation=Exec-manager  →  [0, 0, 1, 0, ...]
```

Fiecare categorie devine o coloană separată cu valori 0 sau 1.

### 3. Scalarea Variabilelor Numerice

**Problemă**: Variabilele au scale diferite (age: 17-90, capital-gain: 0-99,999).

**Soluție**: **StandardScaler** - transformă fiecare variabilă să aibă:
- Media = 0
- Deviația standard = 1

**De ce?** Algoritmii sunt sensibili la scale. Fără scalare, `capital-gain` ar domina toate deciziile doar pentru că are valori mai mari.

### 4. Feature Engineering

Creăm **variabile noi** din cele existente pentru a îmbunătăți predicțiile:

| Feature Nou | Formula | Intuiție |
|-------------|---------|----------|
| `gain_loss_ratio` | capital-gain / (capital-loss + 1) | Raportul câștig/pierdere |
| `is_high_hours` | 1 dacă hours > 40, altfel 0 | Lucrează overtime? |
| `education_efficiency` | education-num / age | Cât de repede a avansat în educație |

---

## 📋 Task 1: Clasificare (`classification.py`)

### Ce este Clasificarea?
**Clasificarea** este procesul de a prezice o **categorie** (clasă) pentru date noi. În cazul nostru: va câștiga persoana >$50K sau nu?

### Algoritmii Folosiți

#### 1. Decision Tree (Arbore de Decizie)
```
                    [Age > 30?]
                   /           \
                 Yes            No
                /                 \
        [Education > 12?]      [Income: <=50K]
           /        \
         Yes         No
          |           |
    [Income: >50K]  [Hours > 40?]
                      /     \
                    Yes      No
                     |        |
              [>50K]      [<=50K]
```

**Cum funcționează?** Ia decizii secvențiale bazate pe întrebări simple. E ca un joc de "20 de întrebări".

**Avantaje**: Ușor de interpretat, rapid
**Dezavantaje**: Poate "memora" datele (overfitting)

#### 2. Random Forest (Pădure Aleatoare)
Creează **multe arbori de decizie** (ex: 100) și îi lasă să voteze.

```
Arbore 1: >50K     ┐
Arbore 2: <=50K    │
Arbore 3: >50K     ├──→ Majoritatea: >50K  ✓
Arbore 4: >50K     │
Arbore 5: <=50K    ┘
```

**De ce mai mulți arbori?** Un singur arbore poate greși. 100 de arbori care votează sunt mai stabili - "înțelepciunea mulțimii".

**Avantaje**: Foarte precis, rezistent la overfitting
**Dezavantaje**: Mai lent, mai greu de interpretat

#### 3. Logistic Regression (Regresie Logistică)
Nu confunda cu regresia obișnuită! Calculează **probabilitatea** de a aparține unei clase.

```
P(income = >50K) = 1 / (1 + e^(-(w₁·age + w₂·education + ... + b)))
```

Rezultatul e între 0 și 1. Dacă > 0.5, prezice >50K.

**Avantaje**: Rapid, oferă probabilități, interpretabil
**Dezavantaje**: Presupune relații liniare

#### 4. XGBoost (Extreme Gradient Boosting)
Algoritmul "superstar" al competițiilor de ML. Construiește arbori **secvențial**, fiecare corectând greșelile precedentului.

```
Arbore 1: Predicție inițială (slabă)
         ↓
Arbore 2: Corectează erorile arborelui 1
         ↓
Arbore 3: Corectează erorile rămase
         ↓
...
Final: Sumă ponderată a tuturor arborilor
```

**Avantaje**: Cel mai precis de obicei
**Dezavantaje**: Mai multe hiperparametri de reglat

### GridSearchCV - Găsirea Celor Mai Buni Hiperparametri

**Hiperparametri** = setări ale algoritmului (ex: câți arbori, cât de adânci).

**GridSearchCV** testează **toate combinațiile** și alege cea mai bună:
```
max_depth: [5, 10, 15]
n_estimators: [50, 100]
                ↓
Testează: (5,50), (5,100), (10,50), (10,100), (15,50), (15,100)
                ↓
Alege combinația cu cel mai bun scor
```

### Metrici de Evaluare

| Metrică | Ce Măsoară | Formulă |
|---------|------------|---------|
| **Accuracy** | % predicții corecte | (TP+TN) / Total |
| **Precision** | Din cei preziceți >50K, câți chiar sunt | TP / (TP+FP) |
| **Recall** | Din cei care chiar sunt >50K, câți am găsit | TP / (TP+FN) |
| **F1** | Media armonică Precision-Recall | 2·P·R / (P+R) |
| **ROC-AUC** | Capacitatea de a separa clasele | Aria sub curba ROC |

Unde: TP=True Positive, TN=True Negative, FP=False Positive, FN=False Negative

---

## 🔍 Task 2: Detectarea Outlierilor (`outlier_detection.py`)

### Ce sunt Outlierii?
**Outlierii** (anomalii) sunt puncte de date care sunt **semnificativ diferite** de restul.

```
Date normale:    ● ● ● ● ● ● ● ●
Outlier:                           ★ (departe de grup)
```

### De ce îi căutăm?
- Pot indica **fraude** (tranzacții suspecte)
- Pot fi **erori** de introducere date
- Pot fi **cazuri rare** interesante (tineri foarte bogați)

### Metoda 1: Isolation Forest
**Ideea**: Outlierii sunt mai ușor de "izolat" (separat) decât punctele normale.

```
Pas 1: Alege o variabilă random (ex: age)
Pas 2: Alege un prag random (ex: 35)
Pas 3: Împarte datele: age < 35 | age >= 35
Pas 4: Repetă până izolezi fiecare punct

Outlierii: Necesită PUȚINE împărțiri pentru izolare
Normalii: Necesită MULTE împărțiri
```

### Metoda 2: Local Outlier Factor (LOF)
**Ideea**: Compară **densitatea locală** a unui punct cu vecinii săi.

```
Punct normal: Densitate similară cu vecinii
Outlier: Densitate mult mai mică decât vecinii (izolat)
```

LOF = Densitatea vecinilor / Densitatea punctului
- LOF ≈ 1: Normal
- LOF >> 1: Outlier (vecinii sunt mai denși)

### Ce analizăm?
Filtrăm doar persoanele cu **income >50K** și căutăm anomalii bazate pe:
- `age` (vârstă)
- `hours-per-week` (ore lucrate)
- `capital-gain` (câștiguri din investiții)

**Exemplu de outlier găsit**: Persoană de 22 ani, care lucrează 99 ore/săptămână și are capital-gain de $99,999. Extrem de neobișnuit!

---

## 🎯 Task 3: Clustering (`clustering.py`)

### Ce este Clustering-ul?
**Clustering** = gruparea datelor în categorii **fără a ști dinainte** care sunt acele categorii. Algoritmul le descoperă singur.

```
Input: ● ● ● ○ ○ ○ ○ ■ ■ ■ ■ (puncte fără etichete)
Output: Grup1(●) Grup2(○) Grup3(■)
```

### Diferența față de Clasificare
- **Clasificare**: Știm categoriile (>50K sau <=50K), învățăm să le recunoaștem
- **Clustering**: NU știm categoriile, le descoperim

### Metoda 1: K-Means

**Algoritmul**:
```
1. Alege K centre aleatorii
2. Atribuie fiecare punct la centrul cel mai apropiat
3. Recalculează centrele (media punctelor din fiecare cluster)
4. Repetă pașii 2-3 până nu se mai schimbă nimic
```

**Problema**: Trebuie să alegem K (numărul de clustere).

**Soluția 1: Elbow Method**
```
Inertia
   │
   │╲
   │ ╲
   │  ╲___________  ← "cotul" - aici K e optim
   │
   └────────────────── K
        2  3  4  5
```
Inertia = suma distanțelor la centru. Căutăm "cotul" unde scăderea încetinește.

**Soluția 2: Silhouette Score**
Măsoară cât de bine separat e fiecare cluster. Scor între -1 și 1:
- 1 = clustere perfect separate
- 0 = clustere suprapuse
- -1 = puncte în clusterul greșit

### Metoda 2: DBSCAN
**Density-Based Spatial Clustering of Applications with Noise**

**Ideea**: Clusterele sunt **zone dense** separate de zone goale.

```
Parametri:
- eps: Raza de căutare
- min_samples: Minimum puncte pentru o zonă densă

Tipuri de puncte:
● Core point: Are ≥ min_samples vecini în raza eps
○ Border point: În raza unui core point, dar nu are destui vecini
✗ Noise: Nici core, nici border (outlier)
```

**Avantaj**: Descoperă clustere de **orice formă** (nu doar sferice ca K-Means).

### PCA - Reducerea Dimensionalității

**Problema**: Avem ~100 de variabile după encoding. Imposibil de vizualizat.

**Soluția**: **Principal Component Analysis (PCA)** - comprimă datele păstrând informația importantă.

```
100 variabile → PCA → 5 componente (păstrează ~60% din informație)
                   → 2 componente (pentru vizualizare)
```

---

## 🔗 Task 4: Reguli de Asociere (`association_rules.py`)

### Ce sunt Regulile de Asociere?
Descoperă **relații de tipul "dacă X, atunci Y"** în date.

**Exemplul clasic**: Analiza coșului de cumpărături
```
Dacă client cumpără pâine ȘI unt → probabil cumpără și lapte
```

### În proiectul nostru
```
Dacă age=36-45 ȘI education=Bachelors ȘI hours=41-50 → income=>50K
```

### Discretizarea
Transformăm valorile numerice în categorii pentru a crea reguli:

| Variabilă | Bins |
|-----------|------|
| age | 0-25, 26-35, 36-45, 46-55, 56-100 |
| hours-per-week | 0-30, 31-40, 41-50, 51-100 |
| capital-gain | 0, 1-5000, 5001+ |

### Metrici pentru Reguli

| Metrică | Formulă | Interpretare |
|---------|---------|--------------|
| **Support** | P(A ∩ B) | Cât de frecvent apare regula |
| **Confidence** | P(B\|A) = Support/P(A) | Dacă A, cât de probabil B? |
| **Lift** | Confidence / P(B) | De câte ori mai probabil B cu A vs. fără A |

**Lift > 1**: Asociere pozitivă (A crește șansa lui B)
**Lift = 1**: Independente
**Lift < 1**: Asociere negativă (A scade șansa lui B)

### Algoritmii

#### Apriori
```
1. Găsește itemuri frecvente (support ≥ min_support)
2. Combină-le în perechi, păstrează pe cele frecvente
3. Combină în triplete, etc.
4. Generează reguli din itemset-urile frecvente
```

#### FP-Growth
Mai eficient ca Apriori - construiește un arbore compact al tranzacțiilor și extrage regulile fără a scana datele de mai multe ori.

---

## 🎚️ Task 5: Selecția Features (`feature_selection.py`)

### De ce selectăm features?
- **Prea multe variabile** = risc de overfitting
- **Curse of dimensionality**: performanța scade cu prea multe dimensiuni
- **Eficiență**: model mai rapid cu mai puține variabile

### Metodele Comparate

#### 1. Chi-Square (χ²)
**Ideea**: Măsoară dependența dintre o variabilă și target.

```
Întrebare: Este distribuția lui X diferită între clase?
           (ex: distribuția "education" e diferită între >50K și <=50K?)

χ² mare → Variabila e relevantă
χ² mic → Variabila e probabil irelevantă
```

#### 2. Mutual Information
**Ideea**: Cât de multă informație despre target obținem din variabilă?

```
I(X; Y) = cât de mult reduce incertitudinea în Y cunoașterea lui X

I(education; income) = mare (știind educația, știm mai bine venitul)
I(fnlwgt; income) = mic (ponderea nu spune nimic despre venit)
```

#### 3. RFE (Recursive Feature Elimination)
**Ideea**: Elimină iterativ cele mai slabe variabile.

```
1. Antrenează model cu TOATE variabilele
2. Identifică variabila cea mai puțin importantă
3. Elimin-o
4. Repetă până rămân K variabile
```

**Avantaj**: Ia în considerare interacțiunile între variabile.
**Dezavantaj**: Lent (antrenează modelul de multe ori).

### Experimentul
Testăm fiecare metodă cu K = 5, 8, 10, 12 variabile și comparăm accuracy-ul.

---

## 🚀 Cum să Rulezi Proiectul

### 1. Instalare Dependențe
```bash
cd /home/bogdan/PersonalProjects/DataMining
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rulare

```bash
# Toate task-urile (durează ~30-60 minute)
python main.py

# Un singur task
python main.py --task 1   # Doar clasificare
python main.py --task 2   # Doar outlieri
python main.py --task 3   # Doar clustering
python main.py --task 4   # Doar reguli asociere
python main.py --task 5   # Doar feature selection

# Mai multe task-uri
python main.py --task 1,3,5
```

### 3. Rezultate

După rulare, găsești:
- **`results/models/`** - Modele antrenate (.joblib)
- **`results/plots/`** - Grafice (.png)
- **`results/reports/`** - Tabele rezultate (.csv)

---

## 📊 Exemplu de Rezultate

### Clasificare (Task 1)
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| RandomForest | 86.2% | 0.71 |
| XGBoost | 87.1% | 0.73 |
| LogisticRegression | 85.0% | 0.68 |
| DecisionTree | 81.5% | 0.62 |

### Outlieri (Task 2)
- **Detectați**: ~10% din instanțele high-income sunt outlieri
- **Cel mai extrem**: 22 ani, 55 ore/săpt, $99,999 capital-gain

### Clustering (Task 3)
- **K optim**: 3-4 clustere (după silhouette score)
- **Cluster 1**: Tineri, educație medie, ore normale
- **Cluster 2**: Maturi, educație înaltă, ore multe

---

## 📚 Resurse pentru Învățare

1. **Scikit-learn Documentation**: https://scikit-learn.org/stable/
2. **Coursera - Machine Learning** (Andrew Ng)
3. **Kaggle Learn**: https://www.kaggle.com/learn
4. **UCI ML Repository**: https://archive.ics.uci.edu/

---

## 🤝 Autor

Proiect creat pentru cursul de **Data Mining**.

---

*"In God we trust. All others must bring data."* — W. Edwards Deming
