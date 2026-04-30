# Compte rendu — Étape 2 : Analyse des sentiments des conversations

**Projet** : Analyse de conversations patient-thérapeute sur la santé mentale   
**Date** : 30 avril 2026  
**Auteur du CR** : LARROUX Logan  
**Encadrante** : Lynda Tamine-Lechani

---

## 1. Contexte et organisation

L'étape 2 a divisé le groupe 1 en deux sous-groupes travaillant en parallèle sur le même jeu de données d'entraînement (`combined_data.csv`, *Sentiment Analysis for Mental Health* sur Kaggle), avec pour objectif commun d'annoter automatiquement le dataset principal `train.csv` en vue de sa transmission au Groupe 2.

| Sous-groupe | Méthode | Membres |
|---|---|---|
| Supervisé | Naive Bayes (BoW + TF-IDF) + Logistic Regression (TF-IDF) | Diogo, Ulysse, Logan |
| Non-supervisé | K-Means (Word2Vec + TF-IDF) | Emin, Victor |

Le pipeline commun aux deux sous-groupes comprend : nettoyage du texte, tokenisation (NLTK + Porter Stemmer), et vectorisation (BoW, TF-IDF, Word2Vec). Le split des données est identique pour les deux : **60% train / 20% validation / 20% test**, avec stratification sur les classes.

### 1.1. La différence entre supervisée et non-supervisée ?

**Une méthode supervisée** apprend à partir d'**exemples déjà étiquetés**. Cela veut dire que pour chaque entrée, on connaît la bonne réponse pendant l'entraînement. Le modèle apprend donc à faire une prédiction à partir de ces couples entrée-sortie.  
*Exemple*: classer des e-mails en spam / non-spam, prédire un prix, détecter une maladie à partir de symptômes etc.

**Une méthode non-supervisée**, quand à elle, apprend à partir de **données non étiquetées**. Ici, il n'y a pas de bonne réponse fournie au départ. Le but est plutôt de découvrir des structures cachées, des groupes ou des patterns dans les données.  
*Exemple*: regrouper des clients en segments, détecter des anomalies, réduire la dimension des données etc.

En résumé:
- Supervisée -> on connaît la réponse attendue.
- Non-supervisée -> on ne connaît pas la réponse, on cherche des structures.

### 1.2. Analyse et comparaison des résultats

Puisque le but de cette partie est de créer un modèle performant permettant de produire des étiquettes sur le dataset principal `train.csv`, nous allons devoir le définir en comparant les résultats des mesures suivante:
#### 1.2.1. Accuracy 
Définit un pourcentage de sentiments correctement prédits  
$$
    Accuracy = \frac{\text{Nb. de bonne prédictions}}{\text{Nb. total d'exemples}}
$$  
*Par exemple* on a 100 images:
- 87 bien classées
- 13 mal classées  

On a $\text{Accuracy} = \frac{87}{100} = 0.87$ donc **87%**.

#### 1.2.2. MSE (Mean Squared Error)
Définit l'erreur moyenne au carré entre la vraie valeur et celle prédite.  
$$
    MSE = \frac{1}{n}\sum{(y_i-\hat{y_i})^2}
$$  
Par exemple, on a les valeurs
- vraies: [10,20,30]
- prédites: [12,18,35]  

Ce qui correspond aux erreurs -2,+2,-5, mis au carré on a 4,4,25.  
Ce qui nous donnes la moyenne $(4+4+25)/3 = 11$
> Le carré permet de pénaliser fortement les grosses erreurs  

> Plus le MSE est proche de 0 mieux c'est
#### 1.2.3. R² (Coefficient de détermination)
Mesure combien le modèle explique la variance des données.  
$$
    R^2 = 1 - \frac{\text{Erreur modèle}}{\text{Erreur moyenne}}
$$  
Par exemple, si notre modèle suit bien les vrais sentiments d'un patient: $R^2 = 0.89$.
Cela signifie que le modèle explique **89% de la variabilité**.

#### 1.2.4. Classification Report
C'est un très grand résumé contenant d'autre variable à comparer.
- Precision — celui-ci pose la question *"parmi ce que j'ai prédit positif, **combien sont vraiment positifs ?**"*  
$$
    Precision = \frac{TP}{TP+FP}
$$  
- Recall — celui-ci pose la question "*parmi les vrais positifs, **combien j'en ai retrouvés ?***"  
$$  
    Recall = \frac{TP}{TP+FN}
$$
- F1-score — celui-ci est un compromis entre precision et recall:  
$$
    F1 = 2 \times\frac{Precision\times Recall}{Precision + Recall}
$$  

> TP = vrai positif  
  TN = vrai négatif  
  FP = faux positif  
  FN = faux négatif  

#### 1.2.5. Matrice de confusion
La matrice de confusion elle permet de visualiser les vrais classes des classes prédites.

Par exemple voici une matrice avec 3 classes avec 120 chats/chiens/oiseaux

| | | | |
|--|--|--|--|
| Chat   | 100  |  20   |   0    |
| Chien  |  20  |  100  |   0    |
| Oiseau |   0  |   0   |   120  |
|        | Chat | Chien | Oiseau |

Les lignes correspondent aux vrais classes et les colonnes aux classes prédites, ici on remarque que le modèle a prédit correctement 100 chats mais a prédit 20 chats comme étant des chiens. De plus, il aurait prédit correctement 100 chiens mais mal les 20 autre en supposant que c'étaient des chats.

#### 1.2.6. Purity Globale

La **Purity** est une mesure utilisée en **clustering** (méthodes non-supervisées).  
Elle permet de savoir si les groupes créés par l’algorithme contiennent majoritairement une seule vraie classe.

Autrement dit :

- un cluster est **pur** si presque tous ses éléments appartiennent à la même classe réelle ;
- plus la purity est élevée, plus les clusters sont cohérents.

La formule est :

$$
Purity = \frac{1}{N}\sum_k \max_j |C_k \cap L_j|
$$

Avec :

- $N$ = nombre total d’exemples,
- $C_k$ = cluster $k$,
- $L_j$ = classe réelle $j$.

Pour chaque cluster, on garde seulement la classe majoritaire.

*Exemple* :

- Cluster A : 20 chats, 3 chiens
- Cluster B : 18 chiens, 2 chats

On garde :

- 20 pour le cluster A
- 18 pour le cluster B

Donc :

$$
Purity = \frac{20+18}{43}=0.88
$$

Soit **88%**.

> Plus la Purity est proche de 1, meilleur est le regroupement.

> Attention : si chaque point forme son propre cluster, la purity vaut 1. Cette mesure peut donc être trompeuse seule.

---

#### 1.2.7. ARI (Adjusted Rand Index)

Le **ARI** mesure la similarité entre :

- les vrais labels
- les clusters trouvés

Il compare les paires de points :

- points qui devraient être ensemble
- points qui devraient être séparés

Puis il corrige le score pour tenir compte du hasard.

Sa valeur est comprise entre :

- **1** : clustering parfait
- **0** : résultat proche du hasard
- **< 0** : pire qu’un regroupement aléatoire

Contrairement à la Purity, l’ARI pénalise :

- un trop grand nombre de clusters,
- les mauvais regroupements,
- les résultats artificiellement bons.

> Plus l’ARI est proche de 1, meilleur est le clustering.

---

### 1.2.8. Résumé des mesures à comparer

| Mesure | Supervisée | Non-supervisée | Interprétation |
|--------|------------|----------------|----------------|
| Accuracy | Oui | Oui (après association clusters/classes) | Pourcentage de bonnes prédictions |
| MSE | Oui | Non | Erreur moyenne au carré |
| R² | Oui | Non | Variance expliquée |
| Precision | Oui | Non | Fiabilité des prédictions positives |
| Recall | Oui | Non | Capacité à retrouver les vrais positifs |
| F1-score | Oui | Non | Compromis Precision / Recall |
| Matrice de confusion | Oui | Oui | Détail des erreurs |
| Purity | Non | Oui | Pureté des clusters |
| ARI | Non | Oui | Similarité réelle clustering / labels |

---

### 1.2.9. Choix des mesures dans notre cas

Dans notre projet, nous cherchons à générer les meilleures étiquettes possibles sur `train.csv`.

Nous comparerons donc :

#### Pour les méthodes supervisées :

- Accuracy
- F1-score
- Classification report
- Matrice de confusion

#### Pour les méthodes non-supervisées :

- Accuracy (si correspondance possible entre clusters et labels)
- Matrice de confusion
- Purity
- ARI

Ainsi, nous pourrons comparer :

- la qualité brute de prédiction,
- la cohérence des regroupements,
- la robustesse globale des modèles.

---

## 2. Résultats par membre

### 2.1 Logan — Méthode supervisée (notebook de référence commun + Logistic Regression)

Avec la grand aide de Diogo, Logan a produit le notebook de base structurant le travail commun. Sa contribution principale sur la partie individuelle est l'implémentation d'une **Régression Logistique** (LR + TF-IDF optimisé) en plus du Naive Bayes, identifiée comme « version extra-performante ».

**Paramétrage TF-IDF amélioré** :
- `ngram_range=(1,3)`, `max_features=50 000`, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`

**Tuning de l'hyperparamètre `alpha` (Naive Bayes)** :
- Valeurs testées : `[0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1]`
- Meilleur alpha BoW : **0.05**
- Meilleur alpha TF-IDF : **0.05**

**Résultats NB sur l'ensemble test** :

| Modèle | Accuracy test |
|---|---|
| Naive Bayes + BoW | ~0.65 (estimé) |
| Naive Bayes + TF-IDF | **0.6875** |

**Analyse par classe (NB + TF-IDF)** :

| Classe | Recall NB TF-IDF | Recall LR TF-IDF | Évolution |
|---|---|---|---|
| Anxiety | 70% | 82% | +12% |
| Bipolar | 57% | 76% | +19% |
| Depression | 77% | 67% | **−10%** |
| Normal | 79% | 92% | +13% |
| Personality disorder | 21% | 62% | **+41%** |
| Stress | 25% | 57% | **+32%** |
| Suicidal | 60% | 65% | +5% |

**Observations clés** :
- TF-IDF pénalise les mots fréquents communs à plusieurs classes (ex. *Stress*, *Personality disorder*), ce qui cause des confusions vers *Depression*.
- La LR avec `class_weight='balanced'` améliore très significativement les classes minoritaires mais au détriment du recall sur *Depression*.
- Le cas *Suicidal* reste problématique : 529 vrais *Suicidal* sur 2 129 sont prédits *Depression* — une confusion à risque dans le contexte médical.
- Le R² est jugé non pertinent pour ce problème de classification multi-classes (labels non ordonnés).

**Modèle déployé sur `train.csv`** : `nb_model_bow` (BoW) pour produire `train_with_sentiments.csv`.

---

### 2.2 Diogo — Méthode supervisée (Naive Bayes BoW + TF-IDF)

Diogo a produit les résultats de référence pour la méthode supervisée, repris dans la comparaison finale du notebook K-Means.

**Résultats confirmés** :

| Modèle | Accuracy test |
|---|---|
| Naive Bayes + TF-IDF | **0.6875** |

**Classification report (NB + TF-IDF)** résumé :

| Classe | F1-score |
|---|---|
| Normal | 0.83 |
| Anxiety | 0.73 |
| Bipolar | 0.68 |
| Depression | ~0.62 |
| Suicidal | ~0.62 |
| Personality disorder | Précision 0.97 / Recall 0.29 |
| Stress | Précision 0.83 / Recall 0.27 |

**Observations** : Les performances reproduisent le profil connu du NB sur données déséquilibrées — très bonne précision sur les classes rares mais recall faible. *Personality disorder* et *Stress* sont systématiquement sous-détectés.

---

### 2.3 Ulysse — Méthode supervisée (Naive Bayes BoW + TF-IDF)

Ulysse a conduit une comparaison directe BoW vs. TF-IDF sur les métriques de précision et de recall par classe.

**Résultats obtenus** :

| Classe | Recall BoW | Précision TF-IDF | Meilleur sur recall |
|---|---|---|---|
| Anxiety | — | — | BoW |
| Bipolar | — | — | BoW |
| Depression | 0.57 | 0.54 | BoW |
| Normal | 0.92 | 0.90 | BoW |
| Personality disorder | 0.50 | 0.97 | BoW (recall) |
| Stress | 0.52 | 0.83 | BoW (recall) |
| Suicidal | 0.60 | 0.65 | TF-IDF |

**Décision d'Ulysse** : Retenir le modèle **BoW** pour le déploiement, au motif que l'objectif est d'annoter le maximum de données → **le recall prime sur la précision**.

**Fichier produit** : `train_with_sentiments.csv` (colonne `predicted_sentiment`, via `bow_vectorizer`).

---

### 2.4 Emin & Victor — Méthode non-supervisée (K-Means)

Le sous-groupe non-supervisé a comparé deux approches de vectorisation pour K-Means : **TF-IDF + SVD** et **Word2Vec**.

#### Pipeline commun

- TF-IDF : `TruncatedSVD(n_components=100)` → normalisation L2
- Word2Vec : `vector_size=100`, `window=5`, `min_count=2` → normalisation L2 directe (pas de SVD nécessaire)
- Sélection de `k` par maximisation du **score silhouette cosinus** sur la validation, avec plancher à `y_train.nunique() = 7`

#### Résultats comparatifs K-Means

| Critère | K-Means + TF-IDF | K-Means + Word2Vec |
|---|---|---|
| Meilleur k retenu | 7 | 7 |
| **Accuracy sur test** | **0.4636** | 0.20 |
| Purity globale | **0.54** | 0.51 |
| **ARI** | **0.25** | 0.13 |
| Besoin de SVD ? | Oui | Non |
| Pureté max cluster | 0.85 (Depression) | 0.90 (Normal) |
| Pureté min cluster | 0.30 (Personality disorder) | 0.30 (Bipolar) |

#### Analyse des résultats

**Pourquoi TF-IDF surpasse Word2Vec ?**

Contra-intuitivement, TF-IDF (accuracy 46%) fait plus du double de Word2Vec (20%). L'hypothèse avancée par le sous-groupe : les catégories cliniques du dataset (*Depression*, *Anxiety*, etc.) se distinguent avant tout par un **vocabulaire symptomatique spécifique**, ce que TF-IDF met précisément en avant. Word2Vec, qui regroupe les mots par contexte sémantique large, tend à fusionner *Depression* et *Suicidal* dans les mêmes clusters (vocabulaire négatif proche), puis le mapping glouton force ces gros clusters sur des labels minoritaires, effondrant l'accuracy globale.

**Classes bien gérées / mal gérées** :

| Classe | Pureté TF-IDF | Pureté Word2Vec |
|---|---|---|
| Depression | 0.85 ✅ | 0.87 ✅ |
| Normal | 0.61 | 0.90 ✅ |
| Anxiety | 0.56 | 0.47 |
| Suicidal | 0.48 | 0.56 |
| Stress | 0.34 | 0.46 |
| Bipolar | 0.35 | 0.30 |
| Personality disorder | 0.30 | 0.37 |

**Décision du sous-groupe** : retenir **K-Means + TF-IDF** pour l'annotation.  
**Fichier produit** : `train_with_sentiments_kmeans.csv`.

---

## 3. Comparaison globale supervisé vs. non-supervisé

| Critère | Naive Bayes TF-IDF (supervisé) | K-Means + TF-IDF (non-supervisé) |
|---|---|---|
| **Accuracy sur test** | **0.6875** | 0.4636 |
| Macro avg F1 | 0.62 | — |
| Weighted avg F1 | 0.69 | — |
| Voit les labels à l'entraînement ? | Oui | Non |
| Hyperparamètre clé | alpha = 0.05 | k = 7 |
| Fichier annoté | `train_with_sentiments.csv` | `train_with_sentiments_kmeans.csv` |

**Écart** : +22 points d'accuracy en faveur du supervisé. Cet écart est cohérent et attendu : le supervisé a accès aux vrais labels pendant l'entraînement, le non-supervisé pas. Les deux méthodes utilisent la même représentation TF-IDF, donc l'écart reflète purement la différence supervisé/non-supervisé.

**Problèmes communs aux deux approches** : les mêmes classes sont difficiles pour les deux méthodes (*Personality disorder*, *Stress*, confusion *Depression*/*Suicidal*). Cela pointe vers une limite des données elles-mêmes (classes sous-représentées et vocabulaire qui se chevauche), plutôt qu'un problème algorithmique.

---

## 4. Décision finale : modèle transmis au Groupe 2

**→ Modèle retenu : Naive Bayes + TF-IDF (notebook de Diogo/Logan)**

| Critère | Justification |
|---|---|
| Accuracy 68.75% | Meilleure performance absolue parmi tous les modèles testés |
| Annotations fiables | Les labels transmis au Groupe 2 doivent être les plus corrects possible pour que le filtrage par sentiments soit utile |
| Modèle léger | NB est rapide à déployer et à sérialiser (`.pkl`) |

**Fichier à transmettre au Groupe 2** : `train_with_sentiments.csv`  
**Colonne d'annotation** : `predicted_sentiment`  
**Classes prédites** : `Anxiety`, `Bipolar`, `Depression`, `Normal`, `Personality disorder`, `Stress`, `Suicidal`

> **Note pour le Groupe 2** : la précision sur *Personality disorder* et *Stress* reste faible (recall ~25–30%). Ces classes sont sous-annotées dans le dataset source. Il est conseillé de traiter les réponses filtrées sur ces deux classes avec prudence, ou d'élargir le seuil de similarité pour compenser.

---

## 5. Points de vigilance et perspectives

- **Déséquilibre des classes** : *Depression* est sur-représentée dans `combined_data.csv`, ce qui biaise tous les modèles. Une solution (non implémentée faute de temps) serait le sur-échantillonnage SMOTE ou l'ajustement de `class_weight`.
- **Confusion Depression / Suicidal** : problème critique dans le contexte médical. La LR avec régularisation L1 (testée par Logan) améliore ce point mais le temps d'entraînement est prohibitif (~4h). À envisager si plus de ressources de calcul sont disponibles.
- **R² non pertinent** : cette métrique ne doit pas être utilisée pour évaluer des classifieurs multi-classes à labels nominaux. L'accuracy, le F1-score et la matrice de confusion sont les indicateurs à retenir.
- **Amélioration possible** : la Logistic Regression (LR + TF-IDF, `class_weight='balanced'`) de Logan est prometteuse (+12 à +41% de recall sur les classes minoritaires) et devrait être envisagée comme version finale si le temps le permet.