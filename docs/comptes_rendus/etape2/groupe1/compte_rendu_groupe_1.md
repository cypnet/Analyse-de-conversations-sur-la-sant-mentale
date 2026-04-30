# Compte rendu — Étape 2 : Analyse des sentiments des conversations

**Projet** : Analyse de conversations patient-thérapeute sur la santé mentale  
**Auteur** : LARROUX Logan  
**Date** : avril 2026  
**Dataset** : `combined_data.csv` — 52 632 énoncés, 7 classes (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder)  
**Split** : 60% train (31 579) / 20% validation (10 526) / 20% test (10 527) — stratifié

---

## 1. Résultats par membre

### 1.1 Groupe supervisé — Naive Bayes

#### Logan (notebook principal)

Split 60/20/20 avec tuning de l'hyperparamètre alpha sur la validation.

| Variante | Alpha | Acc. val | Acc. test | MSE test | R² test |
|---|---|---|---|---|---|
| NB + BoW | 0.10 | 0.6606 | 0.6666 | 3.0897 | 0.0296 |
| NB + TF-IDF | 0.05 | 0.6761 | **0.6875** | **2.8781** | **0.0961** |

F1-score par classe (TF-IDF, ensemble test) :

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Normal | 0.90 | 0.77 | 0.83 |
| Anxiety | 0.76 | 0.71 | 0.73 |
| Bipolar | 0.82 | 0.58 | 0.68 |
| Depression | 0.54 | 0.77 | 0.63 |
| Stress | — | — | ~0.41 |
| Personality disorder | — | — | ~0.45 |
| Suicidal | — | — | ~0.57 |

**Observation** : TF-IDF domine BoW sur toutes les métriques. Les classes les plus déséquilibrées (Stress, Personality disorder) obtiennent les F1 les plus bas. Normal et Anxiety sont bien classifiées.

---

#### Ulysse (notebook Ulysse)

Même pipeline que Logan (split 60/20/20, tuning alpha). Résultats quasi-identiques, confirmant la reproductibilité.

| Variante | Alpha | Acc. test | MSE test | R² test |
|---|---|---|---|---|
| NB + BoW | 0.10 | 0.6666 | 3.0897 | 0.0296 |
| NB + TF-IDF | 0.05 | **0.6875** | **2.8781** | **0.0961** |

F1-score par classe (TF-IDF, ensemble test) :

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Normal | 0.90 | 0.77 | 0.83 |
| Anxiety | 0.76 | 0.71 | 0.73 |
| Bipolar | 0.82 | 0.58 | 0.68 |
| Depression | 0.54 | 0.77 | 0.63 |
| Stress | 0.83 | 0.27 | 0.41 |
| Personality disorder | 0.97 | 0.29 | 0.45 |
| Suicidal | — | — | ~0.57 |

**Observation** : Résultats identiques à Logan, ce qui valide la robustesse du pipeline commun. L'alpha optimal retenu est 0.05 pour TF-IDF dans les deux cas.

---

### 1.2 Groupe non-supervisé — K-Means

Les deux notebooks du groupe non-supervisé (Emin — `ETAPE2-kmeans-.ipynb` ; Victor — `ETAPE2-Analyse_des_sentiments_des_conversations_TEMPLATE.ipynb`) ont comparé K-Means avec deux représentations vectorielles : **TF-IDF** et **Word2Vec** (100 dimensions, entraîné sur le corpus, puis normalisé L2).

#### Emin (notebook kmeans)

k optimal retenu : **7** pour les deux variantes (sélection sur validation par silhouette cosinus).

| Variante | k | Acc. test | Purity | ARI |
|---|---|---|---|---|
| K-Means + TF-IDF | 7 | 0.4636 | 0.5414 | **0.2481** |
| K-Means + Word2Vec | 7 | 0.1211 | 0.5300 | 0.1153 |

Mapping cluster → label (TF-IDF, meilleur modèle) :

| Cluster | Label assigné | Pureté |
|---|---|---|
| 1 | Depression | 85% |
| 4 | Normal | 61% |
| 3 | Anxiety | 56% |
| 2 | Suicidal | 48% |
| 5 | Bipolar | 35% |
| 0 | Stress | 34% |
| 6 | Personality disorder | 30% |

**Observation** : TF-IDF est nettement supérieur à Word2Vec sur ce corpus. La pureté élevée de Depression (85%) et Normal (61%) contraste avec les classes rares (Stress, Personality disorder) qui sont mal isolées. L'ARI de 0.25 indique une correspondance modérée avec les vraies étiquettes.

---

#### Victor (notebook template)

k optimal retenu : **7** (Word2Vec uniquement évalué sur test dans ce notebook).

| Variante | k | Acc. test | Purity | ARI |
|---|---|---|---|---|
| K-Means + Word2Vec | 7 | 0.2489 | 0.4853 | 0.1304 |

Mapping cluster → label (Word2Vec) :

| Cluster | Label assigné | Pureté |
|---|---|---|
| 3 | Normal | 89% |
| 0 | Suicidal | 85% |
| 1 | Depression | 44% |
| 5 | Stress | 42% |
| 2 | Bipolar | 41% |
| 6 | Anxiety | 37% |
| 4 | Personality disorder | 28% |

**Observation** : Les résultats Word2Vec sont cohérents entre les deux membres du groupe non-supervisé. Pureté légèrement inférieure à Emin (4853 vs 5300), ce qui peut s'expliquer par des différences mineures d'initialisation ou d'ordre de traitement.

---

## 2. Tableau comparatif global

| Méthode | Vectorisation | Acc. test | Purity | ARI | R² |
|---|---|---|---|---|---|
| **Naive Bayes** | **TF-IDF** | **0.6875** | — | — | **0.0961** |
| Naive Bayes | BoW | 0.6666 | — | — | 0.0296 |
| K-Means | TF-IDF | 0.4636 | 0.5414 | 0.2481 | — |
| K-Means | Word2Vec | ~0.19 | ~0.51 | ~0.13 | — |

> **Note méthodologique** : la comparaison accuracy supervisé / non-supervisé est à interpréter avec précaution. Le Naive Bayes est entraîné avec les labels — il optimise directement la classification. Le K-Means cluster sans labels et n'a accès aux vraies étiquettes que *a posteriori* via le mapping par vote majoritaire. L'ARI est la métrique la plus fiable pour évaluer la qualité intrinsèque du clustering indépendamment de ce biais.

---

## 3. Difficultés communes observées

- **Déséquilibre des classes** : Normal (31%) et Depression (29%) dominent le corpus ; Personality disorder (2%) et Stress (5%) sont systématiquement sous-performants dans les deux approches.
- **Stress et Suicidal** : ces deux classes sont linguistiquement proches de Depression, ce qui génère des confusions récurrentes dans les matrices de confusion.
- **Word2Vec** : l'entraînement sur un corpus de taille moyenne (≈15k tokens uniques après stemming) limite la qualité des embeddings. Les résultats inférieurs à TF-IDF sur ce corpus suggèrent que la fréquence des termes est plus discriminante que leur contexte sémantique pour ce type de textes courts.

---

## 4. Décision : modèle retenu pour le Groupe 2

### ✅ Modèle retenu : **Naive Bayes + TF-IDF** (alpha = 0.05)

**Justification** :

Le Naive Bayes + TF-IDF est retenu pour trois raisons principales. Premièrement, il obtient la meilleure accuracy sur le test (68.75%), soit un écart de plus de 20 points par rapport au meilleur modèle non-supervisé (46.36%). Deuxièmement, ses prédictions sont directement des labels de classes (pas de mapping intermédiaire), ce qui les rend plus fiables et directement exploitables par le Groupe 2 pour le filtrage par sentiments. Troisièmement, le modèle est reproductible — Logan et Ulysse obtiennent exactement les mêmes résultats, ce qui valide la stabilité du pipeline.

Le K-Means présente un ARI correct (0.25 pour TF-IDF) mais une accuracy trop faible pour être utilisé comme annotateur fiable sur `train.csv`. Des erreurs d'annotation à 54% de fréquence introduiraient un bruit trop important dans le système Q/R du Groupe 2.

### Livrable pour le Groupe 2

Le modèle `nb_model_tfidf` (entraîné sur le split 60% train avec alpha=0.05) est appliqué sur `train.csv` pour produire les annotations automatiques. Le fichier résultant contient une colonne `predicted_sentiment` avec l'une des 7 classes pour chaque réponse de thérapeute. Ce fichier annoté est le point d'entrée du système de filtrage par sentiment du Groupe 2.

```python
# Snippet de déploiement sur train.csv
train_df = pd.read_csv("../../data/trains.csv")
train_tokens = train_df['Response'].apply(simple_tokenizer).apply(' '.join)
train_tfidf  = tfidf_vectorizer.transform(train_tokens)
train_df['predicted_sentiment'] = nb_model_tfidf.predict(train_tfidf)
train_df.to_csv("../../data/trains_annote.csv", index=False)
```