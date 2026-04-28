# Étape 2 — Ma partie (K-Means non-supervisé)

## Contexte

J'étais avec Victor sur la méthode non-supervisée (K-Means). Victor a commencé avant moi et a fait toute l'implémentation de base : K-Means avec Word2Vec, le mapping cluster -> label, et l'évaluation. Quand je suis arrivé, sa partie était déjà finie, donc j'ai pris en charge ce qui restait à faire.

## Ce que j'ai fait

### 1. K-Means avec TF-IDF (section B.3bis)

Victor avait fait tourner K-Means uniquement avec Word2Vec. Pour pouvoir comparer les deux représentations (ce que demandait la section B.4 du template), j'ai refait le pipeline complet avec TF-IDF :
- Sélection du k optimal (méthode du coude + silhouette)
- Entraînement K-Means + mapping cluster -> label
- Évaluation (accuracy, purity, ARI, matrice de confusion)

J'ai gardé la même logique que Victor (mêmes paramètres, même structure de code), juste en changeant les variables d'entrée pour utiliser TF-IDF au lieu de Word2Vec.

### 2. Comparaison TF-IDF vs Word2Vec (section B.4)

J'ai construit le tableau comparatif avec les chiffres des deux méthodes et écrit l'analyse.

### 3. Correction du bug de la cellule B.6 (déploiement)

Le template demandait d'utiliser TF-IDF dans le pipeline mais de prédire avec le modèle K-Means entraîné sur Word2Vec, ce qui est incohérent (les deux espaces vectoriels n'ont rien à voir). Victor avait déjà signalé ce problème dans son README. Comme on a retenu TF-IDF dans la comparaison B.4, j'ai modifié la cellule B.6 pour qu'elle utilise `kmeans_final_tfidf` et `cluster_to_label_tfidf` partout. Comme ça le pipeline est cohérent de bout en bout.

J'ai aussi rajouté quelques `print` pour vérifier que tout se passe bien (nombre de lignes chargées, distribution des prédictions).

### 4. B.5 (à faire)

La comparaison avec la méthode supervisée (Naive Bayes) sera faite quand le sous-groupe de Diogo aura fini sa partie.

## Ce que j'ai trouvé d'intéressant

Le truc qui m'a surpris c'est que **TF-IDF fait largement mieux que Word2Vec** sur cette tâche, alors que je m'attendais à l'inverse (Word2Vec capture la sémantique, TF-IDF est juste du comptage).

En creusant, j'ai compris pourquoi : Word2Vec a en fait des clusters individuels très purs (Normal à 0.90, Depression à 0.87), mais ses deux plus gros clusters contiennent surtout du Depression et du Suicidal mélangés. Comme la stratégie de mapping force chaque cluster à prendre une classe différente, ces gros clusters se retrouvent étiquetés "Stress" et "Personality disorder" alors qu'ils ne représentent pas du tout ces classes-là — d'où l'accuracy qui s'écroule à 20%.

En gros : Word2Vec a regroupé les textes selon une logique sémantique cohérente, mais qui ne correspond pas à la taxonomie du dataset. Pour notre cas (où on veut reproduire les vraies classes), TF-IDF est mieux adapté.

## Modèle retenu

**K-Means + TF-IDF** pour annoter `train.csv` qui sera transmis au Groupe 2.
- Accuracy : 0.46
- Purity globale : 0.54
- ARI : 0.25

Fichier de sortie : `train_with_sentiments_kmeans.csv`

## Modifs côté projet

J'ai aussi fixé un conflit de versions entre `gensim` et `scipy` dans le `requirements.txt` (la fonction `triu` a été retirée de scipy 1.13+, ce qui cassait l'import de gensim). J'ai ajouté `langdetect` aussi parce qu'il était utilisé dans la cellule de cleaning sans être dans le requirements.