1\. Le choix de l'outil : Word2Vec au lieu de TF-IDF



* Pourquoi ? Parce que Word2Vec comprend le sens des mots. Il sait que "triste" et "déprimé" sont des mots proches. C'est beaucoup plus efficace pour regrouper des émotions que TF-IDF (qui ne fait que compter les mots).



2\. La création des groupes (K-Means)



* Le nombre de groupes (k) : On ne l'a pas choisi au hasard. Le code teste tout seul de faire entre 2 et 15 groupes, et il garde automatiquement la configuration qui donne les groupes les plus nets (grâce au "score de silhouette").



3\. Donner un nom aux groupes (Le Mapping)



* Le problème : L'algorithme crée des groupes numérotés (Groupe 0, Groupe 1...), il ne sait pas ce qu'est la "Dépression".



* La solution (Vote majoritaire) : On regarde les vraies étiquettes cachées dans chaque groupe. Si le Groupe 0 contient 80% de textes parlant d'Anxiété, on baptise officiellement le Groupe 0 "Anxiété".



* En cas de conflit : Si deux groupes différents réclament l'étiquette "Anxiété", c'est le groupe le plus "pur" (celui qui a le plus grand pourcentage d'anxiété) qui gagne. L'autre groupe prendra son deuxième choix.



4\. Le point crucial pour la fin (Déploiement sur train.csv)



* Interdiction de ré-entraîner : Pour faire les prédictions finales sur les nouvelles données, il faut absolument utiliser le modèle tel qu'il est. Dans le code, on utilise uniquement .transform() et .predict() (surtout pas de .fit(), sinon ça efface la mémoire du modèle).



* L'erreur du template : Le fichier de base demande d'utiliser la méthode TF-IDF pour l'application finale (cellule B.6), alors qu'on vient d'entraîner le modèle avec Word2Vec. Il ne faut pas suivre le template ici : il faut utiliser Word2Vec jusqu'au bout, sinon ça va planter car les dimensions mathématiques ne seront pas les mêmes.
