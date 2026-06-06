# Analyse de conversations patient-thérapeute sur la santé mentale

<div align="center">
  <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Flatestlogo.com%2Fwp-content%2Fuploads%2F2025%2F07%2Funiversite-de-toulouse-logo-scaled.png&f=1&nofb=1&ipt=e30d2c423bcf905088eba3bbf558bfd8e3cd3f9f4de9dee853e1eeaefa627440" alt="Université de Toulouse" width="300">
  &nbsp;&nbsp;&nbsp;
  <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.irit.fr%2FSTORM%2Fsite%2Fwp-content%2Fuploads%2F2022%2F10%2FLogo_intitule_horiz_1ligne_tutelles_PNG.png&f=1&nofb=1&ipt=4fb92cd5e46a5343e8d3969f6e82e3e26b32dab3a70a4b6fd0235f049f6d3f2b" alt="IRIT" width="300">
</div><br>


> Projet universitaire — Université de Toulouse | IRIT | [Info5.BE-SHS]  
> Encadrante : **Lynda Tamine-Lechani**

---

## 📋 Description

Ce projet vise à analyser un corpus de conversations entre patients et thérapeutes afin de :
- Comprendre la structure et les thématiques des échanges
- Détecter les émotions et états psychologiques des patients (analyse de sentiments)
- Développer deux outils d'accès à l'information : un système d'**analyse des sentiments** et un système de **question-réponse** sur la santé mentale

---

## 📁 Structure du dépôt

```
├── data
│   ├── combined_data.csv
│   ├── test.csv
│   ├── train.csv
│   ├── train_unique_e2p2.csv
│   └── train_with_sentiments.csv
├── docs
│   ├── BE___Cahier_des_charges.pdf
│   ├── bilan_mi_parcours
│   │   ├── bilan-mi-parcours.pdf
│   │   └── bilan-mi-parcours.tex
│   ├── compte_rendu_final
│   │   ├── compte_rendu_final.pdf
│   │   └── compte_rendu_final.tex
│   ├── comptes_rendus
│   │   ├── etape1
│   │   │   ├── compte_rendu_etape1.md
│   │   │   ├── compte_rendu_etape1.pdf
│   │   │   └── compte_rendu_etape1.tex
│   │   └── etape2
│   │       ├── groupe1
│   │       │   ├── compte_rendu_etape2_BERT.pdf
│   │       │   ├── compte_rendu_etape2_BERT.tex
│   │       │   ├── compte_rendu_etape2.pdf
│   │       │   ├── compte_rendu_etape2.tex
│   │       │   └── compte_rendu_groupe_1.md
│   │       └── groupe2
│   │           ├── compte_rendu_etape2.pdf
│   │           └── compte_rendu_groupe2.md
│   ├── img
│   │   └── ...
│   └── structure_resume_etape2
│       ├── structure_etape2.pdf
│       └── structure_etape2.tex
├── img
│   ├── IRIT_logo.png
│   └── universite-de-toulouse-logo.png
├── notebooks
│   ├── CRUZDiogo
│   ├── EMIN
│   ├── idir
│   ├── LAURE
│   ├── LOGAN
│   ├── LUCAS
│   ├── Trésor
│   ├── ULYSSE
│   ├── Victor
│   └── Yvan
├── README.md
└── requirements.txt

```

---

## ⚙️ Installation

**Prérequis :** Python 3.9+

```bash
# Cloner le dépôt
git clone https://github.com/cypnet/Analyse-de-conversations-sur-la-sant-mentale.git
cd 

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

**Une démonstration du projet est disponible** sur [HugginFace 🤗](https://huggingface.co/spaces/dempxl/sentiment-analysis-for-mental-health)  

Ce projet n'a pas pour but d'être exécuté dans sa totalité, celui-ci propose divers dossiers dans lesquels se situent les notebook Jupyter de l'équipe que vous pouvez exécuter à part.

Les notebooks Jupyter dans `notebooks/` contiennent les analyses détaillées et les visualisations associées à chaque étape.

---

## 🔬 Méthode

### Étape 1 — Exploration du jeu de données
- **Pré-traitement** : tokenisation, lemmatisation, suppression des stop-words
- **Statistiques descriptives** : nombre de patients, distribution des conversations, longueur moyenne des échanges
- **Analyse lexicale** : n-grammes, fréquences d'apparition par patient et par thérapeute
- **Topic Modeling** : comparaison de 4 approches (lexique manuel, TF-IDF + K-Means, Word2Vec/GloVe + clustering, LDA)

### Étape 2 — Outils d'accès à l'information
| Groupe | Tâche |
|--------|-------|
| **Groupe 1** | Analyse des sentiments des conversations |
| **Groupe 2** | Question-réponse sur la santé mentale |  

#### Groupe 1 — Analyse des sentiments des conversations
La première consiste à implémenter un "catégoriseur" de sentiments sur les question-réponses des thérapeutes selon une liste prédéfinie par les thérapeutes 
(ex. Anxiété, Normal, Dépression, etc.) sur la base uniquement du contenu (mots, phrases, sujet).

#### Groupe 2 — Question-réponse sur la santé mentale
La seconde, quant à elle, devra implémenter un système de question-réponse simple sur le thème de la santé mentale, à une question en langage naturelle,
le système retourne une liste de réponses, candidates ordonnées selon un score de pertinence.

> L'étape 2 étant encore nouvelle, peut légèrement changer
---

## 📦 Données

Le jeu de données utilisé sur l'étape 1 est publié sur [Kaggle](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations) et contient des conversations patient-thérapeute structurées en deux colonnes :
- `patient` : message du patient
- `therapist` : réponse du thérapeute  

Le jeu de donnée utilisé pour l'étape 2 est publié sur [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data) et contient un corpus d'entraînement sur 
les catégories des sentiments où les annotations sont fournies manuellement par les thérapeutes.


> ⚠️ Les fichiers `train.csv` et `combined_data.csv` sont commun à toute l'équipe et se situe dans le dossier `data/`.

---

## 📅 Calendrier & Livrables

| Livrable | Date prévue | Statut |
|----------|-------------|--------|
| Cahier des charges | 15 mars 2026 | [x] Rendu |
| Bilan de mi-parcours | 20 avril 2026 | [x] Rendu |
| Livrable technique mensuel | Mensuel | [x] Terminé |
| Compte rendu des étapes | Chaque fin <br> d'étapes | [x] Terminé |
| Rapport final + documentation technique | 1 juin 2026 | [ ] En cours |

---

## 👥 Équipe

**Chef de projet :** Logan LARROUX

| Groupe 1 — Analyse des sentiments des conversations | Groupe 2 — Q&R sur la santé mentale |
|--------------------------------------|--------------------------------------|
| Cruz FERNANDES DIOGO (Chef de groupe) | Idir YAHIAOUI (Chef de groupe) |
| Ulysse CHASSEIGNE | Bantsoukissa LAURE |
| Logan LARROUX | Esso-Y-N Trésor PANA |
| Emin BELKHEIR | Yvan SELLIER |
| Victor BLANQUART-BLANCHET | Lucas DA SILVA |

---

## 📄 Licence

Projet académique — Université de Toulouse, 2026. Usage éducatif uniquement.