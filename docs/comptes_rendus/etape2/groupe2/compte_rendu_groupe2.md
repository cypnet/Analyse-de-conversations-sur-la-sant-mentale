# Système de Question-Réponse en santé mentale

## Objectif 

L'objectif de cette partie est de développer un système de question-réponse (Q&A) capable de proposer des réponses pertinentes à des questions liées à la santé mentale.

## Données 

Le corpus est composé de paires Question-Réponse issues de conversations thérapeuthiques.

Chaque ligne contient : 
- une question posée par un patient
- une réponse fournie par un thérapeute

## Indexation 

Un index inversé est construit : 

A chaque mot, on associe la liste des questions et réponses contenant ce mot.

## Méthode 

Deux approches ont été implémentées : 

### Méthode 1 : Similarité entre questions

1. Transformer les questions du corpus et la question de l'utilisateur en vecteurs
2. Comparer la question utilisateur avec ces vecteurs
3. Identifier les K questions les plus similaires
4. Retourner les réponses associéés

### Méthode 2 : Similarité entre réponses

1. Transformer les réponses du corpus et la question de l'utilisateur en vecteurs
2. Comparer la question utilisateur avec ces vecteurs
3. Identifier les K réponses les plus similaires
4. Retourner directement ces réponses 

## Vectorisation

Les textes sont transformés en vecteurs numériques à l'aide de : 
- TF-IDF
- Word2Vec
- BERT (optionnel)

Cette étape est essntielle pour permettre la comparaison entre textes

## Mesure de similarité

La similarité entre deux vecteurs est calculée avec cosine similarity

Plus le score est proche de 1, plus les textes sont similaires

## Evaluation

Un mini jeu de test de 10 couples de questions-réponses a été utilisé

### Métriques d'évaluation
\[
Ratio = \frac{Nombre\ de\ bonnes\ réponses}{Nombre\ total\ de\ questions}
\]