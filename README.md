# Application CBIR avec Classification d’Images

## Description
Ce projet implémente un système de recherche d’images basé sur le contenu (CBIR) avec une phase de classification automatique. L’objectif est de prédire la classe d’une image requête afin de limiter la recherche aux images pertinentes.

## Objectifs
- Extraire des caractéristiques visuelles (GLCM, Haralick, Bitdesc)
- Comparer plusieurs modèles de classification
- Sélectionner le meilleur modèle
- Implémenter un système CBIR
- Développer une interface web avec Streamlit
- Ajouter une authentification

## Technologies
- Python
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- SciPy
- Mahotas
- Scikit-image
- Joblib


## Extraction des caractéristiques
Les descripteurs GLCM, Haralick et Bitdesc sont utilisés. Les caractéristiques sont concaténées et sauvegardées dans des fichiers `.npy`.

## Classification
Les modèles utilisés sont :
- Random Forest
- Decision Tree
- SVM

Le meilleur modèle est sélectionné selon les métriques et sauvegardé dans `best_model.pkl`.

## CBIR
Pour une image requête :
- Extraction des caractéristiques
- Prédiction de la classe
- Filtrage des images
- Calcul des distances (Euclidienne, Canberra, Cosinus)
- Affichage des images similaires

## Application Web
L’application permet :
- Upload d’image
- Choix du descripteur
- Choix de la distance
- Choix du nombre d’images
- Affichage des résultats

## Authentification
- Nom d’utilisateur : Angel
- Mot de passe : 1234

## Exécution


Lancer l’application : streamlit run app.py