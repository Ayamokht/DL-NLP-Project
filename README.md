# 🧠 DL-NLP Project

Projet Deep Learning & NLP — Classification des produits Rakuten France :
Étude comparative des méthodes textuelles et
visuelles.

## 🚀 Objectif

Prédire le code produit `prdtypecode` à partir :

- des titres (`designation`)
- des descriptions (`description`)
- des images associées


Ce projet combine des méthodes classiques de traitement du langage naturel (comme TF-IDF couplé à une régression logistique) avec des modèles profonds plus avancés, tels que TextCNN pour le texte et ResNet50 pour les images, afin de prédire efficacement le code produit (prdtypecode).

---

## 🗂️ Structure du projet

```text
DL_NLP_project/
├── data/                 # Données d'entraînement et test (.csv, .jpg) — ignoré par Git
├── Outputs/              # Modèles entraînés, prédictions, tokenizer — ignoré par Git
├── sandbox/              # Notebooks d'exploration (image, texte, etc.)
├── src/                  # Code source du projet
│   ├── preprocessing.py        # Chargement et nettoyage des données texte
│   ├── tfidf_regression.py     # Baseline TF-IDF + régression logistique
│   ├── TextCNN_train.py        # Entraînement du modèle TextCNN et ses performances
│   └── ClassImage.py           # Préprocessing et entraînement image (EfficientNet, ResNet50)
├── main.py              # Script principal qui orchestre l'exécution complète
├── requirements.txt     # Liste des dépendances à installer
└── .gitignore           # Fichiers et dossiers exclus du suivi Git

```


---

## ⚙️ Installation

---

## Travailler sur le Projet

1️⃣ Cloner le projet
```bash
git clone https://github.com/Ayamokht/DL-NLP-Project.git
```
Ensuite se positionner où il y a le fichier cloné

```bash
cd DL-NLP-Project
```
2️⃣ Création d'un environment virtuel

```bash
python3 -m venv .venv
```
3️⃣ Activer l'environnement virtuel

```bash
source .venv/bin/activate   # Sur Windows : .venv\Scripts\activate
```
4️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

## 👥 Authors
- [Tonin RIVORY](https://github.com/ton1rvr)
- [Aya MOKHTAR](https://github.com/ayamokhtar)

