# src/tfidf_regression.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.preprocessing import DataLoader, TextCleaner


def main():
    # --- Chemins des données ---
    X_PATH = "data/X_train_update.csv"
    Y_PATH = "data/Y_train_CVw08PX.csv"

    # --- Chargement et fusion des champs texte ---
    loader = DataLoader(X_PATH, Y_PATH)
    df = loader.get_dataframe()

    # --- Nettoyage du texte fusionné ---
    cleaner = TextCleaner()
    df['clean_text'] = cleaner.transform_series(df['full_text'])

    # --- Encodage des labels ---
    le = LabelEncoder().fit(df['prdtypecode'])
    y = le.transform(df['prdtypecode'])
    X = df['clean_text']

    # --- Séparation train/validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Taille train: {len(X_train)}, validation: {len(X_val)}")

    # --- Vectorisation TF-IDF ---
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    print(f"X_train_tfidf: {X_train_tfidf.shape}, X_val_tfidf: {X_val_tfidf.shape}")

    # --- Entraînement du modèle ---
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='saga',
        multi_class='multinomial',
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)

    # --- Évaluation ---
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, average='weighted')
    print(f"Accuracy: {acc:.4f}, F1-score pondéré: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))


if __name__ == '__main__':
    main()
