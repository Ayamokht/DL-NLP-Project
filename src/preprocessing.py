# src/preprocessing.py

import pandas as pd
import os
import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords

class DataLoader:
    """
    Charge X et Y, fusionne 'designation' + 'description' en 'full_text',
    ajoute chemin et disponibilité des images si spécifié.
    """
    def __init__(self,
                 x_path: str,
                 y_path: str,
                 image_folder: str = None,
                 designation_col: str = 'designation',
                 description_col: str = 'description',
                 label_col: str = 'prdtypecode'):
        # Charger et nettoyer les CSV
        self.x = pd.read_csv(x_path).drop(columns=["Unnamed: 0"], errors="ignore")
        self.y = pd.read_csv(y_path).drop(columns=["Unnamed: 0"], errors="ignore")
        # Fusion des données
        self.df = pd.concat([self.x, self.y], axis=1)
        # Fusion texte
        self.df['full_text'] = (
            self.df[designation_col].fillna("") + " "
            + self.df[description_col].fillna("")
        )
        # Gestion des images si dossier fourni
        if image_folder:
            self.df['image_filename'] = self.df.apply(
                lambda r: f"image_{r['imageid']}_product_{r['productid']}.jpg", axis=1)
            self.df['image_path'] = self.df['image_filename'].apply(
                lambda fn: os.path.join(image_folder, fn))
            self.df['image_exists'] = self.df['image_path'].apply(
                lambda p: Path(p).exists())

    def get_dataframe(self) -> pd.DataFrame:
        """Retourne la DataFrame enrichie (full_text, image_path, etc.)."""
        return self.df.copy()


class TextCleaner:
    """
    Nettoyage de texte :
     - HTML, entités, ponctuation, chiffres
     - passage en minuscules
     - suppression des stopwords FR+EN
     
     - collapse espaces multiples
    """
    def __init__(self, languages=('french', 'english')):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.stop_words = set()
        for lang in languages:
            self.stop_words |= set(stopwords.words(lang))

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1. HTML et entités
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        # 2. Ponctuation et chiffres
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # 3. Minuscules
        text = text.lower()
        # 4. Stopwords
        tokens = text.split()
        tokens = [w for w in tokens if w not in self.stop_words]
        # 5. Collapse espaces
        return re.sub(r'\s+', ' ', ' '.join(tokens)).strip()

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Applique clean_text à chaque élément de la série."""
        return series.fillna("").apply(self.clean_text)


if __name__ == '__main__':
    # --- Exemple de test rapide ---
    X_CSV = "data/X_train_update.csv"
    Y_CSV = "data/Y_train_CVw08PX.csv"
    IMAGE_DIR = "data/image_train"

    # Chargement
    loader = DataLoader(X_CSV, Y_CSV, image_folder=IMAGE_DIR)
    df = loader.get_dataframe()
    print("Shape df:", df.shape)
    print(df[['designation','description','full_text']].head(3))

    # Vérifier les images
    if 'image_exists' in df:
        print("Images trouvées:", df['image_exists'].sum(),"/", len(df))

    # Nettoyage
    cleaner = TextCleaner()
    df['clean_text'] = cleaner.transform_series(df['full_text'])
    print("Extrait clean_text:", df['clean_text'].head(3))
