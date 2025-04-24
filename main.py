from src.preprocessing import DataLoader, TextCleaner
from src.tfidf_regression import main as tfidf_main
from src.TextCNN_train import main as cnn_main
from src.ClassImage import ImagePreprocessor, ImageGeneratorBuilder, CustomClassifier
import pandas as pd

def main():
    print("\n=== Préprocessing des données texte ===")
    loader = DataLoader(
        x_path="data/X_train_update.csv",
        y_path="data/Y_train_CVw08PX.csv",
        image_folder="data/image_train"  # à préciser si utilisé dans DataLoader
    )
    df = loader.get_dataframe()
    cleaner = TextCleaner()
    df['clean_text'] = cleaner.transform_series(df['full_text'])
    print(f"Données prétraitées : {df.shape[0]} lignes, colonnes -> {list(df.columns)}")


    print("\n=== Baseline TF–IDF + Régression logistique ===")
    tfidf_main()


    print("\n=== Entraînement TextCNN ===")
    cnn_main()

    print("\n=== Prétraitement des images ===")
    preprocessor = ImagePreprocessor(img_size=(224, 224), output_dir="Outputs/images_cleaned")
    df["image_path"] = "data/image_train/" + df["imageid"].astype(str) + ".jpg"  # vérifie cette ligne selon ton cas
    df_filtered = preprocessor.apply_filtering_and_save(df, path_column="image_path")


    print("\n=== Génération des données images ===")
    generator_builder = ImageGeneratorBuilder(img_size=(224, 224), batch_size=32, augment=True)
    train_gen, val_gen = generator_builder.build_generators(
        df_filtered,
        path_col="cleaned_path",
        label_col="prdtypecode",
        test_size=0.1,
        seed=42
    )

    num_classes = df_filtered["prdtypecode"].nunique()


    print("\n=== Entraînement Custom Classifier (CNN Images) ===")
    classifier = CustomClassifier(
        model_name='efficientnetb0',
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        learning_rate=1e-4,
        fine_tune=True,
        two_phase=True
    )

    classifier.summary()

    classifier.train(
        train_data=train_gen,
        val_data=val_gen,
        data_type='generator',
        epochs=10,
        phase1_epochs=3,
        phase2_epochs=5
    )

    results = classifier.evaluate(val_gen)
    print(f"\nRésultats Classification Images - Val Loss: {results[0]:.4f}, Val Accuracy: {results[1]:.4f}")

if __name__ == '__main__':
    main()
