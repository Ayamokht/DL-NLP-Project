# src/cnn_train.py

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing import DataLoader, TextCleaner


def main():
    # --- Création du dossier de sortie ---
    output_dir = 'Outputs'
    os.makedirs(output_dir, exist_ok=True)

    # --- Paramètres ---
    MAX_WORDS = 50000
    MAX_LEN = 300
    EMBED_DIM = 300
    BATCH_SIZE = 32
    EPOCHS = 10
    DATA_X = "data/X_train_update.csv"
    DATA_Y = "data/Y_train_CVw08PX.csv"

    # --- Chargement et préparation des données ---
    loader = DataLoader(DATA_X, DATA_Y)
    df = loader.get_dataframe()
    cleaner = TextCleaner()
    df['clean_text'] = cleaner.transform_series(df['full_text'])

    # Encodage des labels
    le = LabelEncoder().fit(df['prdtypecode'])
    y = le.transform(df['prdtypecode'])

    # Split train/validation
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        df['clean_text'], y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --- Tokenization et padding ---
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_raw)
    X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
    X_val_seq = tokenizer.texts_to_sequences(X_val_raw)
    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    # One-hot labels
    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    # --- Construction du modèle CNN multicanal ---
    inp = tf.keras.layers.Input(shape=(MAX_LEN,))
    x = tf.keras.layers.Embedding(MAX_WORDS, EMBED_DIM)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((MAX_LEN, EMBED_DIM, 1))(x)

    pools = []
    for size in (2, 3, 4, 5):
        c = tf.keras.layers.Conv2D(
            128, (size, EMBED_DIM), activation='relu', kernel_regularizer=l2(1e-3)
        )(x)
        c = tf.keras.layers.BatchNormalization()(c)
        pools.append(tf.keras.layers.GlobalMaxPooling2D()(c))
    x = tf.keras.layers.Concatenate()(pools)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-3))(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    # Class weights
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(np.unique(y_train), weights))

    # --- Entraînement ---
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, class_weight=class_weight, verbose=1
    )

    # --- Évaluation ---
    y_pred = np.argmax(model.predict(X_val), axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"Validation accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    print(classification_report(y_val, y_pred))

    # --- Sauvegarde ---
    model.save(os.path.join(output_dir, 'cnn_model.h5'))
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == '__main__':
    main()
