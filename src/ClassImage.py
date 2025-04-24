import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


class ImagePreprocessor:
    def __init__(self, img_size=(224, 224), output_dir="images_cleaned"):
        self.img_size = img_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_and_filter_image(self, path):
        try:
            # Chargement image 224x224
            img = load_img(path, target_size=self.img_size)
            img_array = img_to_array(img).astype(np.uint8)

            # Vérification taille brute
            img_raw = cv2.imread(path)
            if img_raw is None or img_raw.shape[0] < 100 or img_raw.shape[1] < 100:
                return None

            # Image quasi vide (std faible)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            if np.std(gray) < 5:
                return None

            # Final preprocessing
            return preprocess_input(img_array.astype(np.float32))

        except Exception as e:
            print(f"[Erreur chargement] {path} : {e}")
            return None

    def save_filtered_image(self, row):
        try:
            filtered = self.load_and_filter_image(row["image_path"])
            if filtered is None:
                return None

            # Re-normaliser pour sauvegarde PNG propre
            arr_uint8 = np.clip((filtered + 1.0) * 127.5, 0, 255).astype(np.uint8)
            output_path = os.path.join(self.output_dir, os.path.basename(row["image_path"]))
            Image.fromarray(arr_uint8).save(output_path)
            return output_path

        except Exception as e:
            print(f"[Erreur sauvegarde] {row['image_path']} : {e}")
            return None

    def apply_filtering_and_save(self, df, path_column="image_path"):
        tqdm.pandas()
        df["cleaned_path"] = df.progress_apply(self.save_filtered_image, axis=1)
        df = df[df["cleaned_path"].notnull()].reset_index(drop=True)
        return df

    def load_cleaned_images(self, df, path_column="cleaned_path"):
        tqdm.pandas()
        return np.stack(df[path_column].progress_apply(self._load_cleaned_image).values)

    def _load_cleaned_image(self, path):
        img = load_img(path, target_size=self.img_size)
        img_array = img_to_array(img).astype(np.float32)
        return preprocess_input(img_array)
    



class ImageGeneratorBuilder:
    def __init__(self, img_size=(224, 224), batch_size=32, augment=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment

    def build_generators(self, df, path_col="image_path", label_col="prdtypecode", test_size=0.1, seed=42):
        # Séparer entraînement / validation
        df_train, df_val = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df[label_col], 
            random_state=seed
        )

        # Convertir labels en string
        df_train[label_col + "_str"] = df_train[label_col].astype(str)
        df_val[label_col + "_str"] = df_val[label_col].astype(str)

        # Générateur avec ou sans augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True if self.augment else False,
            zoom_range=0.1 if self.augment else 0.0
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Création des générateurs
        train_gen = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col=path_col,
            y_col=label_col + "_str",
            target_size=self.img_size,
            class_mode="categorical",
            batch_size=self.batch_size,
            shuffle=True
        )

        val_gen = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col=path_col,
            y_col=label_col + "_str",
            target_size=self.img_size,
            class_mode="categorical",
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_gen, val_gen


from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import inspect


from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


class CustomClassifier:
    def __init__(self, 
                 model_name='resnet50',
                 input_shape=(224, 224, 3),
                 num_classes=10,
                 learning_rate=1e-4,
                 fine_tune=False,
                 two_phase=False):
        
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.fine_tune = fine_tune
        self.two_phase = two_phase
        self.model = None
        self.base_model = self._load_base_model()

    def _load_base_model(self):
        if self.model_name == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_name == 'efficientnetb0':
            return EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_name == 'efficientnetb2':
            return EfficientNetB2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Modèle '{self.model_name}' non supporté.")

    def _build_model(self, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate

        model = Sequential([
            self.base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def summary(self):
        return self.model.summary()

    def train(self, train_data, val_data,
              data_type='generator',
              epochs=10, phase1_epochs=5, phase2_epochs=5):
        """
        data_type: 'generator' ou 'array'
        - generator : train_data = train_gen, val_data = val_gen
        - array : train_data = (X_train, y_train), val_data = (X_val, y_val)
        """
        lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
        early_stop = EarlyStopping(patience=3, restore_best_weights=True, verbose=1)

        if self.two_phase:
            print("\n=== Phase 1 : gel complet du modèle ===")
            self.base_model.trainable = False
            self._build_model(learning_rate=self.learning_rate)

            if data_type == 'generator':
                self.model.fit(train_data, validation_data=val_data,
                               epochs=phase1_epochs, callbacks=[lr_callback, early_stop])
            else:
                X_train, y_train = train_data
                X_val, y_val = val_data
                self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                               epochs=phase1_epochs, callbacks=[lr_callback, early_stop])

            print("\n=== Phase 2 : dégel partiel des dernières couches ===")
            for layer in self.base_model.layers[:-20]:
                layer.trainable = False
            for layer in self.base_model.layers[-20:]:
                layer.trainable = True

            self._build_model(learning_rate=self.learning_rate / 10)
            if data_type == 'generator':
                return self.model.fit(train_data, validation_data=val_data,
                                      epochs=phase2_epochs, callbacks=[lr_callback, early_stop])
            else:
                return self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                      epochs=phase2_epochs, callbacks=[lr_callback, early_stop])

        else:
            print("\n=== Entraînement standard ===")
            self.base_model.trainable = self.fine_tune
            self._build_model()

            if data_type == 'generator':
                return self.model.fit(train_data, validation_data=val_data,
                                      epochs=epochs, callbacks=[lr_callback, early_stop])
            else:
                X_train, y_train = train_data
                X_val, y_val = val_data
                return self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                      epochs=epochs, callbacks=[lr_callback, early_stop])

    def evaluate(self, data):
        return self.model.evaluate(data)

    def predict(self, data):
        return self.model.predict(data)
