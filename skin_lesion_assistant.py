import os
import sys
import argparse
import warnings
import urllib.request
import zipfile
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense, Flatten, MaxPooling2D, concatenate,
    Activation, Dropout, Layer, InputSpec, Multiply, Concatenate
)
from tensorflow.keras import backend as K
import keras.layers as kl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

from evaluation_metrics import evaluate_full, per_class_metrics


CLASS_NAMES = ['nevus_comun', 'nevus_atipico', 'melanoma']
CLASS_DESCRIPTIONS = {
    'nevus_comun':   'Nevus Común / Lunar típico - Benigno, sin tratamiento necesario',
    'nevus_atipico': 'Nevus Atípico / Lunar displásico - Benigno con atipia, vigilancia recomendada',
    'melanoma':      'Melanoma - Maligno agresivo, requiere atención URGENTE'
}
CLASS_SEVERITY = {
    'nevus_comun':   'BAJA (benigno)',
    'nevus_atipico': 'MEDIA (benigno con atipia, vigilancia)',
    'melanoma':      'CRÍTICA (maligno agresivo)'
}
IMG_SIZE = 299
BATCH_SIZE = 16
EPOCHS = 50
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 2.0}

PH2_INFO = {
    'description': 'PH2 Dataset - 200 imágenes dermatoscópicas (3 clases)',
    'url': 'https://www.fc.up.pt/addi/ph2database.html',
    'kaggle': 'https://www.kaggle.com/datasets/ahmedelkomy/ph2-dataset'
}


def download_and_prepare_ph2(data_dir='./PH2'):
    os.makedirs(data_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, 'train_dir')
    test_dir = os.path.join(data_dir, 'test_dir')

    possible_paths = [
        os.path.join(os.getcwd(), 'PH2Dataset'),
        os.path.join(os.getcwd(), 'PH2Dataset', 'PH2Dataset'),
        os.path.join(os.getcwd(), 'PH2_Dataset'),
    ]
    if os.path.exists(os.path.join(data_dir, 'PH2_dataset.xlsx')):
        possible_paths.insert(0, data_dir)

    ph2_source = None
    for p in possible_paths:
        excel = os.path.join(p, 'PH2_dataset.xlsx')
        images = os.path.join(p, 'PH2 Dataset images')
        if os.path.exists(excel) and os.path.exists(images):
            ph2_source = p
            break

    if ph2_source is None:
        print(f"[ERROR] No se encontró PH2 Dataset en el directorio actual.")
        print(f"[INFO] Descarga PH2 desde Kaggle: {PH2_INFO['kaggle']}")
        print(f"[INFO] O desde web original: {PH2_INFO['url']} (registro gratuito)")
        print(f"[INFO] Coloca la carpeta 'PH2Dataset/' en: {os.getcwd()}")
        print(f"[INFO] Estructura esperada:")
        print(f"  PH2Dataset/")
        print(f"    ├── PH2_dataset.xlsx")
        print(f"    └── PH2 Dataset images/")
        print(f"          ├── IMD002/")
        print(f"          │   └── IMD002_Dermoscopic_Image/IMD002.bmp")
        print(f"          ├── IMD003/")
        print(f"          │   └── IMD003_Dermoscopic_Image/IMD003.bmp")
        print(f"          └── ...")
        return

    print(f"[DATA] PH2 Dataset encontrado en: {ph2_source}")
    images_source = os.path.join(ph2_source, 'PH2 Dataset images')
    metadata_path = os.path.join(ph2_source, 'PH2_dataset.xlsx')

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"[DATA] Splits train/test ya existen: {train_dir}, {test_dir}")
        return

    df = pd.read_excel(metadata_path, header=12)
    df = df.dropna(subset=['Image Name', 'Common Nevus', 'Atypical Nevus', 'Melanoma'], how='all')
    df = df.reset_index(drop=True)

    def get_class(row):
        if str(row.get('Melanoma', '')).strip().upper() == 'X':
            return 'melanoma'
        elif str(row.get('Atypical Nevus', '')).strip().upper() == 'X':
            return 'nevus_atipico'
        elif str(row.get('Common Nevus', '')).strip().upper() == 'X':
            return 'nevus_comun'
        return None

    df['dx'] = df.apply(get_class, axis=1)
    df = df.dropna(subset=['dx'])
    df = df.rename(columns={'Image Name': 'image_id'})
    df['image_id'] = df['image_id'].astype(str)

    image_map = {}
    for root, dirs, files in os.walk(images_source):
        for f in files:
            if f.lower().endswith('.bmp') and '_Dermoscopic_Image' in root:
                name = f.replace('.bmp', '')
                image_map[name] = os.path.join(root, f)

    print(f"[DATA] Imágenes .bmp encontradas: {len(image_map)}")
    df['path'] = df['image_id'].map(image_map)
    df = df.dropna(subset=['path'])

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['dx'], random_state=42)

    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for split_name, split_df, split_dir in [('train', train_df, train_dir),
                                              ('test', test_df, test_dir)]:
        for _, row in split_df.iterrows():
            src = row['path']
            dst = os.path.join(split_dir, row['dx'], f"{row['image_id']}.bmp")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"\n[DATA] Resumen PH2:")
    for cls in CLASS_NAMES:
        n_train = len(os.listdir(os.path.join(train_dir, cls)))
        n_test = len(os.listdir(os.path.join(test_dir, cls)))
        print(f"  {cls:<15} Train: {n_train:<3} Test: {n_test:<3}")
    print(f"[DATA] ¡Dataset listo! train_dir: {train_dir}, test_dir: {test_dir}")
    print(f"[DATA] Ejecuta ahora: --mode train --data_dir {data_dir}")


class SoftAttention(Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x
        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.i_shape = input_shape
        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads)
        self.out_attention_maps_shape = input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]

        if self.aggregate_channels is False:
            self.out_features_shape = input_shape[:-1] + (input_shape[-1] + (input_shape[-1] * self.multiheads),)
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape

        self.kernel_conv3d = self.add_weight(
            shape=kernel_shape_conv3d, initializer='he_uniform', name='kernel_conv3d')
        self.bias_conv3d = self.add_weight(
            shape=(self.multiheads,), initializer='zeros', name='bias_conv3d')
        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        exp_x = K.expand_dims(x, axis=-1)
        c3d = K.conv3d(exp_x, kernel=self.kernel_conv3d,
                       strides=(1, 1, self.i_shape[-1]), padding='same', data_format='channels_last')
        conv3d = K.bias_add(c3d, self.bias_conv3d)
        conv3d = kl.Activation('relu')(conv3d)
        conv3d = K.permute_dimensions(conv3d, pattern=(0, 4, 1, 2, 3))
        conv3d = K.squeeze(conv3d, axis=-1)
        conv3d = K.reshape(conv3d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2]))
        softmax_alpha = K.softmax(conv3d, axis=-1)
        softmax_alpha = kl.Reshape(target_shape=(self.multiheads, self.i_shape[1], self.i_shape[2]))(softmax_alpha)

        if self.aggregate_channels is False:
            exp_softmax_alpha = K.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = K.permute_dimensions(exp_softmax_alpha, pattern=(0, 2, 3, 1, 4))
            x_exp = K.expand_dims(x, axis=-2)
            u = Multiply()([exp_softmax_alpha, x_exp])
            u = kl.Reshape(target_shape=(self.i_shape[1], self.i_shape[2], u.shape[-1] * u.shape[-2]))(u)
        else:
            exp_softmax_alpha = K.permute_dimensions(softmax_alpha, pattern=(0, 2, 3, 1))
            exp_softmax_alpha = K.sum(exp_softmax_alpha, axis=-1)
            exp_softmax_alpha = K.expand_dims(exp_softmax_alpha, axis=-1)
            u = Multiply()([exp_softmax_alpha, x])

        if self.concat_input_with_scaled:
            o = Concatenate(axis=-1)([u, x])
        else:
            o = u
        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]

    def get_config(self):
        return super(SoftAttention, self).get_config()


def build_irv2_sa_model(freeze_backbone=True):
    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True, weights="imagenet",
        input_tensor=None, input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None, classifier_activation="softmax")
    if freeze_backbone:
        irv2.trainable = False
    conv = irv2.layers[-28].output
    attention_layer, _ = SoftAttention(
        aggregate=True, m=16, concat_with_x=False,
        ch=int(conv.shape[-1]), name='soft_attention')(conv)
    attn_pool = MaxPooling2D(pool_size=(2, 2), padding="same")(attention_layer)
    conv_pool = MaxPooling2D(pool_size=(2, 2), padding="same")(conv)
    x = concatenate([conv_pool, attn_pool])
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(CLASS_NAMES), activation='softmax')(x)
    model = Model(inputs=irv2.input, outputs=output)
    return model


def train(data_dir, save_path='IRV2+SA_PH2.weights.h5'):
    print("[INFO] Construyendo modelo IRV2 + Soft Attention (backbone congelado)...")
    model = build_irv2_sa_model(freeze_backbone=True)
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total = sum(tf.keras.backend.count_params(w) for w in model.weights)
    print(f"[INFO] Parámetros entrenables: {trainable:,} / {total:,} totales")

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    train_dir = os.path.join(data_dir, 'train_dir')
    test_dir = os.path.join(data_dir, 'test_dir')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"[ERROR] Los directorios train_dir/test_dir no existen en {data_dir}")
        print(f"[INFO] Usa --mode prepare_data para preparar PH2")
        print(f"[INFO]   python skin_lesion_assistant.py --mode prepare_data")
        return

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        shear_range=15)

    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)

    train_batches = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True)
    test_batches = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False)

    steps_per_epoch = max(1, train_batches.samples // BATCH_SIZE)

    checkpoint = ModelCheckpoint(
        filepath=save_path, monitor='val_accuracy',
        save_best_only=True, save_weights_only=True)
    print("[INFO] Iniciando entrenamiento ({EPOCHS} épocas completas)...")
    history = model.fit(
        train_batches, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
        verbose=1, validation_data=test_batches,
        callbacks=[checkpoint], class_weight=CLASS_WEIGHTS)

    model.load_weights(save_path)
    test_batches.reset()
    y_pred_proba = model.predict(test_batches, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_true = test_batches.classes[:len(y_pred_proba)]

    evaluate_full(
        y_true, y_pred_class, y_pred_proba,
        history=history, model_name='IRV2+SA_PH2',
        class_names=CLASS_NAMES, save_prefix='ph2_')

    return model, history


def predict_image(image_path, model_weights='IRV2+SA_PH2.weights.h5', show_attention=False):
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encuentra la imagen: {image_path}")
        return

    print(f"[INFO] Cargando imagen: {image_path}")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    print("[INFO] Construyendo modelo...")
    model = build_irv2_sa_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(model_weights):
        print(f"[INFO] Cargando pesos: {model_weights}")
        model.load_weights(model_weights)
    else:
        print(f"\n{'!' * 60}")
        print(f"  NO HAY PESOS ENTRENADOS: '{model_weights}' no existe")
        print(f"  El modelo usará pesos de ImageNet (sin fine-tuning en PH2).")
        print(f"  Las predicciones NO serán precisas para lesiones cutáneas.")
        print(f"{'!' * 60}")
        print(f"\n  Para entrenar el modelo:")
        print(f"  1. Preparar datos:  python skin_lesion_assistant.py --mode prepare_data")
        print(f"  2. Entrenar:        python skin_lesion_assistant.py --mode train --data_dir ./PH2")
        print(f"  3. Predecir:        python skin_lesion_assistant.py --mode predict --image ruta/imagen.jpg\n")

    print("[INFO] Realizando predicción...")
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_class = CLASS_NAMES[predicted_idx]

    print("\n" + "=" * 65)
    print("IRV2+SA - CLASIFICACIÓN DE LESIÓN CUTÁNEA (PH2)")
    print("=" * 65)
    print(f"\n  Imagen analizada: {os.path.basename(image_path)}")

    sorted_indices = np.argsort(predictions)[::-1]
    for i, idx in enumerate(sorted_indices):
        bar = "█" * int(predictions[idx] * 30) + "░" * (30 - int(predictions[idx] * 30))
        print(f"  {i+1}. {CLASS_NAMES[idx]:<8} {predictions[idx]:.1%} {bar}")

    print(f"\n  ▶ Diagnóstico más probable: {predicted_class.upper()}")
    print(f"  ▶ Confianza: {confidence:.1%}")
    print(f"  ▶ Descripción: {CLASS_DESCRIPTIONS[predicted_class]}")
    print(f"  ▶ Severidad: {CLASS_SEVERITY[predicted_class]}")
    print("\n" + "=" * 65)
    print("⚠  Referencia informativa. Consulte a un dermatólogo.")
    print("=" * 65 + "\n")

    if show_attention:
        _show_attention_map(model, img_array, image_path)

    return predicted_class, confidence


def _show_attention_map(model, img_array, image_path):
    try:
        sa_model = Model(inputs=model.inputs, outputs=model.get_layer('soft_attention').output)
        _, sa_maps = sa_model.predict(img_array, verbose=0)
        sum_attnmap = np.sum(sa_maps[0], axis=0)

        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

        attn_resized = cv2.resize(sum_attnmap, (IMG_SIZE, IMG_SIZE),
                                   interpolation=cv2.INTER_CUBIC)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(original)
        ax1.set_title('Imagen Original')
        ax1.axis('off')

        ax2.imshow(attn_resized, cmap='jet')
        ax2.set_title('Mapa de Atención')
        ax2.axis('off')

        ax3.imshow(original, alpha=0.6)
        ax3.imshow(attn_resized, cmap='jet', alpha=0.4)
        ax3.set_title('Superposición')
        ax3.axis('off')

        plt.tight_layout()
        plt.savefig('attention_map_result.png', dpi=150, bbox_inches='tight')
        print("[INFO] Mapa de atención guardado como 'attention_map_result.png'")
        plt.show()
    except Exception as e:
        print(f"[WARN] No se pudo generar mapa de atención: {e}")


def interactive_mode(model_weights='IRV2+SA_PH2.weights.h5'):
    print("=" * 65)
    print("   ASISTENTE DE CLASIFICACIÓN DE LESIONES CUTÁNEAS")
    print("   InceptionResNetV2 + Soft Attention (PH2)")
    print("=" * 65)
    print("\nComandos:")
    print("  predict <ruta>      - Analizar imagen")
    print("  attention <ruta>    - Analizar con mapa de atención")
    print("  help                - Ayuda")
    print("  classes             - Mostrar clases")
    print("  quit                - Salir\n")

    while True:
        try:
            cmd = input(">>> ").strip().split()
            if not cmd:
                continue
            if cmd[0] in ('quit', 'exit', 'q'):
                print("¡Hasta luego!")
                break
            elif cmd[0] == 'classes':
                print("\nClases de lesiones dermatológicas (PH2):")
                print("-" * 50)
                for cls in CLASS_NAMES:
                    print(f"  {cls:<15} - {CLASS_DESCRIPTIONS[cls]}")
                print()
            elif cmd[0] == 'help':
                print("\n  predict <ruta>      - Analizar imagen")
                print("  attention <ruta>    - Analizar con mapa de atención")
                print("  classes             - Ver clases")
                print("  quit                - Salir\n")
            elif cmd[0] in ('predict', 'attention'):
                if len(cmd) < 2:
                    print("  Error: falta ruta")
                    continue
                predict_image(cmd[1], model_weights, show_attention=(cmd[0] == 'attention'))
            else:
                print(f"  Desconocido: {cmd[0]}. Escribe 'help'")
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Asistente de Clasificación de Lesiones Cutáneas')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive', 'prepare_data'],
                        default='interactive')
    parser.add_argument('--data_dir', type=str, default='./PH2',
                        help='Directorio del dataset')
    parser.add_argument('--image', type=str, default=None,
                        help='Ruta a la imagen para predecir')
    parser.add_argument('--weights', type=str, default='IRV2+SA_PH2.weights.h5',
                        help='Archivo de pesos del modelo')
    parser.add_argument('--show_attention', action='store_true',
                        help='Mostrar mapa de atención en modo predict')

    args = parser.parse_args()

    if args.mode == 'prepare_data':
        download_and_prepare_ph2(args.data_dir)
    elif args.mode == 'train':
        train(args.data_dir, args.weights)
    elif args.mode == 'predict':
        if args.image is None:
            print("Error: Modo predict requiere --image <ruta>")
            sys.exit(1)
        predict_image(args.image, args.weights, args.show_attention)
    elif args.mode == 'interactive':
        interactive_mode(args.weights)
