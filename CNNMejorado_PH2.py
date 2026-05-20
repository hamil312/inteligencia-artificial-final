import os
import sys
import argparse
import urllib.request
import zipfile
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import warnings

from evaluation_metrics import evaluate_full, per_class_metrics
warnings.filterwarnings("ignore")


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

IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 80
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 5.0}
L2_REG = 1e-4
TTA_N = 5

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


def build_cnn_mejorado_model():
    model = models.Sequential(name="CNN_Mejorado_PH2")
    model.add(layers.Conv2D(48, (3, 3), padding='same', activation='relu',
                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(48, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(96, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(96, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(192, (3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))
    return model


def build_mobilenetv2_model():
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    base.trainable = False
    model = models.Sequential(name="MobileNetV2_PH2")
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(base)
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))
    return model, base


def train(data_dir, save_path='cnn_mejorado_ph2.weights.h5', backbone='cnn', finetune=False):
    if backbone == 'mobilenet':
        model, base = build_mobilenetv2_model()
        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        total = sum(tf.keras.backend.count_params(w) for w in model.weights)
        print(f"[INFO] MobileNetV2 — entrenables: {trainable:,} / {total:,} totales")
    else:
        print("[INFO] Construyendo modelo CNN Mejorado (desde cero)...")
        model = build_cnn_mejorado_model()
        base = None

    train_dir = os.path.join(data_dir, 'train_dir')
    test_dir = os.path.join(data_dir, 'test_dir')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"[ERROR] No se encuentran {train_dir} o {test_dir}")
        print(f"[INFO] Usa --mode prepare_data para preparar PH2")
        print(f"[INFO]   python CNNMejorado_PH2.py --mode prepare_data")
        return

    def normalize_fixed(x):
        return (x - 0.5) / 0.5

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=normalize_fixed,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        shear_range=10,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=normalize_fixed)

    train_batches = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True,
        class_mode='sparse', classes=CLASS_NAMES)
    test_batches = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False,
        class_mode='sparse', classes=CLASS_NAMES)

    steps_per_epoch = max(1, train_batches.samples // BATCH_SIZE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    warmup_epochs = EPOCHS // 2
    if backbone == 'mobilenet' and base is not None:
        base.trainable = False

    lr_schedule_1 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0005,
        decay_steps=steps_per_epoch * warmup_epochs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_1),
        loss=loss_fn, metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path, monitor='val_accuracy',
        save_best_only=True, save_weights_only=True)
    print(f"[INFO] Fase 1 — warmup ({warmup_epochs} épocas completas)...")
    h1 = model.fit(
        train_batches, steps_per_epoch=steps_per_epoch, epochs=warmup_epochs,
        verbose=1, validation_data=test_batches,
        callbacks=[checkpoint], class_weight=CLASS_WEIGHTS)

    is_finetune = (backbone == 'mobilenet' and finetune and base is not None)
    if is_finetune:
        for layer in base.layers[-30:]:
            layer.trainable = True
        trainable_ft = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        print(f"[INFO] Fase 2 — fine-tuning ({EPOCHS - warmup_epochs} épocas, "
              f"{trainable_ft:,} params entrenables, LR=1e-5)...")

        ft_epochs = EPOCHS - warmup_epochs
        lr_schedule_2 = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.00001,
            decay_steps=steps_per_epoch * ft_epochs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_2),
            loss=loss_fn, metrics=['accuracy'])

        h2 = model.fit(
            train_batches, steps_per_epoch=steps_per_epoch, epochs=ft_epochs,
            verbose=1, validation_data=test_batches,
            callbacks=[checkpoint], class_weight=CLASS_WEIGHTS)

        for k in h1.history:
            h1.history[k] = h1.history[k] + h2.history[k]
    history = h1

    model.load_weights(save_path)
    test_batches.reset()
    y_pred_proba = model.predict(test_batches, verbose=0)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    y_true = test_batches.classes[:len(y_pred_proba)]

    model_name = f"{'MobileNetV2' if backbone == 'mobilenet' else 'CNN_Mejorado'}_PH2"
    evaluate_full(
        y_true, y_pred_class, y_pred_proba,
        history=history, model_name=model_name,
        class_names=CLASS_NAMES, save_prefix='ph2_')

    return model, history


def tta_predict(model, image, datagen, n=TTA_N):
    preds = []
    for _ in range(n):
        aug_img = datagen.random_transform(image)
        pred = model.predict(np.expand_dims(aug_img, axis=0), verbose=0)
        preds.append(pred)
    return np.mean(preds, axis=0)


def predict_image(image_path, model_weights='cnn_mejorado_ph2.weights.h5', use_tta=True, backbone='cnn'):
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encuentra: {image_path}")
        return None, None

    if backbone == 'mobilenet':
        model = build_mobilenetv2_model()[0]
    else:
        model = build_cnn_mejorado_model()
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0005, decay_steps=50000)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    if os.path.exists(model_weights):
        model.load_weights(model_weights)
        print(f"[INFO] Pesos cargados: {model_weights}")
    else:
        print(f"\n{'!' * 60}")
        print(f"  NO HAY PESOS ENTRENADOS: '{model_weights}' no existe")
        print(f"  Las predicciones serán aleatorias (sin entrenamiento).")
        print(f"{'!' * 60}")
        print(f"\n  Para entrenar el modelo:")
        print(f"  1. Preparar datos:  python CNNMejorado_PH2.py --mode prepare_data")
        print(f"  2. Entrenar:        python CNNMejorado_PH2.py --mode train --data_dir ./PH2")
        print(f"  3. Predecir:        python CNNMejorado_PH2.py --mode predict --image ruta/imagen.jpg\n")

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_normalized = (img_array - 0.5) / 0.5

    if use_tta:
        datagen_tta = ImageDataGenerator(
            rotation_range=20, width_shift_range=0.15,
            height_shift_range=0.15, horizontal_flip=True,
            zoom_range=0.1, brightness_range=[0.8, 1.2])
        predictions = tta_predict(model, img_normalized, datagen_tta, n=TTA_N)[0]
        metodo = f"TTA (promedio de {TTA_N} aumentaciones)"
    else:
        predictions = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0]
        metodo = "predicción única"

    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_class = CLASS_NAMES[predicted_idx]

    print("\n" + "=" * 60)
    print("CNN MEJORADO - CLASIFICACIÓN DE LESIÓN CUTÁNEA (PH2)")
    print(f"Método: {metodo}")
    print("=" * 60)
    sorted_idx = np.argsort(predictions)[::-1]
    for i, idx in enumerate(sorted_idx):
        bar = "█" * int(predictions[idx] * 25) + "░" * (25 - int(predictions[idx] * 25))
        print(f"  {i+1}. {CLASS_NAMES[idx]:<8} {predictions[idx]:.1%} {bar}")

    print(f"\n  ▶ Diagnóstico: {predicted_class.upper()} ({confidence:.1%})")
    print(f"  ▶ {CLASS_DESCRIPTIONS[predicted_class]}")
    print(f"  ▶ Severidad: {CLASS_SEVERITY[predicted_class]}")
    print(f"\n  {'─' * 60}")
    print(f"  ⚠  Referencia informativa. Consulte a un dermatólogo.")
    print(f"  {'─' * 60}\n")

    plt.figure(figsize=(4, 4))
    plt.imshow(img_resized)
    plt.title(f"Predicción: {predicted_class}\n{confidence:.1%}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('cnn_mejorado_ph2_prediction.png', dpi=150)
    plt.show()

    return predicted_class, confidence


def interactive_mode(model_weights='cnn_mejorado_ph2.weights.h5', backbone='cnn'):
    print("=" * 60)
    if backbone == 'mobilenet':
        print("  ASISTENTE DE LESIONES CUTÁNEAS (PH2) - MobileNetV2")
    else:
        print("  CNN MEJORADO - ASISTENTE DE LESIONES CUTÁNEAS (PH2)")
    print("=" * 60)
    print("\nComandos:")
    print("  predict <ruta>        - Clasificar (con TTA)")
    print("  predict_simple <ruta> - Clasificar (sin TTA)")
    print("  classes               - Mostrar clases")
    print("  help                  - Ayuda")
    print("  quit                  - Salir\n")

    while True:
        try:
            cmd = input(">>> ").strip().split()
            if not cmd:
                continue
            if cmd[0] in ('quit', 'exit', 'q'):
                print("¡Hasta luego!")
                break
            elif cmd[0] == 'classes':
                print()
                for c in CLASS_NAMES:
                    print(f"  {c:<15} - {CLASS_DESCRIPTIONS[c]}")
                print()
            elif cmd[0] == 'help':
                print("\n  predict <ruta>        - Clasificar con TTA")
                print("  predict_simple <ruta>  - Clasificar sin TTA")
                print("  classes               - Ver clases")
                print("  quit                  - Salir\n")
            elif cmd[0] == 'predict':
                if len(cmd) < 2:
                    print("  Error: falta ruta")
                    continue
                predict_image(cmd[1], model_weights, use_tta=True, backbone=backbone)
            elif cmd[0] == 'predict_simple':
                if len(cmd) < 2:
                    print("  Error: falta ruta")
                    continue
                predict_image(cmd[1], model_weights, use_tta=False, backbone=backbone)
            else:
                print(f"  Desconocido: {cmd[0]}")
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CNN Mejorado - Clasificador de Lesiones Cutáneas (PH2)')
    parser.add_argument('--mode', choices=['train', 'predict', 'interactive', 'prepare_data'],
                        default='interactive')
    parser.add_argument('--data_dir', default='./PH2',
                        help='Directorio con train_dir/ y test_dir/')
    parser.add_argument('--image', default=None, help='Ruta de imagen a clasificar')
    parser.add_argument('--weights', default='cnn_mejorado_ph2.weights.h5',
                        help='Archivo de pesos')
    parser.add_argument('--backbone', choices=['cnn', 'mobilenet'], default='cnn',
                        help='Arquitectura: cnn (desde cero) o mobilenet (transfer learning)')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tuning (solo con --backbone mobilenet)')
    parser.add_argument('--no_tta', action='store_true',
                        help='Desactivar TTA en modo predict')
    args = parser.parse_args()

    if args.mode == 'prepare_data':
        download_and_prepare_ph2(args.data_dir)
    elif args.mode == 'train':
        train(args.data_dir, args.weights, backbone=args.backbone, finetune=args.finetune)
    elif args.mode == 'predict':
        if args.image is None:
            print("Error: Modo predict requiere --image")
            sys.exit(1)
        predict_image(args.image, args.weights, use_tta=not args.no_tta, backbone=args.backbone)
    elif args.mode == 'interactive':
        interactive_mode(args.weights, backbone=args.backbone)