"""
run_tableII_experiment.py

Optimizing Lung Disease Classification with Transfer Learning (multiple backbones)
Performs Repeated Stratified 5-Fold Cross-Validation (10 repeats -> 50 folds) over full dataset (8100 images).

Dataset layout (relative to DATA_ROOT):
  train/Normal/*.jpg
  train/Pneumonia/*.jpg
  test/Normal/*.jpg
  test/Pneumonia/*.jpg

Outputs:
  - per-model per-fold CSV files: metrics_<model_name>.csv
  - summary_table.csv with mean ± std for each model (Accuracy, Precision, Recall, F1)
  - optional saved best model files (one per fold) in MODEL_SAVE_DIR

Notes:
  - Requires: tensorflow (2.x), scikit-learn, pandas, numpy
  - GPU strongly recommended.
"""

import os
import math
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# -------------------------
# Configuration
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = r"./dataset"   # <-- change this to your dataset folder path on Windows
# The script will look under DATA_ROOT/train and DATA_ROOT/test for Normal/ Pneumonia subfolders
IMG_SIZE = (299, 299)      # 299 works well for Inception/ResNet-Inception/Xception; other models can accept 299 too
BATCH_SIZE = 8             # small because 299x299 images and many folds; increase if you have memory
EPOCHS_HEAD = 6            # train only top head for these many epochs per fold (small default)
EPOCHS_FINETUNE = 2        # optional small fine-tuning after unfreeze
FINE_TUNE = False          # set True to unfreeze top layers and fine-tune (slower)
N_SPLITS = 5
N_REPEATS = 10             # 5-fold * 10 repeats = 50 folds
VERBOSE = 1
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Models to run (keys are friendly names; values are Keras model factory names we handle)
MODEL_NAMES = [
    "ResNet152V2",
    "DenseNet121",
    "VGG16",
    "Xception",
    "MobileNetV3Large",
    "EfficientNetV2L",
    "InceptionV3",
    "NASNetMobile",
    "InceptionResNetV2"   # Proposed
]

TARGET_LABELS = ["Normal", "Pneumonia"]
LABEL_MAP = {TARGET_LABELS[0]: 0, TARGET_LABELS[1]: 1}

# -------------------------
# Utility: build list of all images and labels
# -------------------------
def build_dataframe_from_dataset(data_root):
    rows = []
    for split in ["train", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for label in TARGET_LABELS:
            d = os.path.join(split_dir, label)
            if not os.path.isdir(d):
                continue
            imgs = glob(os.path.join(d, "*"))
            for p in imgs:
                rows.append({"filepath": os.path.abspath(p), "label": label})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle
    return df

# -------------------------
# Utility: model factory
# -------------------------
def get_backbone_model(name, input_shape=(299,299,3), dropout_rate=0.5):
    # Import models lazily to avoid heavy import overhead when not needed
    from tensorflow.keras import applications

    name_u = name.lower()
    base = None
    try:
        if name == "ResNet152V2":
            base = applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "DenseNet121":
            base = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "VGG16":
            base = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "Xception":
            base = applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "MobileNetV3Large":
            # Some TF versions use tf.keras.applications.MobileNetV3Large
            base = applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "EfficientNetV2L":
            # EfficientNetV2L exists in TF >= 2.9
            base = applications.EfficientNetV2L(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "InceptionV3":
            base = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "NASNetMobile":
            base = applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
        elif name == "InceptionResNetV2":
            base = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"Unknown backbone: {name}")
    except Exception as e:
        raise ImportError(f"Failed to construct base model '{name}': {e}")

    # Build top head
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model, base

# -------------------------
# Data generator factory
# -------------------------
def get_generators_from_df(df_train, df_val, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        color_mode='rgb',
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False
    )
    return train_gen, val_gen

# -------------------------
# Prediction helper to get preds aligned with generator samples
# -------------------------
def predict_on_generator(model, generator):
    # iterate generator to get exact preds and trues
    generator.reset()
    steps = int(math.ceil(generator.samples / generator.batch_size))
    y_pred_probs = []
    y_true = []
    for _ in range(steps):
        Xb, yb = next(generator)
        pb = model.predict_on_batch(Xb)
        y_pred_probs.extend(pb.reshape(-1).tolist())
        y_true.extend(yb.reshape(-1).tolist())
    # truncate to generator.samples (last batch may pad)
    y_pred_probs = np.array(y_pred_probs)[:generator.samples]
    y_true = np.array(y_true)[:generator.samples]
    y_pred_bin = (y_pred_probs >= 0.5).astype(int)
    return y_true.astype(int), y_pred_bin.astype(int), y_pred_probs[:generator.samples]

# -------------------------
# Main experiment loop
# -------------------------
def run_experiments(df_all):
    X = df_all['filepath'].values
    y_text = df_all['label'].values
    y_numeric = np.array([LABEL_MAP[l] for l in y_text])

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    summary_rows = []

    for model_name in MODEL_NAMES:
        print("\n" + "="*80)
        print(f"Starting experiments for model: {model_name}  ({datetime.now().isoformat()})")
        print("="*80)
        fold_metrics = []
        fold_idx = 0

        # iterate folds
        for train_idx, val_idx in rskf.split(X, y_numeric):
            fold_idx += 1
            print(f"\nModel {model_name} — Fold {fold_idx} / {N_SPLITS*N_REPEATS}")

            df_train = df_all.iloc[train_idx].reset_index(drop=True)
            df_val = df_all.iloc[val_idx].reset_index(drop=True)

            # class weights
            y_train_num = np.array([LABEL_MAP[l] for l in df_train['label'].values])
            classes = np.unique(y_train_num)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_num)
            class_weight_dict = {int(classes[i]): float(cw[i]) for i in range(len(classes))}
            print("Class weights:", class_weight_dict)

            # generators
            train_gen, val_gen = get_generators_from_df(df_train, df_val, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

            # model build
            tf.keras.backend.clear_session()
            try:
                model, base_model = get_backbone_model(model_name, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
            except Exception as e:
                print(f"ERROR building model {model_name}: {e}")
                break

            # callbacks
            model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_fold{fold_idx}.h5")
            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
                EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
                ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=0)
            ]

            steps_per_epoch = max(1, int(math.ceil(train_gen.samples / train_gen.batch_size)))
            validation_steps = max(1, int(math.ceil(val_gen.samples / val_gen.batch_size)))

            # Train head
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=EPOCHS_HEAD,
                validation_data=val_gen,
                validation_steps=validation_steps,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=VERBOSE
            )

            # Optional fine-tune (unfreeze top layers)
            if FINE_TUNE:
                # Unfreeze last N layers of base model
                unfreeze_at = -50  # unfreeze last 50 layers (tweakable)
                for layer in base_model.layers[unfreeze_at:]:
                    layer.trainable = True
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
                print("Starting fine-tuning...")
                model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=EPOCHS_FINETUNE,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    verbose=VERBOSE
                )

            # Evaluate on val generator
            y_true, y_pred_bin, y_pred_prob = predict_on_generator(model, val_gen)
            acc = accuracy_score(y_true, y_pred_bin)
            prec = precision_score(y_true, y_pred_bin, zero_division=0)
            rec = recall_score(y_true, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)

            print(f"Fold results -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            fold_metrics.append({
                "fold": fold_idx,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "n_val_samples": int(len(y_true))
            })

            # Save per-fold metrics as we go (append)
            pd.DataFrame(fold_metrics).to_csv(f"metrics_{model_name}.csv", index=False)

            # free memory occasionally
            tf.keras.backend.clear_session()

        # aggregate for model
        df_metrics = pd.DataFrame(fold_metrics)
        if df_metrics.shape[0] == 0:
            print(f"No metrics for model {model_name}, skipping summary.")
            continue
        mean_acc = df_metrics['accuracy'].mean()
        std_acc  = df_metrics['accuracy'].std()
        mean_prec = df_metrics['precision'].mean()
        std_prec  = df_metrics['precision'].std()
        mean_rec = df_metrics['recall'].mean()
        std_rec  = df_metrics['recall'].std()
        mean_f1 = df_metrics['f1'].mean()
        std_f1  = df_metrics['f1'].std()

        summary_rows.append({
            "model": model_name,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "precision_mean": mean_prec,
            "precision_std": std_prec,
            "recall_mean": mean_rec,
            "recall_std": std_rec,
            "f1_mean": mean_f1,
            "f1_std": std_f1
        })

        # final save for model
        df_metrics.to_csv(f"metrics_{model_name}.csv", index=False)
        print(f"Saved metrics_{model_name}.csv with {len(df_metrics)} fold rows.")

    # summary table
    df_summary = pd.DataFrame(summary_rows)
    # create nice display columns like "mean ± std" for each metric
    def fmt_mean_std(m, s):
        return f"{m*100:.2f} ± {s*100:.2f}"  # percent format

    if not df_summary.empty:
        df_summary['Accuracy'] = df_summary.apply(lambda r: fmt_mean_std(r['accuracy_mean'], r['accuracy_std']), axis=1)
        df_summary['Precision'] = df_summary.apply(lambda r: fmt_mean_std(r['precision_mean'], r['precision_std']), axis=1)
        df_summary['Recall'] = df_summary.apply(lambda r: fmt_mean_std(r['recall_mean'], r['recall_std']), axis=1)
        df_summary['F1-Score'] = df_summary.apply(lambda r: fmt_mean_std(r['f1_mean'], r['f1_std']), axis=1)

        df_summary[['model','Accuracy','Precision','Recall','F1-Score']].to_csv("summary_table.csv", index=False)
        print("\nSaved summary_table.csv")
        print(df_summary[['model','Accuracy','Precision','Recall','F1-Score']].to_string(index=False))
    else:
        print("No summary produced (no models produced metrics).")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    print(f"Preparing dataset from: {DATA_ROOT}")
    df_all = build_dataframe_from_dataset(DATA_ROOT)
    print(f"Total samples found: {len(df_all)}")
    print(df_all['label'].value_counts())

    if len(df_all) == 0:
        print("No images found. Please check DATA_ROOT path and folder structure.")
        raise SystemExit(1)

    # confirm counts match expected (optional)
    # print(df_all.groupby('label').size())

    # Run experiments
    run_experiments(df_all)

    print("Experiment finished.")
