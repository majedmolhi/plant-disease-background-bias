# ============================================================
# Stage 2: Background Bias Elimination
# Phase A (Warm-start) + Phase B (Mixed refinement)
# ============================================================

import os, json, random, cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# Paths & constants
# ============================================================

DATA_ROOT   = "/content/plantdisease_split"
STAGE1_PATH = "outputs/stage1/models/effnet_finetune.h5"
OUT_DIR     = "outputs/stage2"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
rng = random.Random(SEED)

# ============================================================
# Import preprocessing functions
# ============================================================

from src.utils import (
    leaf_only_black,
    get_leaf_mask
)

def train_randbg(rgb_img):
    mask = get_leaf_mask(rgb_img, dilate_px=1)
    bg = np.random.randint(0, 256, rgb_img.shape, dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (21,21), 7)
    mixed = np.where(mask[...,None] > 0, rgb_img, bg)
    mixed = cv2.resize(mixed, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return preprocess_input(mixed.astype(np.float32))

def preprocess_original(rgb_img):
    img = cv2.resize(rgb_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return preprocess_input(img.astype(np.float32))

# ============================================================
# Build dataset index
# ============================================================

def build_index(split):
    classes = sorted(os.listdir(os.path.join(DATA_ROOT, split)))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    files = []

    for c in classes:
        cdir = os.path.join(DATA_ROOT, split, c)
        for f in os.listdir(cdir):
            files.append((os.path.join(cdir, f), class_to_idx[c]))

    return files, classes

train_files, class_names = build_index("train")
val_files, _             = build_index("val")
NUM_CLASSES = len(class_names)

# ============================================================
# Class weights → used INSIDE generators
# ============================================================

y_train = [y for _, y in train_files]
cw = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)
class_weights = {i: float(w) for i,w in enumerate(cw)}

# ============================================================
# Phase A generator (ORIGINAL images only)
# ============================================================

def original_generator(files, batch_size):
    while True:
        rng.shuffle(files)
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]

            X, y, sw = [], [], []
            for path, label in batch:
                rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                X.append(preprocess_original(rgb))
                y.append(label)
                sw.append(class_weights[label])

            yield (
                np.array(X),
                tf.keras.utils.to_categorical(y, NUM_CLASSES),
                np.array(sw)
            )

# ============================================================
# Phase B generator (1/3 original + 1/3 leaf-only + 1/3 rand-bg)
# ============================================================

def mixed_generator(files, batch_size):
    third = batch_size // 3

    while True:
        rng.shuffle(files)
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            X, y, sw = [], [], []

            for j, (path, label) in enumerate(batch):
                rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

                if j < third:
                    img = preprocess_original(rgb)
                elif j < 2*third:
                    img = leaf_only_black(rgb)
                else:
                    img = train_randbg(rgb)

                X.append(img)
                y.append(label)
                sw.append(class_weights[label])

            yield (
                np.array(X),
                tf.keras.utils.to_categorical(y, NUM_CLASSES),
                np.array(sw)
            )

# ============================================================
# Load Stage-1 model
# ============================================================

model = load_model(STAGE1_PATH, compile=False)

if model.layers[-1].units != NUM_CLASSES:
    raise ValueError("Number of classes mismatch")

# ============================================================
# Phase A — Warm-start
# ============================================================

loss_fn = CategoricalCrossentropy(label_smoothing=0.05)
optA = AdamW(learning_rate=3e-4, weight_decay=1e-5)

for l in model.layers:
    l.trainable = False
for l in model.layers[-3:]:
    l.trainable = True

model.compile(optimizer=optA, loss=loss_fn, metrics=["accuracy"])

histA = model.fit(
    original_generator(train_files, BATCH_SIZE),
    validation_data=original_generator(val_files, BATCH_SIZE),
    steps_per_epoch=len(train_files)//BATCH_SIZE,
    validation_steps=len(val_files)//BATCH_SIZE,
    epochs=2,
    callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2)],
    verbose=1
)

model.save_weights(os.path.join(OUT_DIR, "M_A.weights.h5"))

# ============================================================
# Phase B — Mixed refinement
# ============================================================

optB = AdamW(learning_rate=1e-4, weight_decay=1e-5)

for l in model.layers[:-40]:
    l.trainable = False
for l in model.layers[-40:]:
    l.trainable = True

model.compile(optimizer=optB, loss=loss_fn, metrics=["accuracy"])

ckpt = os.path.join(OUT_DIR, "M_final_best.weights.h5")

histB = model.fit(
    mixed_generator(train_files, BATCH_SIZE),
    validation_data=original_generator(val_files, BATCH_SIZE),
    steps_per_epoch=len(train_files)//BATCH_SIZE,
    validation_steps=len(val_files)//BATCH_SIZE,
    epochs=8,
    callbacks=[
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor="val_accuracy",
                        save_best_only=True, save_weights_only=True)
    ],
    verbose=1
)

# ============================================================
# Save final outputs
# ============================================================

model.save(os.path.join(OUT_DIR, "M_final.keras"))

with open(os.path.join(OUT_DIR, "history_stage2.json"), "w") as f:
    json.dump({"phaseA": histA.history, "phaseB": histB.history}, f)

print("Stage-2 training completed successfully.")
