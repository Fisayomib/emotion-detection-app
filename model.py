# model.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

print("Looking for dataset at:", DATA_DIR)     # debug print
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Missing folder: {DATA_DIR}. "
                            "Create dataset/ with class subfolders.")

# model.py
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "emotion_model.h5"
LABELS_PATH = BASE_DIR / "labels.json"

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
EPOCHS = 10

print("Looking for dataset at:", DATA_DIR)
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Missing folder: {DATA_DIR}")

# 1) Create RAW datasets first (do NOT map/prefetch yet)
train_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# 2) Grab class_names NOW (before mapping/prefetching)
class_names = train_raw.class_names  # ["Angry", "Disgust", ...]
num_classes = len(class_names)
print("Classes:", class_names)

# 3) Now do normalization + performance pipeline
AUTOTUNE = tf.data.AUTOTUNE
norm = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
train_ds = train_raw.map(norm).cache().prefetch(AUTOTUNE)
val_ds   = val_raw.map(norm).cache().prefetch(AUTOTUNE)

# 4) Define a simple CNN
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 5) Save model + labels
model.save(MODEL_PATH)
with open(LABELS_PATH, "w") as f:
    json.dump(class_names, f)

print(f"Saved model to {MODEL_PATH}")
print(f"Saved labels to {LABELS_PATH}")
