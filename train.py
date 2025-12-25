import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "data.csv"
BASE_IMAGE_DIR = "datasets"

IMG_H, IMG_W = 224, 224
EPOCHS = 10
BATCH_SIZE = 32
VAL_SPLIT = 0.2

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(CSV_PATH, header=None, names=["img1", "img2", "label"])

def clean_path(p):
    return str(p).strip().replace("\\", "/")

df["img1"] = df["img1"].apply(clean_path)
df["img2"] = df["img2"].apply(clean_path)
df["label"] = df["label"].astype(int)

df = df[df.apply(
    lambda r: os.path.exists(os.path.join(BASE_IMAGE_DIR, r["img1"])) and
              os.path.exists(os.path.join(BASE_IMAGE_DIR, r["img2"])),
    axis=1
)].reset_index(drop=True)

print("Total samples:", len(df))

# -----------------------------
# TRAIN / VAL SPLIT
# -----------------------------
df = df.sample(frac=1).reset_index(drop=True)
val_size = int(len(df) * VAL_SPLIT)

val_df = df[:val_size]
train_df = df[val_size:]

print("Train samples:", len(train_df))
print("Val samples:", len(val_df))

# -----------------------------
# IMAGE LOADER
# -----------------------------
def load_image(rel_path):
    img = cv2.imread(os.path.join(BASE_IMAGE_DIR, rel_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H))
    return img.astype("float32") / 255.0

# -----------------------------
# GENERATOR
# -----------------------------
def pair_generator(dataframe, batch_size):
    while True:
        dataframe = dataframe.sample(frac=1)
        for i in range(0, len(dataframe), batch_size):
            batch = dataframe.iloc[i:i+batch_size]
            X1, X2, y = [], [], []
            for _, row in batch.iterrows():
                X1.append(load_image(row["img1"]))
                X2.append(load_image(row["img2"]))
                y.append(row["label"])
            yield [np.array(X1), np.array(X2)], np.array(y)

# -----------------------------
# MODEL
# -----------------------------
base_model = MobileNet(
    input_shape=(IMG_H, IMG_W, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inp = layers.Input((IMG_H, IMG_W, 3))
x = base_model(inp)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
embedding = Model(inp, x)

a = layers.Input((IMG_H, IMG_W, 3))
b = layers.Input((IMG_H, IMG_W, 3))

fa = embedding(a)
fb = embedding(b)

dist = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([fa, fb])
out = layers.Dense(1, activation="sigmoid")(dist)

model = Model([a, b], out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    pair_generator(train_df, BATCH_SIZE),
    steps_per_epoch=len(train_df)//BATCH_SIZE,
    validation_data=pair_generator(val_df, BATCH_SIZE),
    validation_steps=len(val_df)//BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# GRAPHS
# -----------------------------
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()

print("Training complete")
print("Best model saved as best_model.h5")