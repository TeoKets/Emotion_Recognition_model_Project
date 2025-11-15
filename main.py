import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import os
import json

path = r"/mnt/c/Users/ketse/Downloads/archive/Data"
img_size = (128, 128)  # or (256,256) if you want
batch_size = 32  # adjust based on GPU memory

# Training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=64,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

# Validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=64,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(128, 128, 1)),
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", kernel_initializer="he_normal"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", kernel_initializer="he_normal"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", kernel_initializer="he_normal"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", kernel_initializer="he_normal"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(
            128, (3, 3), padding="same", kernel_initializer="he_normal"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(
            128, (3, 3), padding="same", kernel_initializer="he_normal"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation="softmax"),
    ]
)

model.summary()
print(train_ds.class_names)
print(val_ds.class_names)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)


history = model.fit(
    train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop, tensorboard]
)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

model.save(f"models/emotion_model{datetime.now().strftime('%Y%m%d-%H%M%S')}.keras")
with open(
    f"models/training_history{datetime.now().strftime('%Y%m%d-%H%M%S')}.json", "w"
) as f:
    json.dump(history.history, f)

