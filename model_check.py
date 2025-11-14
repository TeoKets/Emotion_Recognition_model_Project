import tensorflow as tf
import matplotlib.pyplot as plt
import json

model = tf.keras.models.load_model("models/emotion_model.keras")

path = "/mnt/c/Users/ketse/Downloads/archive/Data"

img_size = (128, 128)
batch_size = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",  # same split as before
    seed=64,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

loss, accuracy = model.evaluate(test_ds)
print(f"validation loss: {round(loss, 4)}")
print(f"validation accuracy: {round(accuracy, 4)}")


with open("models/training_history20251112-194247.json", "r") as f:
    history = json.load(f)

# Plot training vs validation loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history["loss"], label="Training Loss")
ax1.plot(history["val_loss"], label="Validation Loss")
ax1.set_title("Model Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(history["accuracy"], label="Training Accuracy")
ax2.plot(history["val_accuracy"], label="Validation Accuracy")
ax2.set_title("Model Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
