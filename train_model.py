import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_SAVE_PATH = "plant_disease_model.keras"
CLASS_INDICES_PATH = "class_indices.json"

# Set random seed for reproducibility
tf.random.set_seed(42)

# ==========================================
# 2. LOAD DATASETS
# ==========================================
print("--> Loading training and validation datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Extract and save class names mapping
class_names = train_dataset.class_names
class_indices = {i: name for i, name in enumerate(class_names)}
num_classes = len(class_names)

print(f"--> Detected {num_classes} classes: {class_names}")

# Save class index mapping to JSON for Flask app to use later
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f, indent=4)
print(f"--> Saved class mapping to '{CLASS_INDICES_PATH}'")

# Optimize input pipeline performance with prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. DATA AUGMENTATION & MODEL ARCHITECTURE
# ==========================================
print("--> Building MobileNetV2 model architecture...")

# Data augmentation layer (applied during training only)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Load pre-trained MobileNetV2 without the top classification layer
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model weights for initial transfer learning
base_model.trainable = False

# Construct final custom classification head
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
# MobileNetV2 expects pixel values in [-1, 1] range
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 4. TRAINING CALLBACKS
# ==========================================
callbacks = [
    # Save the best model based on validation loss
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    # Stop training early if validation loss stops improving
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Lower learning rate when learning plateau is hit
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# 5. MODEL TRAINING
# ==========================================
print("--> Starting model training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# 6. EVALUATION & METRICS PLOTTING
# ==========================================
print("\n--> Plotting training history...")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss", marker="o")
plt.plot(epochs_range, val_loss, label="Validation Loss", marker="o")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()

print(f"\n[SUCCESS] Model training complete!")
print(f"--> Trained weights saved to: '{MODEL_SAVE_PATH}'")
print(f"--> Class labels map saved to: '{CLASS_INDICES_PATH}'")
print(f"--> Metrics plot saved to: 'training_metrics.png'")