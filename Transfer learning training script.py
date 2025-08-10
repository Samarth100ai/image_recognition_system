import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
DATA_DIR = 'dataset'  # Expected structure: dataset/train/<class>/*, dataset/val/<class>/*
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save class indices
with open('classes.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# Build model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=outputs)

model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# Optionally unfreeze some layers and fine-tune
for layer in base.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=Adam(LEARNING_RATE/10), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=5, validation_data=val_gen)

# Save model
model_path = os.path.join(MODEL_DIR, 'model.h5')
model.save(model_path)
print(f'Model saved to {model_path}')