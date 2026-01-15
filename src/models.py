from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import EfficientNetB0

def build_baseline(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224,224,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_effnet_finetune(num_classes):
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    for l in base.layers[:-40]:
        l.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(base.input, out)
