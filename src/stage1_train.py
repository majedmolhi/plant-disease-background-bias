import json, os
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.models import build_baseline, build_effnet_finetune

DATA_ROOT = "/content/plantdisease_split"



OUT = "outputs/stage1"

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT + "/models", exist_ok=True)
os.makedirs(OUT + "/history", exist_ok=True)


def main():
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25, zoom_range=0.2,
        width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=True
    ).flow_from_directory(
        DATA_ROOT + "/train",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical"
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    ).flow_from_directory(
        DATA_ROOT + "/val",
        target_size=(224,224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_gen.num_classes

    # Baseline
    baseline = build_baseline(num_classes)
    baseline.compile(
        optimizer=Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    hist_b = baseline.fit(train_gen, validation_data=val_gen, epochs=15)
    baseline.save(f"{OUT}/models/baseline.h5")

    # EfficientNet
    effnet = build_effnet_finetune(num_classes)
    effnet.compile(
        optimizer=Adam(3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    hist_e = effnet.fit(train_gen, validation_data=val_gen, epochs=15)
    effnet.save(f"{OUT}/models/effnet_finetune.h5")

    json.dump(hist_b.history, open(f"{OUT}/history/baseline.json","w"))
    json.dump(hist_e.history, open(f"{OUT}/history/effnet.json","w"))

if __name__ == "__main__":
    main()
