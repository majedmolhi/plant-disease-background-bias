import opendatasets as od
import os

DATA_URL = "https://www.kaggle.com/datasets/emmarex/plantdisease"
DATA_ROOT = "/content/plantdisease"

def main():
    if not os.path.exists(DATA_ROOT):
        print("Downloading dataset to Colab (/content)...")
        od.download(DATA_URL, data_dir="/content")
    else:
        print("Dataset already exists. Skipping download.")

    base_dir = os.path.join(DATA_ROOT, "PlantVillage")
    classes = sorted(os.listdir(base_dir))

    print("Dataset ready.")
    print("Number of classes:", len(classes))
    print("Classes:", classes)

if __name__ == "__main__":
    main()
