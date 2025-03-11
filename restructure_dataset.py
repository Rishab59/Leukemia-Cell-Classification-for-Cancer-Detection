import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
print("Initializing Kaggle API")
api = KaggleApi()
api.authenticate()

# Download dataset
dataset_name = "andrewmvd/leukemia-classification"
print(f"Downloading dataset: \"{dataset_name}\" ...")
api.dataset_download_files(dataset_name, path = "./", unzip = True)

# Define paths
source_path = "./C-NMC_Leukemia/training_data"
destination_path = "./LeukemiaCellClassification"

# Create new folder structure
os.makedirs(destination_path, exist_ok = True)

# Move images to new structure
print("Restructuring dataset...")
for fold in ["fold_0", "fold_1", "fold_2"]:
    for category in ["all", "hem"]:
        src = os.path.join(source_path, fold, category)
        dest = os.path.join(destination_path, category)

        os.makedirs(dest, exist_ok = True)

        for file in os.listdir(src):
            os.rename(os.path.join(src, file), os.path.join(dest, file))

print("✅ Dataset restructured successfully!")
print("\nScript Closed...\nThank You !!!")