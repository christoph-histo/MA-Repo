import os
import shutil
import pandas as pd
import Pretrained_Architectures as models
import gc
import torch


def clean_all():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

base_path = "/your/data/root"
csv_directory = base_path + "/fold_plan"

csv_file = "fold_plan.csv"
# Load the table
df = pd.read_csv("fold_plan.csv")  

# Iterate over all folds
for current_fold in df["fold"].unique():

    # Filter for current fold
    df_fold = df[df["fold"] == current_fold]

    for _, row in df_fold.iterrows():
        src = os.path.join(base_path, row["origin_folder"], row["filename"])
        dst = os.path.join(base_path, row["target_folder"], row["filename"])

        # Make sure the destination folder exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        print(f"Moving {src} -> {dst}")
        shutil.move(src, dst)

        models.Classification_3D_SwinTransformer_setup(mode="train", augmentation="tripath")
        clean_all()
        models.Classification_3D_SwinTransformer_setup(mode="eval", augmentation="tripath")
        clean_all()
