from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
from PIL import Image
import io
import torch
from torchvision import transforms as T
import pandas as pd
import os
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from PIL import Image
import h5py
from torch.utils.data import Dataset  # just for Dataset base class

from pathlib import Path
import h5py

#example of patch-level dataset contains patch embedding and etc to guide 
class CancerDataset2D_Embedding(Dataset):
    def __init__(self, csv_file, path_root, embedding_h5_path, p_uncond=0.0, mode="train", val_split=0.1, seed=42, image_size=256):
        self.path_root = Path(path_root)
        self.metadata = pd.read_csv(csv_file)
        self.embedding_h5 = h5py.File(embedding_h5_path, "r")  # Keep file open
        self.p_uncond = p_uncond
        self.image_size = image_size

        # Split train/val if needed
        if mode in ["train", "val"]:
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                self.metadata,
                test_size=val_split,
                stratify=self.metadata["Binary_Label"],
                random_state=seed
            )
            self.metadata = train_df if mode == "train" else val_df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        patch_name = f"PS{row['Sample']}.tif"
        img_path = self.path_root / row["WSI"] / patch_name

        # Load image
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        if img.shape[:2] != (self.image_size, self.image_size):
            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((self.image_size, self.image_size)), dtype=np.float32)
        img = img / 127.5 - 1.0

        # Random flips
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=0).copy()
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1).copy()

        # Load embedding directly using patch name as key
        # full_patch_name = f"{row['WSI']}/{patch_name}"  # matches what you stored in HDF5
        embedding = self.embedding_h5[patch_name][:].astype(np.float32)

        # Optional unconditional embedding
        if np.random.rand() < self.p_uncond:
            embedding = np.zeros_like(embedding)

        return {
            "image": img,
            "feat_patch": embedding,
            "human_label": patch_name   # <-- human-readable patch name
        }
