import os
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# dataset class for delivering patches of wsi images along with their names, wsi name, index of patch and its corresponding coordinaes on the wsi image. 
class TumorImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, resize_size=(224, 224), transform=True):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.resize_size = resize_size
        self.transform = transform
        
        # Load CSV and process labels
        self.metadata = pd.read_csv(csv_file)

        # Generate the transformation pipeline
        self.transforms = self.generate_transform(self.resize_size)

    def generate_transform(self, resize_size):
        """Generate transformation pipeline for preprocessing image data."""
        transform_list = []

        # Convert image to tensor
        transform_list.append(self.contiguous_tensor)

        # Resize image if size is provided
        if resize_size:
            transform_list.append(transforms.Resize(resize_size))

        # Rescale image to range [0, 1]
        transform_list.append(self.rescale_tensor)

        # Normalize image (optional)
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transform_list)

    def contiguous_tensor(self, image):
        """Convert image to tensor and ensure contiguous memory layout."""
        # Ensure the image is in (H, W, C) format
        image = np.array(image)  # Convert PIL image to numpy array
        if image.ndim == 2:  # For grayscale images, add the channel dimension
            image = np.expand_dims(image, axis=-1)
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        return torch.from_numpy(image).contiguous()

    def rescale_tensor(self, tensor):
        """Rescale tensor to range [0, 1]."""
        return tensor.to(dtype=torch.get_default_dtype()).div(255)

    def __getitem__(self, idx):
        sample_row = self.metadata.iloc[idx]

        wsi_name = sample_row['WSI']
        patch_index = sample_row['Index']
        row = sample_row['Row']
        column = sample_row['Column']
        
        label = int(sample_row['Binary_Label'])  # ✅ use Binary_Label instead of Label
        alt_label = -1 if label == 0 else 1      # ✅ derive AltLabel manually

        # gradcam_weight = sample_row['Densenet_Gradcam_Weight']
        # gradcam_importance = sample_row['Densenet_Gradcam_Saliency_Importance']

        # Build filename and path
        patch_name = f"PS{wsi_name}_{patch_index}_{row}_{column}.tif"
        patch_path = os.path.join(self.root_dir, wsi_name, patch_name)

        # Load image
        image = Image.open(patch_path).convert('RGB')
        if self.transform:
            image = self.transforms(image)

        return image, (torch.tensor(label, dtype=torch.long), torch.tensor(alt_label, dtype=torch.long)), patch_name, (wsi_name, patch_index, (row,column))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata)


class Tumor_Image_Patch_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, resize_size=(224, 224), transform=True):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.resize_size = resize_size
        self.transform = transform
        
        # Load CSV and process labels
        self.metadata = pd.read_csv(csv_file)

        # Generate the transformation pipeline
        self.transforms = self.generate_transform(self.resize_size)

    def generate_transform(self, resize_size):
        """Generate transformation pipeline for preprocessing image data."""
        transform_list = []

        # Convert image to tensor
        transform_list.append(self.contiguous_tensor)

        # Resize image if size is provided
        if resize_size:
            transform_list.append(transforms.Resize(resize_size))

        # Rescale image to range [0, 1]
        transform_list.append(self.rescale_tensor)

        # Normalize image (optional)
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transform_list)

    def contiguous_tensor(self, image):
        """Convert image to tensor and ensure contiguous memory layout."""
        # Ensure the image is in (H, W, C) format
        image = np.array(image)  # Convert PIL image to numpy array
        if image.ndim == 2:  # For grayscale images, add the channel dimension
            image = np.expand_dims(image, axis=-1)
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        return torch.from_numpy(image).contiguous()

    def rescale_tensor(self, tensor):
        """Rescale tensor to range [0, 1]."""
        return tensor.to(dtype=torch.get_default_dtype()).div(255)

    def __getitem__(self, idx):
        sample_row = self.metadata.iloc[idx]

        wsi_name = sample_row['WSI']
        patch_index = sample_row['Index']
        row = sample_row['Row']
        column = sample_row['Column']
        
        label = int(sample_row['Binary_Label'])  # ✅ use Binary_Label instead of Label
        alt_label = -1 if label == 0 else 1      # ✅ derive AltLabel manually

        # Build filename and path
        patch_name = f"PS{wsi_name}_{patch_index}_{row}_{column}.tif"
        patch_path = os.path.join(self.root_dir, wsi_name, patch_name)

        # Load image
        image = Image.open(patch_path).convert('RGB')
        if self.transform:
            image = self.transforms(image)

        return image, (torch.tensor(label, dtype=torch.long), torch.tensor(alt_label, dtype=torch.long)), patch_name, (wsi_name, patch_index, (row,column))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata)
    

class SyntheticTumorImageDataset(Dataset):
    def __init__(
        self,
        benign_dir: str = None,
        malignant_dir: str = None,
        mode: str = "both",               # "benign", "malignant", or "both"
        row_range: tuple[int,int] = (0, 10000),
        col_range: tuple[int,int] = (0, 10000),
        weight_range: tuple[float,float] = (0.0, 1.0),
        resize_size: tuple[int,int] = (224, 224),
        transform: bool = True,
        seed: int = 42,
    ):
        assert mode in ("benign", "malignant", "both")
        self.row_range = row_range
        self.col_range = col_range
        self.weight_range = weight_range
        self.resize_size = resize_size
        self.transform = transform

        # 1) Gather (path, label, alt_label, sample_type) tuples
        self.samples = []
        if mode in ("benign", "both") and benign_dir:
            for fname in sorted(os.listdir(benign_dir)):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                    p = os.path.join(benign_dir, fname)
                    self.samples.append((p, 0, -1, "benign"))
        if mode in ("malignant", "both") and malignant_dir:
            for fname in sorted(os.listdir(malignant_dir)):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                    p = os.path.join(malignant_dir, fname)
                    self.samples.append((p, 1, +1, "malignant"))
        if not self.samples:
            raise ValueError(f"No images found for mode={mode}")

        # 2) Shuffle & build metadata DataFrame
        random.seed(seed)
        random.shuffle(self.samples)
        rows = []
        for idx, (_, lbl, alt, sample_type) in enumerate(self.samples):
            row = np.random.randint(self.row_range[0], self.row_range[1]+1)
            col = np.random.randint(self.col_range[0], self.col_range[1]+1)
            rows.append({
                'SampleType': sample_type,
                'Index': idx,
                'Row': row,
                'Column': col,
                'Label': lbl,
                'AltLabel': alt,
            })
        self.metadata = pd.DataFrame(rows)

        # 3) Build torchvision transforms
        tfms = [transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]
                )]
        self.transforms = transforms.Compose(tfms) if transform else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl, alt, sample_type = self.samples[idx]
        meta = self.metadata.iloc[idx]

        # Load + transform
        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)

        coords = (int(meta['Row']), int(meta['Column']))
        # weight = float(meta['Densenet_Gradcam_Weight'])
        fname  = os.path.basename(path)

        return (
            img,
            (torch.tensor(lbl, dtype=torch.long), torch.tensor(alt, dtype=torch.long)),
            fname,
            (sample_type, idx, coords)
        )



