import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py

class PCamDataset(Dataset):
    """
    Dataset class for the PCam (PatchCamelyon) dataset.
    PCam consists of 96x96 RGB images of histopathology patches.
    Each patch is labeled as either containing a tumor (1) or not (0).
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Directory with the PCam h5 files
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply to the images
        """
        assert mode in ['train', 'val', 'test']
        
        # Define file paths
        self.file_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{mode}_x.h5')
        self.labels_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{mode}_y.h5')
        
        self.transform = transform
        
        # Load data references (actual data will be loaded on-the-fly)
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = f['x'].shape[0]
        
    def __getitem__(self, index):
        # Load image
        with h5py.File(self.file_path, 'r') as f:
            image = f['x'][index]
        
        # Load label
        with h5py.File(self.labels_path, 'r') as f:
            label = f['y'][index][0][0].astype(np.int64)  # Binary label
        
        # Convert to PIL image for transforms
        image = Image.fromarray(image)
        
        # Verify image dimensions before transforming
        if np.array(image).shape[:2] != (96, 96):
            print(f"Warning: Found image with unexpected dimensions: {np.array(image).shape}")
            
        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default conversion to tensor if no transform provided
            # Add padding to make 97x97 as required by the model
            img_array = np.array(image, dtype=np.float32)
            padded = np.pad(img_array, ((0, 1), (0, 1), (0, 0)), mode='constant')
            image = torch.from_numpy(padded.transpose((2, 0, 1)) / 255.0)
        
        return image, label
    
    def __len__(self):
        return self.num_samples


# Alternative PyTorch Dataset implementation that preloads data into memory for faster training
class PCamDatasetInMemory(Dataset):
    """
    Memory-efficient version of PCamDataset that loads the entire dataset into memory.
    Useful for faster training when dataset fits in memory.
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Directory with the PCam h5 files
            mode: 'train', 'val', or 'test'
            transform: Optional transforms to apply to the images
        """
        assert mode in ['train', 'val', 'test']
        
        # Define file paths
        file_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{mode}_x.h5')
        labels_path = os.path.join(root_dir, f'camelyonpatch_level_2_split_{mode}_y.h5')
        
        self.transform = transform
        
        # Load data and labels into memory
        with h5py.File(file_path, 'r') as f:
            self.images = f['x'][()]  # Load entire dataset
        
        with h5py.File(labels_path, 'r') as f:
            self.labels = f['y'][()].reshape(-1).astype(np.int64)  # Flatten labels
        
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        # Get image and label
        image = self.images[index]
        label = self.labels[index]
        
        # Convert to PIL image for transforms
        image = Image.fromarray(image)
        
        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default conversion to tensor if no transform provided
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose((2, 0, 1)) / 255.0)
        
        return image, label
    
    def __len__(self):
        return self.num_samples