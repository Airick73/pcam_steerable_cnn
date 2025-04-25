import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import random
import math

# Define standard transformations for PCam dataset
class TransformBuilder:
    """
    Utility class to build transformations for the PCam dataset.
    """
    @staticmethod
    def build_train_transform():
        """
        Returns a composition of transforms for training data.
        Includes data augmentation techniques to improve model robustness.
        """
        return transforms.Compose([
            # Pad to 97x97 to be consistent with the model's architecture
            transforms.Pad((0, 0, 1, 1), fill=0),  # Add 1 pixel to right and bottom to make 97x97
            
            # Data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            
            # Color jittering for histopathology images
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7, 0.6, 0.7], std=[0.15, 0.15, 0.15])  # Approximate PCam stats
        ])
    
    @staticmethod
    def build_test_transform():
        """
        Returns a composition of transforms for validation and test data.
        No data augmentation is applied.
        """
        return transforms.Compose([
            # Pad to 97x97 to be consistent with the model's architecture
            transforms.Pad((0, 0, 1, 1), fill=0),  # Add 1 pixel to right and bottom to make 97x97
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7, 0.6, 0.7], std=[0.15, 0.15, 0.15])  # Approximate PCam stats
        ])
    
    @staticmethod
    def build_rotation_test_transform(angle=0):
        """
        Returns a transform that rotates an image by a specific angle.
        Useful for testing rotation equivariance.
        
        Args:
            angle: Rotation angle in degrees
        """
        return transforms.Compose([
            transforms.Pad((0, 0, 1, 1), fill=0),  # Add 1 pixel to right and bottom to make 97x97
            transforms.Lambda(lambda img: img.rotate(angle, InterpolationMode.BILINEAR)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7, 0.6, 0.7], std=[0.15, 0.15, 0.15])
        ])


# Custom transform for equalizing histology images (optional)
class HistogramEqualization:
    """
    Apply histogram equalization to histopathology images.
    This can enhance contrast and make tissue structures more visible.
    """
    def __call__(self, img):
        img_np = np.array(img)
        
        # Apply equalization to each channel
        for i in range(3):  # RGB channels
            img_np[:, :, i] = cv2.equalizeHist(img_np[:, :, i])
        
        return Image.fromarray(img_np)