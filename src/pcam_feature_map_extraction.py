import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import os
import sys

# Add the src directory to the path so we can import the model
sys.path.append('/teamspace/studios/this_studio/pcam_steerable_cnn/src/')

# Import the model and required libraries
from models import C8SteerableCNN
from escnn import nn as enn
from escnn import gspaces
import torchvision.transforms.functional as TF

# Configuration
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # checkpoint_path = "/teamspace/studios/this_studio/pcam_steerable_cnn/src/output/pcam_steerable_cnn_20250425_224101/checkpoints/best_model.pth"
    # checkpoint_path = "/teamspace/studios/this_studio/pcam_steerable_cnn/src/output/augmented_training_checkpoint/checkpoints/best_model.pth"
    checkpoint_path = "/teamspace/studios/this_studio/pcam_steerable_cnn/src/output/unaugmented_training_checkpoint/checkpoints/best_model.pth"
    data_path = '/teamspace/studios/this_studio/pcam_steerable_cnn/data/camelyonpatch_level_2_split_test_x.h5'
    sample_idx = 0  # Index of the sample to visualize
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Angles to test (multiples of 45° work best with C8)
    max_channels = 8  # Maximum number of channels to display per layer

config = Config()

# Create and load the model
print("Loading model...")
model = C8SteerableCNN(in_channels=3, n_classes=2)

# Load the checkpoint
checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
if 'model_state_dict' in checkpoint:
    # If the checkpoint is from the Trainer class
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # If it's just the state dict
    model.load_state_dict(checkpoint)

model = model.to(config.device)
model.eval()

# Dictionary to store feature maps
feature_maps = {}

# Helper function to register hooks on escnn modules
def register_hooks(model):
    # Clear any existing hooks
    feature_maps.clear()
    
    # Hook function for escnn modules
    def hook_fn(name):
        def hook(module, input, output):
            # For escnn modules, output is a GeometricTensor
            feature_maps[name] = output
        return hook
    
    # Register hooks on key modules
    handles = []
    
    # Register hooks on specific layers of interest
    handles.append(model.block1.register_forward_hook(hook_fn('block1')))
    handles.append(model.block2.register_forward_hook(hook_fn('block2')))
    handles.append(model.pool1.register_forward_hook(hook_fn('pool1')))
    handles.append(model.block3.register_forward_hook(hook_fn('block3')))
    handles.append(model.block4.register_forward_hook(hook_fn('block4')))
    handles.append(model.pool2.register_forward_hook(hook_fn('pool2')))
    handles.append(model.block5.register_forward_hook(hook_fn('block5')))
    handles.append(model.pool2_5.register_forward_hook(hook_fn('pool2_5')))
    handles.append(model.block6.register_forward_hook(hook_fn('block6')))
    handles.append(model.pool3.register_forward_hook(hook_fn('pool3')))
    handles.append(model.gpool.register_forward_hook(hook_fn('gpool')))
    
    return handles

# Load a test image
print(f"Loading test image from {config.data_path}...")
with h5py.File(config.data_path, 'r') as f:
    # Get a single image
    image = f['x'][config.sample_idx]

# Convert to PIL image for rotation
image_pil = Image.fromarray(image)

# Pad to 97x97 as required by the model
padded_image = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode='constant')
print(f"Padded image shape: {padded_image.shape}")

# Function to preprocess image for the model
def preprocess_image(img_array):
    # Convert to tensor with shape [1, 3, H, W]
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
    
    # Normalize to [0, 1]
    img_tensor = img_tensor / 255.0
    
    # Apply normalization similar to what's used in transforms.py
    # Approximate PCam stats: mean=[0.7, 0.6, 0.7], std=[0.15, 0.15, 0.15]
    mean = torch.tensor([0.7, 0.6, 0.7]).view(1, 3, 1, 1).to(config.device)
    std = torch.tensor([0.15, 0.15, 0.15]).view(1, 3, 1, 1).to(config.device)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.to(config.device)

# Function to rotate image
def rotate_image(img_pil, angle):
    rotated_pil = img_pil.rotate(angle, Image.BILINEAR)
    rotated_array = np.array(rotated_pil)
    # Pad to 97x97
    padded = np.pad(rotated_array, ((0, 1), (0, 1), (0, 0)), mode='constant')
    return padded

# Function to visualize feature maps from geometric tensors
def visualize_geometric_tensor(tensor, title, max_channels=8):
    """Visualize a geometric tensor's feature maps"""
    # Convert GeometricTensor to numpy array
    if hasattr(tensor, 'tensor'):
        # This is a GeometricTensor
        tensor_data = tensor.tensor.detach().cpu()
    else:
        # This is already a torch.Tensor
        tensor_data = tensor.detach().cpu()
    
    # For C8 group, reshape [B, C*8, H, W] to [B, C, 8, H, W]
    if tensor_data.ndim == 4:
        B, C, H, W = tensor_data.shape
        if C % 8 == 0:
            tensor_data = tensor_data.reshape(B, C//8, 8, H, W)
    
    # Prepare the visualization
    if tensor_data.ndim == 5:
        # We have a tensor with shape [B, C, G, H, W] where G is the group dimension
        B, C, G, H, W = tensor_data.shape
        
        # Limit the number of channels to display
        C = min(C, max_channels)
        
        fig, axes = plt.subplots(C, G, figsize=(2*G, 2*C))
        if C == 1:
            axes = axes.reshape(1, -1)
        
        for c in range(C):
            for g in range(G):
                # Get the feature map for this channel and group element
                fmap = tensor_data[0, c, g]
                
                # Display the feature map
                im = axes[c, g].imshow(fmap, cmap='viridis')
                axes[c, g].axis('off')
                
                # Add colorbar
                if g == G-1:
                    plt.colorbar(im, ax=axes[c, g])
        
        plt.suptitle(f"{title} - Feature Maps [Channels x Group Elements]")
        plt.tight_layout()
        plt.savefig(f"/teamspace/studios/this_studio/unaugmented_dataset_featmaps{title}.png")
    
    elif tensor_data.ndim <= 4:
        # Regular tensor
        if tensor_data.ndim == 4:
            B, C, H, W = tensor_data.shape
        elif tensor_data.ndim == 3:
            C, H, W = tensor_data.shape
            tensor_data = tensor_data.unsqueeze(0)
            B = 1
        else:
            return  # Can't visualize
        
        # Limit the number of channels to display
        C = min(C, max_channels)
        
        fig, axes = plt.subplots(1, C, figsize=(3*C, 3))
        if C == 1:
            axes = [axes]
        
        for c in range(C):
            im = axes[c].imshow(tensor_data[0, c], cmap='viridis')
            axes[c].axis('off')
            plt.colorbar(im, ax=axes[c])
        
        plt.suptitle(f"{title} - Feature Maps")
        plt.tight_layout()
        plt.savefig(f"/teamspace/studios/this_studio/unaugmented_dataset_featmaps{title}.png")

# Run the model at different rotation angles and collect feature maps
print("Running the model at different rotation angles...")
all_feature_maps = {}

# Register the hooks
handles = register_hooks(model)

for angle in config.angles:
    print(f"Processing angle {angle}°...")
    # Rotate the image
    rotated_img = rotate_image(image_pil, angle)
    
    # Preprocess the image
    img_tensor = preprocess_image(rotated_img)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
    
    # Store the feature maps
    all_feature_maps[angle] = {k: v for k, v in feature_maps.items()}
    
    # Clear feature maps for next iteration
    feature_maps.clear()

# Remove the hooks
for handle in handles:
    handle.remove()

# Visualize feature maps for each angle
def visualize_all_angles(all_feature_maps, layer_name):
    """Visualize feature maps from a specific layer across all angles"""
    print(f"Visualizing feature maps for layer: {layer_name}")
    
    # Check if this layer has feature maps
    if not any(layer_name in maps for maps in all_feature_maps.values()):
        print(f"No feature maps found for layer {layer_name}")
        return
    
    # Display feature maps for each angle
    for angle, maps in all_feature_maps.items():
        if layer_name in maps:
            visualize_geometric_tensor(
                maps[layer_name], 
                f"Angle {angle}° - {layer_name}", 
                max_channels=config.max_channels
            )

# Example usage - visualize each layer's feature maps across angles
print("Visualizing feature maps...")
for layer_name in ['block1', 'block2', 'pool1', 'block3', 'block4', 'pool2', 'block5', 'pool2_5', 'block6', 'pool3', 'gpool']:
    visualize_all_angles(all_feature_maps, layer_name)

# Optional: compare model outputs at different angles
print("\nComparing model outputs at different angles:")
outputs = {angle: all_feature_maps[angle].get('output', None) for angle in config.angles}
print(outputs)
for angle, output in outputs.items():
    if output is not None:
        probs = torch.softmax(output, dim=1)
        print(f"Angle {angle}°: Class probabilities = {probs[0].cpu().numpy()}")



