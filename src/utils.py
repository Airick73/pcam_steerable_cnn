import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

def check_dimensions(tensor, expected_shape=None):
    """
    Debug helper to check tensor dimensions
    """
    if expected_shape and tensor.shape[2:] != expected_shape:
        print(f"WARNING: Expected spatial dimensions {expected_shape} but got {tensor.shape[2:]}")
        return False
    return True

def test_model_rotation_invariance(model, x, device='cuda'):
    """
    Test if a model produces the same output for rotated versions of an input image.
    
    Args:
        model: The model to test
        x: Input image (PIL Image)
        device: Device to run the model on
    """
    # Check image dimensions before starting
    img_array = np.array(x)
    print(f"Original test image shape: {img_array.shape}")
    model.eval()
    
    print('\n' + '='*80)
    print('Testing rotation invariance')
    print('='*80)
    
    with torch.no_grad():
        outputs = []
        
        # Test at 8 different rotation angles (45 degree increments)
        for r in range(8):
            angle = r * 45
            
            # Rotate the image
            x_rotated = x.rotate(angle, Image.BILINEAR)
            
            # Convert to tensor and add batch dimension
            x_tensor = torch.from_numpy(np.array(x_rotated, dtype=np.float32))
            x_tensor = x_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            x_tensor = x_tensor.to(device)
            
            # Make sure the pixel values are in the right range [0, 1]
            if x_tensor.max() > 1.0:
                x_tensor = x_tensor / 255.0
            
            # Forward pass
            y = model(x_tensor)
            
            # Store the outputs
            outputs.append(y.cpu().numpy().squeeze())
            
            # Print the results
            print(f"Rotation angle: {angle}°")
            print(f"Output: {y.cpu().numpy().squeeze()}")
            print("-" * 40)
        
        # Calculate mean and standard deviation across all rotations
        outputs = np.array(outputs)
        mean_output = np.mean(outputs, axis=0)
        std_output = np.std(outputs, axis=0)
        
        print(f"Mean output across all rotations: {mean_output}")
        print(f"Standard deviation across all rotations: {std_output}")
        print(f"Max standard deviation: {np.max(std_output)}")
        
        if np.max(std_output) < 0.1:
            print("Model appears to be approximately rotation invariant (std < 0.1)")
        else:
            print("Model is not strictly rotation invariant (std >= 0.1)")
    
    print('='*80 + '\n')


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader containing the evaluation dataset
        device: Device to run the model on
        
    Returns:
        Dictionary containing performance metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # For binary classification, get probability of positive class
            if outputs.shape[1] == 2:
                probs = F.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
            else:
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    metrics = {
        'accuracy': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def visualize_predictions(model, data_loader, num_samples=5, device='cuda'):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing the test dataset
        num_samples: Number of samples to visualize
        device: Device to run the model on
    """
    model.eval()
    
    # Get samples from the data loader
    samples = []
    for inputs, labels in data_loader:
        for i in range(min(num_samples, len(inputs))):
            samples.append((inputs[i], labels[i]))
        if len(samples) >= num_samples:
            break
    
    # Set up the plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    with torch.no_grad():
        for i, (image, label) in enumerate(samples):
            # Make prediction
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            
            # Get prediction probability
            if output.shape[1] == 2:
                prob = F.softmax(output, dim=1)[0, 1].item()
                pred = torch.argmax(output, dim=1).item()
            else:
                prob = torch.sigmoid(output).item()
                pred = int(prob > 0.5)
            
            # Convert tensor to numpy for visualization
            img = image.cpu().numpy().transpose(1, 2, 0)
            
            # Denormalize if necessary
            if img.max() <= 1.0:
                img = np.clip(img, 0, 1)
            
            # Plot original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"True: {'Tumor' if label == 1 else 'Normal'}")
            axes[i, 0].axis('off')
            
            # Plot heatmap (you could implement a more sophisticated visualization here)
            axes[i, 1].imshow(img)
            axes[i, 1].set_title(f"Pred: {'Tumor' if pred == 1 else 'Normal'} ({prob:.2f})")
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()