import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

# Import from our modules
from models import C8SteerableCNN
from datasets import PCamDataset, PCamDatasetInMemory
from transforms import TransformBuilder
from train import train_model
from utils import evaluate_model, test_model_rotation_invariance, visualize_predictions

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train E(n)-Equivariant CNN on PCam dataset')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the PCam dataset directory')
    parser.add_argument('--use_in_memory', action='store_true',
                        help='Load the entire dataset into memory for faster training')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                        choices=['reduce_on_plateau', 'cosine', 'none'],
                        help='Learning rate scheduler to use')
    
    # Model arguments
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels (3 for RGB)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of output classes')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name for saving outputs')
    
    return parser.parse_args()

def main():
    """Main function for training and evaluating the model."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Create experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"pcam_steerable_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directories
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    log_dir = os.path.join(exp_dir, 'logs')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create transforms
    train_transform = TransformBuilder.build_train_transform()
    test_transform = TransformBuilder.build_test_transform()
    
    # Create datasets
    dataset_class = PCamDatasetInMemory if args.use_in_memory else PCamDataset
    
    train_dataset = dataset_class(
        root_dir=args.data_dir,
        mode='train',
        transform=train_transform
    )
    
    val_dataset = dataset_class(
        root_dir=args.data_dir,
        mode='val',
        transform=test_transform
    )
    
    test_dataset = dataset_class(
        root_dir=args.data_dir,
        mode='test',
        transform=test_transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    model = C8SteerableCNN(
        in_channels=args.in_channels,
        n_classes=args.n_classes
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create training configuration
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler if args.scheduler != 'none' else None,
        'num_workers': args.num_workers,
        'device': args.device,
        'log_dir': log_dir,
        'save_dir': checkpoint_dir,
        'early_stopping': 10,  # Stop training if no improvement for 10 epochs
        'metric_name': 'auc',  # Use AUC as the primary metric for model selection
    }
    
    # Train the model
    print("Starting training...")
    model, history = train_model(model, train_dataset, val_dataset, config)
    
    # Test on final test dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_metrics = evaluate_model(model, test_loader, device=args.device)
    print("Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save test metrics
    with open(os.path.join(exp_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Plot training history
    train_losses, val_metrics = history
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'training_loss.png'))
    
    # Plot validation metrics
    plt.figure(figsize=(10, 5))
    for metric, values in val_metrics.items():
        plt.plot(values, label=f'Validation {metric}')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'validation_metrics.png'))
    
    # Test rotation invariance
    # Get a sample image from the test set
    sample_image, _ = test_dataset[0]
    sample_image = sample_image.permute(1, 2, 0).numpy()
    if sample_image.max() <= 1.0:
        sample_image = (sample_image * 255).astype(np.uint8)
    sample_image = Image.fromarray(sample_image)
    
    test_model_rotation_invariance(model, sample_image, device=args.device)
    
    # Visualize some predictions
    visualize_predictions(model, test_loader, num_samples=10, device=args.device)
    plt.savefig(os.path.join(exp_dir, 'predictions.png'))
    
    print(f"Training completed. Results saved to {exp_dir}")

if __name__ == '__main__':
    main()