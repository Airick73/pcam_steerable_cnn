import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
import logging
from utils import evaluate_model

class Trainer:
    """
    Trainer class for training and evaluating models.
    """
    def __init__(self, model, train_loader, val_loader, 
                 criterion=None, optimizer=None, scheduler=None,
                 device='cuda', log_dir='./logs'):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function (default: CrossEntropyLoss for binary classification)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (default: None)
            device: Device to run training on
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Default criterion for binary classification
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Default optimizer
        self.optimizer = optimizer if optimizer is not None else \
                        optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        self.scheduler = scheduler
        
        # Setup logging
        self.setup_logging(log_dir)
        
        # Initialize tracking variables
        self.epoch = 0
        self.best_val_metric = 0
        self.train_losses = []
        self.val_metrics = {}
    
    def setup_logging(self, log_dir):
        """
        Setup logging for the training process.
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure file logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        steps = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')
        
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            targets = targets.argmax(dim=1)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            steps += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss/steps:.4f}'})
        
        avg_loss = total_loss / steps
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        
        # Log metrics
        message = f"Validation metrics - "
        message += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(message)
        
        # Store metrics
        for k, v in metrics.items():
            if k not in self.val_metrics:
                self.val_metrics[k] = []
            self.val_metrics[k].append(v)
        
        return metrics
    
    def train(self, num_epochs, save_dir='./checkpoints', 
              save_best_only=True, early_stopping=None, metric_name='accuracy'):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            save_best_only: Only save model if it achieves the best validation metric
            early_stopping: Number of epochs to wait before early stopping (None to disable)
            metric_name: Metric to use for checkpoint saving and early stopping
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize early stopping counter
        no_improvement = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for _ in range(num_epochs):
            self.epoch += 1
            
            # Train one epoch
            epoch_start = time.time()
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            current_metric = val_metrics[metric_name]
            
            # Step learning rate scheduler if provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()
            
            # Log epoch statistics
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {self.epoch}/{num_epochs} - "
                           f"Loss: {train_loss:.4f}, {metric_name.capitalize()}: {current_metric:.4f}, "
                           f"Time: {epoch_time:.2f}s")
            
            # Check if this is the best model
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.logger.info(f"New best {metric_name}: {current_metric:.4f}")
                
                # Save model
                if save_best_only:
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'))
                no_improvement = 0
            else:
                no_improvement += 1
                self.logger.info(f"No improvement for {no_improvement} epochs")
            
            # Always save latest model
            if not save_best_only:
                self.save_checkpoint(os.path.join(save_dir, f'model_epoch_{self.epoch}.pth'))
            
            # Check for early stopping
            if early_stopping is not None and no_improvement >= early_stopping:
                self.logger.info(f"Early stopping after {self.epoch} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.2f} minutes")
        self.logger.info(f"Best {metric_name}: {self.best_val_metric:.4f}")
        
        return self.train_losses, self.val_metrics
    
    def save_checkpoint(self, filepath):
        """
        Save a model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint to
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load a model checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_metric = checkpoint['best_val_metric']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")


def train_model(model, train_dataset, val_dataset, config):
    """
    Train a model using the Trainer class.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Dictionary containing training configuration
    
    Returns:
        Trained model and training history
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Define loss function
    if config.get('class_weights') is not None:
        weights = torch.tensor(config['class_weights']).to(config.get('device', 'cuda'))
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if config.get('optimizer') == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-4)
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
    
    # Define learning rate scheduler
    if config.get('scheduler') == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('lr_factor', 0.1),
            patience=config.get('lr_patience', 5),
            verbose=True
        )
    elif config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.get('device', 'cuda'),
        log_dir=config.get('log_dir', '../logs')
    )
    
    # Train model
    train_losses, val_metrics = trainer.train(
        num_epochs=config.get('num_epochs', 100),
        save_dir=config.get('save_dir', '../checkpoints'),
        save_best_only=config.get('save_best_only', True),
        early_stopping=config.get('early_stopping', None),
        metric_name=config.get('metric_name', 'accuracy')
    )
    
    return model, (train_losses, val_metrics)