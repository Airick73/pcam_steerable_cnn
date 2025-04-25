# E(n)-Equivariant Steerable CNN for PCam Dataset

This repository contains a modular implementation of an E(n)-Equivariant Steerable CNN for the PCam (PatchCamelyon) dataset, designed for cancer detection in histopathology images. The model is based on group equivariant convolutional neural networks that are invariant to rotations.

## Project Structure

```
project_root/
├── models.py       # Model architecture 
├── datasets.py     # Dataset handling
├── transforms.py   # Image transformations
├── train.py        # Training functionality
├── utils.py        # Utility functions
└── main.py         # Entry point script
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- escnn (Equivariant Steerable CNNs library)
- h5py (for PCam dataset)
- scikit-learn
- matplotlib
- tqdm

Install the requirements with:

```
pip install torch torchvision escnn h5py scikit-learn matplotlib tqdm
```

## Dataset

The PCam dataset consists of 96x96 color patches extracted from histopathology scans. The task is to classify each patch as either containing a tumor region (label=1) or not (label=0).

The dataset should be organized in the following structure:

```
data/raw/
├── camelyonpatch_level_2_split_train_x.h5
├── camelyonpatch_level_2_split_train_y.h5
├── camelyonpatch_level_2_split_val_x.h5
├── camelyonpatch_level_2_split_val_y.h5
├── camelyonpatch_level_2_split_test_x.h5
└── camelyonpatch_level_2_split_test_y.h5
```

You can download the PCam dataset from the official source: [PCam Dataset](https://github.com/basveeling/pcam).

## Model Architecture

The model is based on an E(n)-Equivariant Steerable CNN with:

- Group equivariance to 8 rotations (C8 group)
- Multiple convolutional layers with group convolutions
- Group pooling for rotation invariance
- Final fully-connected layers for classification

## Usage

### Training

To train the model with default parameters:

```bash
python main.py --data_dir ../data
```

Advanced usage with custom parameters:

```bash
python main.py \
    --data_dir ./data \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 5e-5 \
    --optimizer adam \
    --scheduler cosine \
    --device cuda \
    --output_dir ./experiments \
    --exp_name pcam_steerable_cnn
```

### Model Evaluation

The model will be automatically evaluated on the test set after training. The results, including metrics and visualizations, will be saved to the output directory.

### Testing Rotation Invariance

The training script includes a function to test the rotation invariance of the trained model. It rotates an input image at different angles and checks if the output predictions remain consistent.

## Customization

- **Dataset**: Modify `datasets.py` to work with different datasets
- **Model**: Adjust the architecture in `models.py` to experiment with different configurations
- **Training**: Customize training parameters in `main.py` or directly in `train.py`

## Example Results

After training, you'll find the following in your experiment directory:

```
experiments/pcam_steerable_cnn/
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
├── args.json            # Training arguments
├── test_metrics.json    # Test set metrics
├── training_loss.png    # Loss curve plot
├── validation_metrics.png  # Validation metrics plot
└── predictions.png      # Visualization of model predictions
```

## Acknowledgments

This implementation is based on the E(n)-Equivariant Steerable CNNs architecture. The original notebook was adapted for modular Python structure and the PCam dataset.

## References

- E(n)-Equivariant Steerable CNNs: [escnn library](https://github.com/QUVA-Lab/escnn)
- PCam Dataset: Veeling et al., "Rotation Equivariant CNNs for Digital Pathology", 2018