# Getting Started with SRD2026

This guide will help you get started with the SRD2026 codebase.

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate srd2026

# Install package in editable mode
pip install -e .
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Data Preparation

[Add instructions for preparing datasets]

## Model Training

To train the model from scratch:

```bash
python scripts/train.py --config experiments/config.yaml
```

To resume training from a checkpoint:

```bash
python scripts/train.py --config experiments/config.yaml --resume checkpoints/latest.pth
```

## Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_path /path/to/test/data
```

## Inference

To run inference on a single image:

```bash
python scripts/inference.py --image /path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

## Configuration

All model and training configurations are defined in YAML files in the `experiments/` directory. 

Key configuration parameters:
- `model`: Model architecture settings
- `training`: Training hyperparameters
- `data`: Data paths and preprocessing settings
- `optimization`: Optimizer and scheduler settings

## Tips

- Start with the provided configuration file and adjust as needed
- Monitor training with TensorBoard or Weights & Biases
- Use mixed precision training for faster training on modern GPUs
- Adjust batch size based on your GPU memory

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in configuration
- Enable gradient checkpointing
- Use smaller image sizes

### Slow Training
- Increase number of data loading workers
- Enable mixed precision training
- Use distributed training for multiple GPUs

For more help, please open an issue on GitHub.
