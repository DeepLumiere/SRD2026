# Project Structure

This document provides an overview of the SRD2026 repository structure.

## Directory Structure

```
SRD2026/
├── .github/                    # GitHub configuration
│   └── workflows/             # CI/CD workflows
│       └── tests.yml          # Automated testing workflow
│
├── docs/                      # Documentation
│   ├── GETTING_STARTED.md     # Getting started guide
│   └── PROJECT_STRUCTURE.md   # This file
│
├── experiments/               # Experiment configurations
│   └── config.yaml            # Default configuration file
│
├── notebooks/                 # Jupyter notebooks
│   └── README.md              # Notebooks documentation
│
├── scripts/                   # Executable scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Inference script
│
├── src/                       # Source code
│   └── srd/                   # Main package
│       ├── __init__.py        # Package initialization
│       ├── models/            # Model implementations
│       ├── data/              # Data loading and preprocessing
│       ├── utils/             # Utility functions
│       └── configs/           # Configuration files
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_import.py         # Basic import tests
│
├── .gitignore                 # Git ignore patterns
├── CITATION.bib               # BibTeX citation
├── CONTRIBUTING.md            # Contributing guidelines
├── LICENSE                    # MIT License
├── Makefile                   # Build automation
├── README.md                  # Main documentation
├── environment.yml            # Conda environment
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
└── setup.py                   # Package setup
```

## Key Components

### Source Code (`src/srd/`)
- **models/**: Neural network architectures and model implementations
- **data/**: Dataset classes, data loaders, and preprocessing utilities
- **utils/**: Helper functions and utilities
- **configs/**: Configuration management

### Scripts (`scripts/`)
- **train.py**: Main training script with experiment management
- **evaluate.py**: Evaluation script for trained models
- **inference.py**: Inference script for making predictions

### Experiments (`experiments/`)
Configuration files for different experiments. Each configuration file defines:
- Model architecture parameters
- Training hyperparameters
- Data paths and preprocessing settings
- Optimization settings

### Tests (`tests/`)
Unit tests and integration tests for the codebase. Run with:
```bash
pytest tests/
```

### Documentation (`docs/`)
Additional documentation including guides, API references, and tutorials.

## Configuration Management

All experiments are configured through YAML files in the `experiments/` directory. 
This allows for:
- Easy reproducibility of experiments
- Version control of experimental settings
- Clear documentation of what was tried

## Development Workflow

1. **Setup**: Install dependencies with `make install` or using conda
2. **Develop**: Make changes to source code in `src/`
3. **Test**: Run `make test` to ensure everything works
4. **Experiment**: Create new config in `experiments/` and run training
5. **Evaluate**: Use evaluation scripts to assess model performance

## Adding New Features

1. Add implementation in appropriate `src/` subdirectory
2. Add tests in `tests/`
3. Update documentation in `docs/`
4. Add example usage in `notebooks/` (optional)
5. Update README.md if necessary

## Package Installation

The package can be installed in development mode:
```bash
pip install -e .
```

This allows you to import the package from anywhere while still being able to edit the source code.
