# See, Read, Describe: Entity-Grounded Captioning with Multimodal LLMs

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-GRAIL--CVPR%202026-blue)](https://github.com/DeepLumiere/SRD2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

## 📋 Overview

This repository contains the official implementation of **"See, Read, Describe: Entity-Grounded Captioning with Multimodal LLMs"**, submitted to GRAIL-CVPR 2026.

**Abstract**: [Add your paper abstract here]

## 🎯 Key Features

- **Entity-Grounded Captioning**: Novel approach for generating descriptions grounded in specific entities
- **Multimodal LLM Integration**: Leverages state-of-the-art multimodal large language models
- **Comprehensive Evaluation**: Extensive experiments on benchmark datasets
- **Reproducible Results**: Full training and evaluation code provided

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ GPU memory recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DeepLumiere/SRD2026.git
cd SRD2026
```

2. Create a conda environment:
```bash
conda env create -f environment.yml
conda activate srd2026
```

Or use pip:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## How to run

Prerequisites
- Python 3.13
- Optional but recommended: GPU with CUDA 11.8+ for running Florence-2 and other large models efficiently
- Ollama (for the captioning/LLM script) if you plan to use local LLMs

Quick start (example workflow)
1. Prepare an input image (or use the bundled example):
   - example image path: example_resources/sample_image.jpg

2. Extract multimodal context (runs Florence-2 inference and writes JSON):
```bash
# from repo root
python scripts/context_extractor.py example_resources/sample_image.jpg
# Output: example_resources/florence2_output.json
```

3. Run the captioning/grounded-LM interface (will try to start Ollama if available):
```bash
# from repo root
python scripts/caption_interface.py
# Uses JSON_SOURCE configured in the script (default: example_resources/florence2_output.json)
```

Notes about Ollama
- If you don't have Ollama installed and want to use a local LLM, install it per https://ollama.com and run:
```bash
ollama serve
```
- The caption_interface script will try to detect and start/pull models when possible; you can also set OLLAMA_API_URL in your environment to point to a different host/port.

Troubleshooting
- If the context extractor fails due to missing packages, re-run the same command; the script will auto-install missing pip packages and restart.
- If Florence-2 or other models are too large for your GPU, run the extractor on CPU (slow) or use a smaller model.

### Inference
For Context Extraction:
```bash
python scripts/context_extractor.py --image /path/to/image.jpg
```
For Captioning/LLM Interface:
```bash
python scripts/caption_interface.py
```
## 📊 Results

[Add your experimental results, tables, and visualizations here]

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
YET TO PUBLISH
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact:
- AUTHOR- [GitHub](https://github.com/DeepLumiere)

## 🙏 Acknowledgments

[Add acknowledgments for datasets, pre-trained models, or other resources used]

---

**Note**: This code is associated with a paper submitted to GRAIL-CVPR 2026. Full details will be available upon publication.