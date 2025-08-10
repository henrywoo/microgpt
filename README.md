# microGPT 🚀

> **Lightweight GPT implementation designed for resource-constrained environments**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

microGPT is a lightweight implementation of GPT (Generative Pre-trained Transformer) language models, inspired by [NanoGPT](https://github.com/karpathy/nanoGPT) but following the same design philosophy as **[microBERT](https://github.com/henrywoo/microbert)**: **significantly reducing computational resource requirements while maintaining model performance**.

## ✨ Key Features

### 🎯 **Lightweight Design**
- **Model Compression**: Significantly reduced parameter count through carefully designed architecture
- **Computational Optimization**: Flash Attention support for improved inference efficiency
- **Memory Efficient**: Optimized for resource-constrained environments

### 🚀 **Resource Adaptation**
- **Mobile-Friendly**: Runs on laptops, embedded devices, and mobile platforms
- **Fast Training**: Supports rapid prototyping and experimentation
- **Flexible Configuration**: Adjustable model size based on hardware resources

## 🏗️ Architecture

### Core Components
- **Transformer Blocks**: Standard self-attention + MLP architecture
- **Flash Attention**: Efficient attention computation for PyTorch 2.0+
- **Weight Tying**: Token embedding and output layer weight sharing
- **Layer Normalization**: Optional bias support

### Default Configuration (Lightweight)
```python
n_layer = 6      # 6 Transformer layers
n_head = 6       # 6 attention heads
n_embd = 384     # 384-dimensional embeddings
block_size = 256 # 256 token context window
```

## 🚀 Quick Start

### Package-Based Training
microGPT is designed to work as a standalone package. After installation, you can train models from any directory without needing the source code.

### 📊 Dataset Preparation

microGPT comes with a built-in Shakespeare dataset for character-level language modeling. The dataset preparation script and raw text data are included in the package for easy access. The dataset preparation process:

1. **Uses** the Shakespeare text included in the package
2. **Tokenizes** characters into integers (vocabulary size: ~65 characters)
3. **Splits** data into training (90%) and validation (10%) sets
4. **Saves** processed data in `./data/shakespeare_char/` relative to your current working directory

### 📦 Installation

#### Development Installation (for contributors)
```bash
# Clone the repository first
git clone https://github.com/henrywoo/microgpt.git
cd microgpt

# Install in editable mode for development
pip install -e .
```

#### Production Installation (for users)
```bash
# Install directly from PyPI
pip install microgpt
```

### 🎓 Training

#### Complete Training Workflow
You can train microGPT from any directory using the installed package:

```bash
# 1. Prepare the dataset (creates ./data/shakespeare_char/)
python -m microgpt.prepare_dataset

# 2. Start training (uses the prepared dataset)
python -m microgpt.pretrain.clm_pretrain_v0
```

**No git repo checkout required!** After installation, you can run training from anywhere.

#### 1. Prepare the Dataset
First, prepare the Shakespeare dataset for character-level language modeling:

```bash
# From any directory where you want to store the data
python -m microgpt.prepare_dataset
```

This script will:
- **Uses** the Shakespeare text included in the package
- **Tokenizes** characters into integers (vocabulary size: ~65 characters)
- **Splits** data into training (90%) and validation (10%) sets
- **Saves** processed data in `./data/shakespeare_char/` relative to your current working directory
- **Shows** the exact path where data is saved for easy reference

#### 2. Start Training

```bash
# From any directory, run the training script directly from the package
python -m microgpt.pretrain.clm_pretrain_v0
```

The training script will automatically:
- Load the prepared dataset from `./data/shakespeare_char/` (relative to current directory)
- Initialize the microGPT model with default configuration
- Train using the specified hyperparameters
- Save checkpoints and generate sample text

### 🔌 API Usage

```python
import torch
from microgpt.model import MicroGPT, MicroGPTConfig

# Create configuration
config = MicroGPTConfig(
    n_layer=6,
    n_head=6, 
    n_embd=384,
    block_size=256,
    vocab_size=65  # Must match the vocabulary size in meta.pkl
)

# Initialize model
model = MicroGPT(config)

# Generate text
# Note: For meaningful text generation, the model should be trained first
# This example shows the structure, but untrained models will generate random text
generated = model.generate(
    idx=torch.tensor([[1, 2, 3]]), 
    max_new_tokens=50,
    temperature=0.8
)

# Decode the generated text
generated_text = MicroGPT.decode_text(generated[0])
print(f"Generated text: {generated_text}")
```

### 🎭 Sampling from Trained Models

After training a model, you can generate text samples using the `sample.py` script. This script loads a trained checkpoint and generates text based on your specifications.

#### 🚀 Basic Usage

```bash
# Generate samples from a trained model
python -m microgpt.sample
```

## Acknowledgments

- [NanoGPT](https://github.com/karpathy/nanoGPT) - Original codebase
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Implementation reference
- [microBERT](https://github.com/henrywoo/microbert) - Design philosophy inspiration for lightweight architecture

## Contact

For questions or suggestions, please:

- Submit an [Issue](https://github.com/henrywoo/microgpt/issues)
- Email: [wufuheng@gmail.com]

---

**microGPT** - Making AI lighter, deployment simpler 🚀
