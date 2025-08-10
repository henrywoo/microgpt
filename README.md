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

```bash
pip install -e .
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
from microgpt.model import GPT, GPTConfig

# Create configuration
config = GPTConfig(
    n_layer=6,
    n_head=6, 
    n_embd=384,
    block_size=256
)

# Initialize model
model = GPT(config)

# Generate text
generated = model.generate(
    idx=torch.tensor([[1, 2, 3]]), 
    max_new_tokens=50,
    temperature=0.8
)
```

### 🎭 Sampling from Trained Models

After training a model, you can generate text samples using the `sample.py` script. This script loads a trained checkpoint and generates text based on your specifications.

#### 🚀 Basic Usage

```bash
# Generate samples from a trained model
python -m microgpt.sample
```

## 🌟 Highlights

- ✅ **Lightweight Architecture**: Parameter count reduced to 1/8 of standard GPT-2
- ✅ **Fast Inference**: Flash Attention support for 3-5x speed improvement
- ✅ **Flexible Configuration**: Adjustable model size based on hardware resources
- ✅ **Easy to Use**: Clean API design for quick adoption
- ✅ **Resource-Friendly**: Suitable for mobile and embedded device deployment
- ✅ **Rapid Training**: Supports fast prototyping and experimentation


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NanoGPT](https://github.com/karpathy/nanoGPT) - Original codebase
- [OpenAI GPT-2](https://github.com/openai/gpt-2) - Architecture reference
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Implementation reference
- [microBERT](https://github.com/henrywoo/microbert) - Design philosophy inspiration for lightweight architecture

## Contact

For questions or suggestions, please:

- Submit an [Issue](https://github.com/your-username/microgpt/issues)
- Email: [wufuheng@gmail.com]

---

**microGPT** - Making AI lighter, deployment simpler 🚀
