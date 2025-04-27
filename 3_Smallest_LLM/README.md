# Mini LLM (Language Model) in PyTorch

This is a **minimalistic Language Model (LLM)** implemented in **PyTorch**, demonstrating key components of a Transformer-based model. It includes **tokenization**, **self-attention**, **feed-forward networks**, and a **language modeling head**. 

The model is built to run on **GPU** if available, and can generate word predictions based on input sequences.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Model Components](#model-components)
5. [Example](#example)
6. [License](#license)

---

## Overview

This is a basic implementation of a **mini LLM** using the Transformer architecture. The model consists of:
- **Tokenization**: Converts words into token IDs.
- **Embedding Layer**: Converts token IDs into word embeddings.
- **Self-Attention Layer**: Applies self-attention to capture word relationships.
- **Feed-Forward Network (FFN)**: Processes the hidden states from attention.
- **Residual Connections**: Short-cuts to maintain information.
- **Language Modeling Head**: Predicts the next word based on the model's output.

---

## Requirements

You can run this on Google Colab with **PyTorch** installed. PyTorch is already pre-installed in Colab.

### Install PyTorch (if not installed):
```bash
!pip install torch
