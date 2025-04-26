# Simple Performance Test Data Model

## Overview

This is a **simple machine learning model** using **DistilGPT-2** to predict system performance from synthetic test data. It learns patterns in **CPU**, **Memory**, and **Disk** metrics to predict performance labels like **LowLoad**, **MediumLoad**, and **HighLoad**.

## Key Features

- **Simple GPT-2 Model**: Fast and efficient.
- **Synthetic Data**: Trained with synthetic performance data.
- **Easy to Use**: Predicts performance labels for test data.

## Usage

1. **Train the Model**: Run `model.py` with synthetic data.
2. **Predict Behavior**:
    ```python
    prompt = "CPU Load: 85%, Memory: 1.5GB, Disk: 95% -> Label:"
    print(generate_label(prompt, model, tokenizer))
    ```

## Requirements

- Python 3.x
- Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Conclusion

This is a simple demo for predicting system performance from test data. It can be extended for real-world applications.
