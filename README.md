
# Multi-Model Text Classification System

## Overview

This repository contains a multi-model text classification system that leverages various models, including LSTM, GRU, and BERT, to classify text prompts into different categories. The system uses different vectorization techniques for feature extraction and provides methods for training, evaluating, and predicting with these models.

## Models

### 1. **LSTM and GRU Models**

The LSTM and GRU models are Recurrent Neural Networks (RNNs) that process sequences of text to perform classification tasks.

- **LSTM Model**: Uses Long Short-Term Memory units to capture long-range dependencies in text.
- **GRU Model**: Uses Gated Recurrent Units as an alternative to LSTM, often with fewer parameters.

### 2. **BERT Model**

The BERT (Bidirectional Encoder Representations from Transformers) model is a state-of-the-art model for various NLP tasks. It uses transformers to capture contextual relationships in text.

## Installation

Ensure you have the required libraries installed. You can install them using pip:

```bash
pip install numpy pandas scikit-learn nltk tensorflow transformers torch textblob gensim
```

## Usage

1. **Prepare Data**: Update the `data` dictionary with your dataset. The dictionary should contain `prompt` (textual prompts) and `category` (categories for each prompt).

2. **Text Preprocessing**: The text is preprocessed by removing stop words and applying stemming.

3. **Vectorization**:
    - **BoW**: Converts text to a matrix of token counts.
    - **TF-IDF**: Converts text to a matrix of TF-IDF features.
    - **Word2Vec**: Generates word embeddings for text.

4. **Model Training**:
    - **LSTM/GRU**: Train LSTM and GRU models using the vectorized text data.
    - **BERT**: Train a BERT model using the text prompts and labels.

5. **Model Evaluation**: Evaluate the performance of each model using classification reports.

## Code Structure

- `text_classification.py`: Contains the code for preprocessing, vectorization, model creation, training, and evaluation for LSTM, GRU, and BERT models.

### Key Functions

- **`preprocess_text(text)`**: Preprocesses text by tokenizing, removing stop words, and applying stemming.
- **`create_rnn_model(vocab_size, embedding_dim, input_length)`**: Creates an LSTM-based RNN model.
- **`create_gru_model(vocab_size, embedding_dim, input_length)`**: Creates a GRU-based RNN model.
- **`train_and_evaluate(model, X_train, X_test, y_train, y_test)`**: Trains and evaluates LSTM and GRU models.
- **`train_bert_model(model, train_loader, epochs)`**: Trains the BERT model.
- **`evaluate_bert_model(model, texts, labels)`**: Evaluates the BERT model.

## Example

Here’s an example of how to use the code:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset

# Sample Data
data = {
    'prompt': ['example text', 'another text'],
    'category': ['category1', 'category2']
}
df = pd.DataFrame(data)

# Text Preprocessing and Vectorization
# ...

# Model Training and Evaluation
# ...

# Training BERT Model
# ...
```

## Notes

- Ensure that your environment supports GPU acceleration for faster training, especially for the BERT model.
- Adjust hyperparameters (e.g., epochs, batch size) based on the size of your dataset and available computational resources.
