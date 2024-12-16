import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import json
from utils import clean_pipeline, preprocess_pipeline
import os


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=128, embedding_size=400, n_layers=2, dropout=0.2):
        super(SentimentModel, self).__init__()

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM layer preserved by PyTorch library
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # convert feature to long
        x = x.long()

        # map input to vector
        x = self.embedding(x)

        # pass forward to lstm
        o, _ = self.lstm(x)

        # get last sequence output
        o = o[:, -1, :]

        # apply dropout and fully connected layer
        o = self.dropout(o)
        o = self.fc(o)

        # sigmoid
        o = self.sigmoid(o)

        return o, _


class SentimentClassifier:
    def __init__(self, device="cpu"):
        """
        Initialize the SentimentClassifier.

        Args:
            model_path (str): Path to the saved model state_dict (.pt file).
            vocab_path (str): Path to the vocabulary JSON file.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.device = torch.device(device)

        # Directory of SentimentClassifier.py
        current_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_dir, "../models/vocab.json"), "r") as f:
            self.vocab = json.load(f)

        vocab_size = len(self.vocab)
        output_size = 1
        embedding_size = 256
        hidden_size = 512
        n_layers = 2
        dropout = 0.25

        # model initialization
        self.model = SentimentModel(
            vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)

        self.model.load_state_dict(torch.load(os.path.join(
            current_dir, "../models/sentiment_lstm.pth"), map_location=self.device))

        self.model.eval()

    def preprocess(self, text):
        """
        Preprocess the input text: tokenize and convert to indices.

        Args:
            text (str): The input text to preprocess.

        Returns:
            torch.Tensor: The processed input tensor.
        """
        text = clean_pipeline(text)
        text = preprocess_pipeline(text)

        tokens = text.split()
        tokens = [token.lower() for token in tokens]

        indices = [self.vocab.get(token, self.vocab.get("<UNK>"))
                   for token in tokens]
        indices = [x for x in indices if x is not None]
        return torch.tensor(indices, dtype=torch.long)

    def classify(self, text):
        """
        Classify the sentiment of a single text.

        Args:
            text (str): The input text.

        Returns:
            float: The sentiment score (0 = negative, 1 = positive).
        """
        input_tensor = self.preprocess(text).unsqueeze(
            0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            output, _ = self.model(input_tensor)
        sentiment = 1 if output.item() >= 0.5 else 0
        return output.item()

    def classify_batch(self, texts):
        """
        Classify the sentiment of a batch of texts.

        Args:
            texts (list of str): A list of input texts.

        Returns:
            list of float: Sentiment scores for each text.
        """
        input_tensors = [self.preprocess(text) for text in texts]
        max_len = max(len(tensor) for tensor in input_tensors)

        # Pad sequences to have equal length
        padded_tensors = [
            torch.cat([tensor, torch.zeros(
                max_len - len(tensor), dtype=torch.long)])
            for tensor in input_tensors
        ]
        input_tensor = torch.stack(padded_tensors).to(self.device)

        with torch.no_grad():
            outputs, _ = self.model(input_tensor)
        return outputs.squeeze().tolist()
