import pytest

import torch

from app.SentimentClassfier import SentimentModel, SentimentClassifier



def test_sentiment_model_forward():

    vocab_size = 100

    output_size = 1

    model = SentimentModel(vocab_size, output_size)



    input_tensor = torch.randint(0, vocab_size, (2, 10))  # Batch of 2, sequence length 10

    output, _ = model(input_tensor)



    assert output.shape == (2, 1)



def test_sentiment_classifier_preprocess():

    classifier = SentimentClassifier()

    processed = classifier.preprocess("This is a test sentence!")

    assert isinstance(processed, torch.Tensor)

    assert len(processed.shape) == 1  # Should be 1D tensor



def test_sentiment_classifier_classify():

    classifier = SentimentClassifier()

    result = classifier.classify("I love this product!")

    assert isinstance(result, float)

