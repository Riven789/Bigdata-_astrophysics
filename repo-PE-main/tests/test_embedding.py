import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from embedding import JointSpectrogramEmbedder, SpectrogramEmbedder

def test_joint_embedding_forward():
    embedder = SpectrogramEmbedder(embedding_dim=128)
    model = JointSpectrogramEmbedder(embedder)

    h1_input = torch.randn(2, 1, 64, 64)
    l1_input = torch.randn(2, 1, 64, 64)

    output = model(h1_input, l1_input)

    assert output.shape == (2, 256), f"Expected output shape (2, 256), got {output.shape}"

def test_joint_embedding_no_nan():
    embedder = SpectrogramEmbedder(embedding_dim=128)
    model = JointSpectrogramEmbedder(embedder)

    h1_input = torch.randn(2, 1, 64, 64)
    l1_input = torch.randn(2, 1, 64, 64)

    output = model(h1_input, l1_input)

    assert not torch.isnan(output).any(), "Output contains NaN values!"

def test_joint_embedding_backward_pass():
    embedder = SpectrogramEmbedder(embedding_dim=128)
    model = JointSpectrogramEmbedder(embedder)

    h1_input = torch.randn(2, 1, 64, 64, requires_grad=True)
    l1_input = torch.randn(2, 1, 64, 64, requires_grad=True)

    output = model(h1_input, l1_input)
    loss = output.mean()
    loss.backward()

    # Check that gradients exist
    assert h1_input.grad is not None, "No gradient computed for h1_input"
    assert l1_input.grad is not None, "No gradient computed for l1_input"

import pytest

def test_joint_embedding_bad_input_shape():
    embedder = SpectrogramEmbedder(embedding_dim=128)
    model = JointSpectrogramEmbedder(embedder)

    h1_input = torch.randn(2, 1, 60, 60)
    l1_input = torch.randn(2, 1, 64, 64)

    with pytest.raises(ValueError):
        model(h1_input, l1_input)
