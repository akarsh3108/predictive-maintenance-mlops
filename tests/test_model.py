"""Tests for the PyTorch classifier."""

from __future__ import annotations

import torch

from src.models.classifier import FailurePredictor


def test_model_forward_shape():
    model = FailurePredictor(n_features=7)
    model.eval()  # batch norm needs eval mode or batch > 1
    x = torch.randn(4, 7)
    out = model(x)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_model_predict_proba_in_unit_interval():
    model = FailurePredictor(n_features=7)
    model.eval()
    x = torch.randn(16, 7)
    probs = model.predict_proba(x)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_model_gradients_flow():
    """Sanity check that backprop works — catches bugs like detached tensors."""
    model = FailurePredictor(n_features=7)
    x = torch.randn(8, 7)
    y = torch.randint(0, 2, (8,)).float()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    # At least one parameter should have a non-zero gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert has_grad, "No gradients flowed — check autograd setup"


def test_model_different_hidden_dims():
    """Verify the architecture is configurable."""
    model = FailurePredictor(n_features=7, hidden_dims=(128, 64))
    model.eval()
    x = torch.randn(2, 7)
    out = model(x)
    assert out.shape == (2,)
