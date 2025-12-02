from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

@pytest.mark.parametrize("batch_size", [16, 64])
def test_nlbse_dataset(batch_size: int) -> None:
    """Tests the NLBSE code comment classification dataset to verify that it can be
    downloaded correctly and that batches match expected sizes and types.

    :param batch_size: Batch size for the DataLoader.
    """
    # Load dataset
    ds = load_dataset("NLBSE/nlbse26-code-comment-classification")

    # Verify train/validation/test splits exist
    assert "train" in ds
    assert "validation" in ds
    assert "test" in ds

    # Check that dataset is non-empty
    total_examples = len(ds["train"]) + len(ds["validation"]) + len(ds["test"])
    assert total_examples > 0

    # Create a simple PyTorch DataLoader for the train split
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=batch_size)
    test_loader = DataLoader(ds["test"], batch_size=batch_size)

    # Check that loaders return batches with correct structure
    batch = next(iter(train_loader))
    # HuggingFace datasets return dicts for columns
    assert isinstance(batch, dict)
    assert "code" in batch
    assert "comment" in batch

    # Check batch size (last batch might be smaller)
    assert len(batch["code"]) <= batch_size
    assert len(batch["comment"]) <= batch_size

    # Check types
    assert all(isinstance(x, str) for x in batch["code"])
    assert all(isinstance(y, str) for y in batch["comment"])

