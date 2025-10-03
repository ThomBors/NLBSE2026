import hydra
import logging
import numpy as np
from omegaconf import OmegaConf
from datasets import concatenate_datasets, DatasetDict,Dataset

def oversample_top_per_label(split_ds, SYNQ, X_augment):
    """
    Keep all real examples and add top synthetic examples per label
    based on similarity_score closeness to SYNQ.
    
    Args:
        split_ds (Dataset): HuggingFace dataset with fields:
            - "labels" (list[int]): one-hot or multi-label encoding
            - "synthetic" (bool)
            - "similarity_score" (float)
        SYNQ (float): target similarity quality.
        X_augment (list[int]): number of synthetic examples to add per label index.
    
    Returns:
        Dataset: new dataset with all real + selected synthetic examples.
    """
    selected_examples = []

    for label_id, n_samples in enumerate(X_augment):
        if n_samples <= 0:
            continue

        # synthetic examples that belong to this label
        label_ds = split_ds.filter(
            lambda x: x["synthetic"] and x["labels"][label_id] == 1
        )

        if len(label_ds) == 0:
            continue

        # closeness to SYNQ
        scores = np.abs(np.array(label_ds["similarity_score"]) - SYNQ)

        # take top-N closest
        top_idx = np.argsort(scores)[:n_samples]
        selected_examples.extend(label_ds.select(top_idx))

    # keep all real and high quelity data
    real_ds = split_ds.filter(
        lambda x: (not x["synthetic"]) or x["similarity_score"] > 0.95
    )
    
    # build augmented dataset
    if selected_examples:
        aug_ds = Dataset.from_list(selected_examples, features=split_ds.features)
        return concatenate_datasets([real_ds,aug_ds])
    else:
        return real_ds
    


def filter_high_quality_synthetic(example, threshold):
    """
    Keep all real data and synthetic data with similarity_score > threshold.
    """
    if not example.get("synthetic", False):
        return True  # keep all real data
    return example.get("similarity_score", 0) > threshold


def filter_dataset_by_quality(ds, threshold):
    """
    Apply high-quality synthetic filter to a Hugging Face dataset.
    Returns a filtered dataset containing:
      - All real data
      - Only synthetic data with similarity_score > threshold
    """
    return ds.filter(filter_high_quality_synthetic, fn_kwargs={"threshold": threshold})


def oversampling(cfg, ds: DatasetDict, SYNQ: float):
    """
    Apply oversampling to training splits, then filter datasets to keep:
      - All real data
      - High-quality synthetic data (similarity_score > SYNQ)
    """
    balancer_cfg = OmegaConf.to_container(cfg.component.balancer, resolve=True)
    Balancer = hydra.utils.instantiate(OmegaConf.create(balancer_cfg))

    filtered_ds = {}

    for split_name, split_ds in ds.items():
        if split_name.endswith("_train"):
            # Compute augmentation numbers using your balancer
            X_augment = Balancer(ds, split_name)
            logging.info(f"Augmentation numbers for {split_name}: {X_augment}")

            if X_augment:
                # Apply per-label oversampling
                split_ds = oversample_top_per_label(split_ds, SYNQ, X_augment)

            # Filter to keep only high-quality synthetic + all real data
            split_ds = filter_dataset_by_quality(split_ds, SYNQ)

        filtered_ds[split_name] = split_ds

    return DatasetDict(filtered_ds)

