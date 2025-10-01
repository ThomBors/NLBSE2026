import hydra
import logging
from omegaconf import OmegaConf
from datasets import concatenate_datasets, DatasetDict

def oversample_top_per_label(split_ds, SYNQ, X_augment):
    pass

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

