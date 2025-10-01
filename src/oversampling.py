import hydra
import pandas as pd
import logging
from omegaconf import OmegaConf
from datasets import concatenate_datasets


def filter_synthetic(example, SyntheticQualityScore):
    """
    Keep all rows with similarity_score > threshold.
    If similarity_score is missing, treat it as 0 (filtered out).
    """
    if not example.get("synthetic", False):
        return True  # keep all real data
    return example.get("similarity_score", 0) > SyntheticQualityScore


def select_top_synthetic(ds, SYNQ, label=None, top_k=1):
    """Select top_k synthetic examples in ds with the exact given label (optional)."""
    # Filter by synthetic quality score
    filtered = ds.filter(filter_synthetic, fn_kwargs={"SyntheticQualityScore": SYNQ})

    # If label is provided, filter by exact label match
    if label is not None:
        filtered = filtered.filter(lambda ex: ex["labels"] == label)

    # Sort by similarity_score increasing (lower similarity first)
    sorted_indices = sorted(
        range(len(filtered)),
        key=lambda i: filtered[i]["similarity_score"],
        reverse=False,
    )


    # Select top_k
    return filtered.select(sorted_indices[:top_k])


def oversample_top_per_label(ds, SYNQ, X_augment):
    """
    Oversample top synthetic examples per label and record counts.
    """
    num_labels = len(X_augment)
    top_examples_all = []

    for idx, n in enumerate(X_augment):
        if n > 0:
            label = [0] * num_labels
            label[idx] = 1
            top_examples = select_top_synthetic(ds, SYNQ, label=label, top_k=n)
            if len(top_examples) > 0:
                top_examples_all.append(top_examples)

    if top_examples_all:
        #ds = ds.concatenate(top_examples_all)
        ds = concatenate_datasets([ds] + top_examples_all)

    return ds

def oversample_per_obser(ds, SYNQ):
    """
    Oversample all synthetic observations <= SYNQ.
    """
    filtered = ds.filter(filter_synthetic, fn_kwargs={"SyntheticQualityScore": SYNQ})
    return filtered

def oversampling(cfg, ds, SYNQ):
    balancer_cfg = OmegaConf.to_container(cfg.component.balancer, resolve=True)
    Balancer = hydra.utils.instantiate(OmegaConf.create(balancer_cfg))

    # Dictionary to store class -> number of added samples
    

    for split_name in ds.keys():
        if split_name.endswith("_train"):
            X_augment = Balancer(ds, split_name)
            logging.info(f'Augmentation numbers: {X_augment}')
            if X_augment:
                ds[split_name] = oversample_top_per_label(
                    ds[split_name], SYNQ, X_augment
                )
            else:
                ds[split_name] = oversample_per_obser(
                    ds[split_name], SYNQ
                )

    return ds
