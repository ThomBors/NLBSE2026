from omegaconf import OmegaConf
from omegaconf import DictConfig
from dotenv import load_dotenv
from tqdm import trange
import logging
import hydra
from datasets import Dataset



def filter_synthetic(example,SyntheticQualityScore):
    # Keep original rows OR augmented rows with similarity_score > SYNQ
    if example["synthetic"] == False:
        return True
    elif example["synthetic"] == True and example["similarity_score"] is not None:
        return example["similarity_score"] > SyntheticQualityScore
    return False

def select_top_synthetic(ds, SYNQ, label=None, top_k=1):
    """Select top_k synthetic examples in ds with the exact given label (optional)."""
    # Filter by synthetic quality score
    filtered = ds.filter(filter_synthetic, fn_kwargs={"SyntheticQualityScore": SYNQ})
    
    # If label is provided, filter by exact label match
    if label is not None:
        filtered = filtered.filter(lambda ex: ex["labels"] == label)
    
    # Sort by similarity_score descending
    sorted_indices = sorted(
        range(len(filtered)),
        key=lambda i: filtered[i]["similarity_score"],
        reverse=True
    )
    
    # Select top_k
    return filtered.select(sorted_indices[:top_k])

def oversample_top_per_label(ds, SYNQ, X_augment):
    """
    Oversample top synthetic examples per label automatically inferred
    from X_augment length.
    """
    num_labels = len(X_augment)
    top_examples_all = []

    for idx, n in enumerate(X_augment):
        if n > 0:
            # Create a one-hot label vector with 1 at idx
            label = [0] * num_labels
            label[idx] = 1
            top_examples = select_top_synthetic(ds, SYNQ, label=label, top_k=n)
            if len(top_examples) > 0:
                top_examples_all.append(top_examples)

    if top_examples_all:
        ds = ds.concatenate(top_examples_all)
    
    return ds

def oversampling(cfg, ds, SYNQ):
    # Compute number of samples per label
    balancer_cfg = OmegaConf.to_container(cfg.component.balancer, resolve=True)
    Balancer = hydra.utils.instantiate(OmegaConf.create(balancer_cfg))

    for split_name in ds.keys():
        if split_name.endswith("_train"):
            # Corrected typo 'remuve' → 'remove'
            X_augment = Balancer(ds, split_name.replace("_train", ""))
            ds[split_name] = oversample_top_per_label(ds[split_name], SYNQ, X_augment)
    
    return ds
