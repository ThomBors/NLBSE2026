import hydra
import pandas as pd
import logging
from omegaconf import OmegaConf



def filter_synthetic(example, SyntheticQualityScore):
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

    # Sort by similarity_score increasing (lower similarity first)
    sorted_indices = sorted(
        range(len(filtered)),
        key=lambda i: filtered[i]["similarity_score"],
        reverse=False,
    )


    # Select top_k
    return filtered.select(sorted_indices[:top_k])


def oversample_top_per_label(ds, SYNQ, X_augment, report_dict):
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
                report_dict[idx] = report_dict.get(idx, 0) + len(top_examples)

    if top_examples_all:
        ds = ds.concatenate(top_examples_all)

    return ds, report_dict


def oversampling(cfg, ds, SYNQ):
    balancer_cfg = OmegaConf.to_container(cfg.component.balancer, resolve=True)
    Balancer = hydra.utils.instantiate(OmegaConf.create(balancer_cfg))

    # Dictionary to store class -> number of added samples
    report_dict = {}

    for split_name in ds.keys():
        if split_name.endswith("_train"):
            X_augment = Balancer(ds, split_name)
            logging.info(f'Augmentation numbers: {X_augment}')
            ds[split_name], report_dict = oversample_top_per_label(
                ds[split_name], SYNQ, X_augment, report_dict
            )

    report_df = pd.DataFrame([
        {"class_index": class_idx, "num_added_samples": count}
        for class_idx, count in report_dict.items()
    ])

    return ds,report_df
