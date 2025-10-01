import logging
import os
import random

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from difflib import SequenceMatcher

def compute_similarity(similarity_model, original_sentence, new_sentence):
    embeddings = similarity_model.encode(
        [original_sentence, new_sentence], convert_to_tensor=True
    )
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Normalize to [0,1] if requested
    score = (score + 1) / 2 
    return score


def predict_masked_token(model, tokenizer, input_ids):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([input_ids])
        outputs = model(inputs)
        predictions = outputs.logits
        masked_index = (
            (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        )
        predicted_id = predictions[0, masked_index].argmax().item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_id)
    return predicted_token


def mask_one_token(input_ids, tokenizer):
    """Randomly replace one non-special token with [MASK]."""
    # Avoid masking special tokens
    candidate_positions = [
        i
        for i, tok_id in enumerate(input_ids)
        if tok_id
        not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    ]
    if not candidate_positions:
        return input_ids, None
    mask_idx = random.choice(candidate_positions)
    original_id = input_ids[mask_idx]
    input_ids[mask_idx] = tokenizer.mask_token_id
    return input_ids, (mask_idx, original_id)


def predict_masked_token_topn(model, tokenizer, input_ids, mask_idx, n=10):
    """Predict a replacement token for the [MASK] at `mask_idx`."""
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([input_ids])
        outputs = model(inputs)
        logits = outputs.logits

        # top-n predictions for the masked index
        topn_logits = torch.topk(logits[0, mask_idx], n)
        topn_ids = topn_logits.indices.tolist()

        # randomly select one
        predicted_id = random.choice(topn_ids)
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_id)

    return predicted_id, predicted_token

def is_too_similar(a, b, threshold=0.99):
    """
    Returns True if a and b are too similar (normalized).
    threshold=0.9 means 90% similar -> considered duplicate
    """
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() > threshold

def augment_example(cfg, example, model, similarity_model, tokenizer, x_augments=5):
    generated_sentences = set()
    attempts = 0
    max_attempts = x_augments * 3

    while len(generated_sentences) < x_augments and attempts < max_attempts:
        attempts += 1

        encoding = tokenizer(example["combo"], return_tensors="pt")
        input_ids = encoding["input_ids"][0].tolist()

        if np.array(example["labels"]).sum() >= 1:
            masked_input_ids, mask_info = mask_one_token(input_ids.copy(), tokenizer)
            if mask_info is None:
                continue
            mask_idx, original_id = mask_info

            predicted_id, predicted_token = predict_masked_token_topn(
                model, tokenizer, masked_input_ids, mask_idx, cfg.component.augment.topn
            )

            # replace the mask
            new_ids = masked_input_ids.copy()
            new_ids[mask_idx] = predicted_id
            new_sentence = tokenizer.decode(new_ids, skip_special_tokens=True)

            # Normalize for duplicate checking
            normalized_new = new_sentence.lower()

            # Skip if exact or nearly identical to original or already generated
            if (normalized_new != example["combo"].lower()
                and not any(is_too_similar(normalized_new, s) for s in generated_sentences)):
                generated_sentences.add(normalized_new)

    augmented_list = []
    for sent in generated_sentences:
        score = compute_similarity(similarity_model, example["combo"], sent)
        augmented_list.append(
            {
                "index": example["index"],
                "class": example["class"],
                "comment_sentence": example["comment_sentence"],
                "partition": example["partition"],
                "combo": sent,
                "labels": example["labels"],
                "similarity_score": score,
                "synthetic": True,
            }
        )
    return augmented_list


def augment_language_multiple(
    cfg, ds_lang, model, similarity_model, tokenizer, x_augments=5
):
    augmented_examples = []
    for example in ds_lang:
        augmented_examples.extend(
            augment_example(
                cfg, example, model, similarity_model, tokenizer, x_augments=x_augments
            )
        )
    return Dataset.from_list(augmented_examples)


def run_augmentation_pipeline(cfg, ds):
    # ------------------------
    # Loop through languages and create DatasetDict
    # ------------------------
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    similarity_model = SentenceTransformer(cfg.component.augment.modelname)
    augmented_datasets = DatasetDict()

    for lang in tqdm(["java", "pharo", "python"], desc="Languages"):

        if os.path.exists(
            f"{cfg.paths.data_dir}/augmented_datasets/{lang}_train"
        ) and os.path.isdir(f"{cfg.paths.data_dir}/augmented_datasets/{lang}_train"):
            logging.info(
                f"Skipping Augmentation Pipeline for {lang}, data already exists at {cfg.paths.data_dir}/augmented_datasets/{lang}_train"
            )
            continue

        model_name = f"{cfg.paths.res_dir}/models/{lang}-finetuned-ModernBert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Original training dataset
        train_ds = ds[f"{lang}_train"]

        # Add similarity_score (NaN) and synthetic (False) columns for original data
        train_ds = train_ds.add_column("similarity_score", [1.0] * len(train_ds))
        train_ds = train_ds.add_column("synthetic", [False] * len(train_ds))

        # Generate augmented examples
        augmented_train = augment_language_multiple(
            cfg,
            train_ds,
            model,
            similarity_model,
            tokenizer,
            x_augments=cfg.component.augment.augments,
        )

        if len(augmented_train) > 0:
            augmented_train = Dataset.from_list(
                augmented_train, features=train_ds.features
            )
            combined_train = concatenate_datasets([train_ds, augmented_train])
        else:
            combined_train = train_ds
            logging.error("no augmented data")

        # Concatenate original + augmented datasets
        combined_train = concatenate_datasets([train_ds, augmented_train])

        augmented_datasets[f"{lang}_train"] = combined_train
        augmented_datasets[f"{lang}_test"] = ds[f"{lang}_test"]  # keep original test

        augmented_datasets.save_to_disk(f"{cfg.paths.data_dir}/augmented_datasets")
