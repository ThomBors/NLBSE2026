from datasets import Dataset, DatasetDict, concatenate_datasets
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import math

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(original_sentence, new_sentence):
    embeddings = similarity_model.encode([original_sentence, new_sentence], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score

def predict_masked_token(model, tokenizer, input_ids):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([input_ids])
        outputs = model(inputs)
        predictions = outputs.logits
        masked_index = (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        predicted_id = predictions[0, masked_index].argmax().item()
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_id)
    return predicted_token

def predict_masked_token_topn(model, tokenizer, input_ids, n=5):
    """
    Predict the masked token by randomly sampling from the top-n predictions.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([input_ids])
        outputs = model(inputs)
        logits = outputs.logits
        masked_index = (inputs == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        # Get top-n predicted token IDs and their probabilities
        topn_logits = torch.topk(logits[0, masked_index], n)
        topn_ids = topn_logits.indices.tolist()

        # Randomly choose one of the top-n tokens
        predicted_id = random.choice(topn_ids)
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_id)

    return predicted_token

def mask_one_token(input_ids, labels, tokenizer):
    if len(labels) < 2:
        return input_ids, None
    multi_label_positions = [i for i, l in enumerate(labels) if l > 1]
    if not multi_label_positions:
        return input_ids, None
    mask_idx = random.choice(multi_label_positions)
    original_id = input_ids[mask_idx]
    input_ids[mask_idx] = tokenizer.mask_token_id
    return input_ids, (mask_idx, original_id)

def augment_example(example, model, tokenizer, x_augments=3):
    generated_sentences = set()
    attempts = 0
    max_attempts = x_augments * 3  # avoid infinite loops

    while len(generated_sentences) < x_augments and attempts < max_attempts:
        attempts += 1

        # Tokenize sentence to get input_ids
        encoding = tokenizer(example["combo"], return_tensors="pt")
        input_ids = encoding["input_ids"][0].tolist()

        masked_input_ids, mask_info = mask_one_token(input_ids.copy(), example["labels"], tokenizer)
        if mask_info is None:
            break
        mask_idx, original_id = mask_info
        predicted_token = predict_masked_token_topn(model, tokenizer, masked_input_ids)
        original_token = tokenizer.convert_ids_to_tokens(original_id)

        if predicted_token != original_token:
            # Replace mask with prediction
            masked_input_ids[mask_idx] = tokenizer.convert_tokens_to_ids(predicted_token)
            new_sentence = tokenizer.decode(masked_input_ids, skip_special_tokens=True)

            # Check uniqueness
            if new_sentence != example["combo"] and new_sentence not in generated_sentences:
                generated_sentences.add(new_sentence)

    augmented_list = []
    for sent in generated_sentences:
        score = compute_similarity(example["combo"], sent)
        augmented_list.append({
            "index": example["index"],
            "class": example["class"],
            "comment_sentence": example["comment_sentence"],
            "partition": example["partition"],
            "combo": sent,
            "labels": example["labels"],
            "similarity_score": score
        })

    return augmented_list

def augment_language_multiple(ds_lang, model, tokenizer, x_augments=3):
    augmented_examples = []
    for example in ds_lang:
        augmented_examples.extend(augment_example(example, model, tokenizer, x_augments=x_augments))
    return Dataset.from_list(augmented_examples)


def run_augmentation_pipeline(ds):
    # ------------------------
    # Loop through languages and create DatasetDict
    # ------------------------
    augmented_datasets = DatasetDict()

    for lang in ["java", "pharo", "python"]:
        model_name = f"../models/{lang}-finetuned-ModernBert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Original training dataset
        train_ds = ds[f"{lang}_train"]
        
        # Add similarity_score (NaN) and synthetic (False) columns for original data
        train_ds = train_ds.add_column("similarity_score", [math.nan] * len(train_ds))
        train_ds = train_ds.add_column("synthetic", [False] * len(train_ds))
        
        # Generate augmented examples
        augmented_train = augment_language_multiple(train_ds, model, tokenizer, x_augments=3)
        
        # Add synthetic=True for augmented examples
        augmented_train = augmented_train.add_column("synthetic", [True] * len(augmented_train))
        
        # Concatenate original + augmented datasets
        combined_train = concatenate_datasets([train_ds, augmented_train])

        augmented_datasets[f"{lang}_train"] = combined_train
        augmented_datasets[f"{lang}_test"] = ds[f"{lang}_test"]  # keep original test

    augmented_datasets.save_to_disk("datasets/augmented_datasets")
