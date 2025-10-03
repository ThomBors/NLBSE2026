import logging
import random
import os
from pathlib import Path
import pandas as pd
from datasets import concatenate_datasets
import numpy as np
import torch


def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["combo"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


# def tokenize_function(batch, tokenizer):
#     # batch["combo"] is a list of strings
#     result = tokenizer(
#         batch["combo"],
#         padding="max_length",
#         truncation=True,
#         return_tensors=None  # Let datasets handle list outputs
#     )

#     # word_ids only works for fast tokenizer and single encoding at a time
#     if tokenizer.is_fast:
#         # create word_ids for each sentence
#         result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

#     return result


def group_texts(examples, chunk_size=64):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def title():
    logging.info(
        """
        
 ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗███████╗████████╗██╗ ██████╗    
 ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔════╝╚══██╔══╝██║██╔════╝    
 ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║█████╗     ██║   ██║██║        
 ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██╔══╝     ██║   ██║██║        
 ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║███████╗   ██║   ██║╚██████╗   
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝     

  ██████╗ ██╗   ██╗███████╗██████╗ ███████╗ █████╗ ███╗   ███╗██████╗ ██╗     ██╗███╗   ██╗ ██████╗                  
 ██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██║████╗  ██║██╔════╝          
 ██║   ██║██║   ██║█████╗  ██████╔╝███████╗███████║██╔████╔██║██████╔╝██║     ██║██╔██╗ ██║██║  ███╗  
 ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██║██║╚██╗██║██║   ██║         
 ╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║     ███████╗██║██║ ╚████║╚██████╔╝         
  ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝     
                                                                          
    """
    )


def labels_and_synthetic_csv(data, lang, labels, SYNQ, report_rows):
    for label in labels[lang]:
        # 2x2 counters
        counts = {
            "synthetic_pos": 0,
            "synthetic_neg": 0,
            "real_pos": 0,
            "real_neg": 0,
        }
        similarity_scores = []

        for d in data:
            if d["synthetic"]:
                if d[label] == 1:
                    similarity_scores.append(d.get("similarity_score", 0))
                    counts["synthetic_pos"] += 1
                else:
                    counts["synthetic_neg"] += 1
            else:
                if d[label] == 1:
                    counts["real_pos"] += 1
                else:
                    counts["real_neg"] += 1

        total_pos = counts["synthetic_pos"] + counts["real_pos"]
        total_neg = counts["synthetic_neg"] + counts["real_neg"]
        total = total_pos + total_neg

        mean_similarity = pd.NA
        sd_similarity = pd.NA
        if similarity_scores:
            mean_similarity = sum(similarity_scores) / len(similarity_scores)
            sd_similarity = pd.Series(similarity_scores).std()

        report_rows.append(
            {
                "language": lang,
                "label": label,
                "num_positive": total_pos,
                "num_negative": total_neg,
                "num_synthetic_positive": counts["synthetic_pos"],
                "num_synthetic_negative": counts["synthetic_neg"],
                "num_real_positive": counts["real_pos"],
                "num_real_negative": counts["real_neg"],
                "mean_similarity_synthetic": mean_similarity,
                "sd_similarity_synthetic": sd_similarity,
                "SYNQ": SYNQ,
            }
        )


def split_list_into_columns(row, lang):
    labels = {
        "java": [
            "summary",
            "Ownership",
            "Expand",
            "usage",
            "Pointer",
            "deprecation",
            "rational",
        ],
        "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
        "pharo": [
            "Keyimplementationpoints",
            "Example",
            "Responsibilities",
            "Intent",
            "Keymessages",
            "Collaborators",
        ],
    }
    values_list = row["labels"]  # Replace 'values' with your actual column name
    dict = {}
    for key in labels[lang]:

        dict[key] = values_list[labels[lang].index(key)]

    return dict


def generate_label_statistics(cfg, ds, SYNQ, output_file_name="label_statistics.csv"):
    """
    Generate per-label statistics for oversampled dataset and save to CSV.
    """
    output_dir = Path(cfg.paths.res_dir) / "performance" / SYNQ
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / output_file_name

    langs = ["java", "python", "pharo"]
    labels = {
        "java": [
            "summary",
            "Ownership",
            "Expand",
            "usage",
            "Pointer",
            "deprecation",
            "rational",
        ],
        "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
        "pharo": [
            "Keyimplementationpoints",
            "Example",
            "Responsibilities",
            "Intent",
            "Keymessages",
            "Collaborators",
        ],
    }

    report_rows = []

    # Convert dataset labels into dicts for easier access
    for lang in langs:
        ds_split = concatenate_datasets([ds[f"{lang}_train"]]).map(
            lambda row: split_list_into_columns(row, lang)
        )
        labels_and_synthetic_csv(ds_split, lang, labels, SYNQ, report_rows)

    # Create DataFrame and save CSV
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_csv_path, index=False)
    print(f"Label statistics CSV saved to {output_csv_path}")

    return report_df
