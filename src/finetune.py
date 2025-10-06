import logging
import math
import os

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ModernBertForMaskedLM,
    Trainer,
    TrainingArguments,
)

from src.utils import group_texts


def createMLforWCft(cfg, ds, langs, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.component.finetune.modelname)
    model = ModernBertForMaskedLM.from_pretrained(cfg.component.finetune.modelname).to(
        device
    )

    # Use batched=True to activate fast multithreading!
    def tokenize_function(examples):
        result = tokenizer(examples["combo"])
        if tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=[
            "index",
            "class",
            "comment_sentence",
            "partition",
            "combo",
            "labels",
        ],
    )

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    lm_tvt_dataset = DatasetDict()
    for l in langs:
        split = lm_datasets[f"{l}_train"].train_test_split(
            test_size=cfg.trainer.ValidationSize, seed=cfg.seed
        )

        # Add to the new dict
        lm_tvt_dataset[f"{l}_train"] = split["train"]
        lm_tvt_dataset[f"{l}_val"] = split["test"]

        # Keep the original test set as is
        lm_tvt_dataset[f"{l}_test"] = lm_datasets[f"{l}_test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=cfg.component.finetune.mlm_probability
    )

    for l in langs:
        model_dir = f"{cfg.paths.res_dir}/models/finetune/{l}-finetuned-ModernBert"

        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            logging.info(
                f"Skipping training for {l}, model already exists at {model_dir}"
            )
            continue

        logging_steps = (
            len(lm_tvt_dataset[f"{l}_train"]) // cfg.component.finetune.batch_size
        )
        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="no",
            logging_dir=f"{cfg.paths.log_dir}/models/logs",
            learning_rate=cfg.component.finetune.learning_rate,
            weight_decay=cfg.component.finetune.weight_decay,
            num_train_epochs=cfg.component.finetune.epochs,
            per_device_train_batch_size=cfg.component.finetune.batch_size,
            per_device_eval_batch_size=cfg.component.finetune.batch_size,
            push_to_hub=False,
            fp16=True,
            logging_steps=logging_steps,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_tvt_dataset[f"{l}_train"],
            eval_dataset=lm_tvt_dataset[f"{l}_val"],
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        eval_results = trainer.evaluate()
        logging.info(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        trainer.train()
        eval_results = trainer.evaluate()
        logging.info(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
