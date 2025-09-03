from transformers import AutoTokenizer, ModernBertForMaskedLM
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, load_dataset
import random
import math
import torch
import os

from src.utils import (tokenize_function,group_texts)

def createMLforWCft(ds,langs,device,batch_size = 64):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base").to(device)

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=['index', 'class', 'comment_sentence', 'partition', 'combo', 'labels']
    )

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    lm_tvt_dataset = DatasetDict()
    for l in langs:
        split = lm_datasets[f"{l}_train"].train_test_split(test_size=0.2, seed=42)
            
        # Add to the new dict
        lm_tvt_dataset[f"{l}_train"] = split["train"]
        lm_tvt_dataset[f"{l}_val"] = split["test"]
        
        # Keep the original test set as is
        lm_tvt_dataset[f"{l}_test"] = lm_datasets[f"{l}_test"]
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    for l in langs:
        logging_steps = len(lm_tvt_dataset[f'{l}_train']) // batch_size
        training_args = TrainingArguments(
            output_dir=f"../models/{l}-finetuned-ModernBert",
            overwrite_output_dir=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="models/logs",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            push_to_hub=False,
            fp16=True,
            logging_steps=logging_steps,
            report_to='none'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_tvt_dataset[f"{l}_train"],
            eval_dataset=lm_tvt_dataset[f"{l}_val"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        trainer.train()
        eval_results = trainer.evaluate()
        print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")