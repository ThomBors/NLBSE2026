# expects that the folder models exists
import pandas as pd
import time
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
import numpy as np
import torch
tqdm.pandas()



langs = ['java', 'python', 'pharo']
labels = {
    'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
    'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
    'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Intent', 'Keymessages', 'Collaborators']
}

def classifiers(cfg,ds):
    print(ds)
    for lang in langs:
        model = SetFitModel.from_pretrained(cfg.component.classifier.modelname, 
                                            multi_target_strategy="multi-output")

        args = TrainingArguments(
            num_epochs=5 if lang == 'java' else 10,
            batch_size=cfg.component.classifier.batch_size,
            num_iterations=cfg.component.classifier.num_iterations
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds[f'{lang}_train'],
            eval_dataset=ds[f'{lang}_test'],
            column_mapping={"combo": "text", "labels": "label"}
        )

        trainer.train()
        trainer.model.save_pretrained(f'{cfg.paths.res_dir}/models/{lang}-classifier-SetFit')
