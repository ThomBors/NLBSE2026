# expects that the folder models exists
from setfit import SetFitModel, Trainer, TrainingArguments
from tqdm.auto import tqdm
import os
tqdm.pandas()
import gc
import torch

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


def classifiers(cfg, ds,SYNQ):

    for lang in langs:
        model = SetFitModel.from_pretrained(
            cfg.component.classifier.modelname, multi_target_strategy="multi-output"
        )

        output_dir = f"{cfg.paths.res_dir}/models/{lang}-classifier-SetFit/{SYNQ}"
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            num_epochs=5 if lang == "java" else 10,
            batch_size=cfg.component.classifier.batch_size,
            num_iterations=cfg.component.classifier.num_iterations,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds[f"{lang}_train"],
            eval_dataset=ds[f"{lang}_test"],
            column_mapping={"combo": "text", "labels": "label"},
        )

        trainer.train()
        trainer.model.save_pretrained(
            output_dir
        )
        # Explicit cleanup
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
