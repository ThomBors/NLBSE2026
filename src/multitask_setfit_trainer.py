# multitask_setfit_trainer.py
import os
import gc
import torch
import torch.nn as nn
from setfit import SetFitModel, Trainer, TrainingArguments

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


# ----------------------------------------------------------------------
# 1. Custom multi-task head
# ----------------------------------------------------------------------
class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, task_dims):
        """
        input_dim: embedding size from SetFit backbone
        task_dims: dict {task_name: num_classes}
        """
        super().__init__()
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_dims.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_classes),
            )

    def forward(self, x):
        # returns a dict of logits per task
        return {task: head(x) for task, head in self.task_heads.items()}


# ----------------------------------------------------------------------
# 2. Custom Trainer (HF-compatible)
# ----------------------------------------------------------------------
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, task_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_names = task_names
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        HF Trainer calls this every step.
        We override it to compute a summed multi-task loss.
        """
        # inputs: dict with "combo" (text) and "labels" (dict of task labels)
        texts = inputs["combo"]
        labels_dict = inputs["labels"]

        # Encode text → embeddings
        embeddings = model.model_body.encode(texts)
        outputs = model.model_head(embeddings)

        # Aggregate multi-task loss
        total_loss = 0
        for task_name in self.task_names:
            logits = outputs[task_name]
            labels = labels_dict[task_name]
            total_loss += self.loss_fn(logits, labels)

        return (total_loss, outputs) if return_outputs else total_loss


# ----------------------------------------------------------------------
# 3. Main classifier training entrypoint
# ----------------------------------------------------------------------
def classifiers(cfg, ds, SYNQ, langs):
    for lang in langs:
        print(f"\n--- Training language: {lang} ---")
        model = SetFitModel.from_pretrained(
            cfg.component.classifier.modelname,
            multi_target_strategy="multi-output",
        )

        # Example multi-task setup (customize to your case)
        input_dim = model.model_head.in_features
        task_dims = {
            "sentiment": 2,  # binary task
            "topic": 5,      # 5-class task
        }
        model.model_head = MultiTaskHead(input_dim, task_dims)

        output_dir = f"{cfg.paths.res_dir}/models/classifier/{lang}-classifier-SetFit/{SYNQ}"
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            num_epochs=5 if lang == "java" else 10,
            batch_size=cfg.component.classifier.batch_size,
            num_iterations=cfg.component.classifier.num_iterations,
        )

        trainer = MultiTaskTrainer(
            model=model,
            args=args,
            train_dataset=ds[f"{lang}_train"],
            eval_dataset=ds[f"{lang}_test"],
            column_mapping={"combo": "text", "labels": "label"},
            task_names=list(task_dims.keys()),
        )

        trainer.train()
        trainer.evaluate()
        trainer.model.save_pretrained(output_dir)

        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
