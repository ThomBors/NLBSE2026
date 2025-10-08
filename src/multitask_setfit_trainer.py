# multitask_setfit_trainer.py
import os
import gc
import torch
import torch.nn as nn
from setfit import SetFitModel, Trainer, TrainingArguments
from src.GradSurg.weight_methods import WeightMethods

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
    def __init__(self, input_dim, n_task):
        """
        input_dim: embedding size from SetFit backbone
        task_dims: dict {task_name: num_classes}
        """
        super().__init__()
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2),
                )
                for _ in range(n_task)
            ]
        )

    def forward(self, x):
        # returns a dict of logits per task
        y_ = [head(x) for head in self.task_heads]
        return [torch.softmax(logits,dim=1) for logits in y_]


# ----------------------------------------------------------------------
# 2. Custom Trainer (HF-compatible)
# ----------------------------------------------------------------------
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, n_task=None,weight_method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_task = n_task
        self.loss_fn = nn.CrossEntropyLoss()
        self.weight_method = weight_method

    def compute_loss(self, model, inputs, return_outputs=False):
        texts = inputs["combo"]
        y = inputs["labels"]

        embeddings = model.model_body.encode(texts)
        y_ = model.model_head(embeddings)

        # Compute task-wise losses
        losses = torch.stack([
                    self.loss_fn(y_pred, y[:, i])  # select labels for task i
                    for i, y_pred in enumerate(y_)
                    ])
        
        # Call your custom backward
        loss, _ = self.weight_method.backward(
            losses=losses,
            shared_parameters=list(model.model_body()),
            task_specific_parameters=list(model.model_head())
        )

        # Return a scalar to satisfy Trainer API (detach so no second backward)
        dummy_loss = loss.detach().sum()  

        if return_outputs:
            return dummy_loss, y_
        return dummy_loss
    

# ----------------------------------------------------------------------
# 3. Main classifier training entrypoint
# ----------------------------------------------------------------------
def classifiers(cfg, ds, SYNQ,device='cpu'):
    for lang in langs:
        print(f"\n--- Training language: {lang} ---")
        model = SetFitModel.from_pretrained(
            cfg.component.classifier.modelname,
            multi_target_strategy="multi-output",
        )

        # Example multi-task setup (customize to your case)
        input_dim = model.model_head.in_features
        n_tasks = len(labels[lang])
        model.model_head = MultiTaskHead(input_dim, n_tasks)

        weight_method = WeightMethods(
            method=cfg.optimization.method,
            n_tasks=n_tasks,
            device=device,
        )

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
            n_task=n_tasks,
            weight_method = weight_method
        )

        trainer.train()
        trainer.evaluate()
        trainer.model.save_pretrained(output_dir)

        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

