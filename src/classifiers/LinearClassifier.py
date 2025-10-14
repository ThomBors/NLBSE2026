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


class ClassifierLinear:
    def __init__(self, classifier_type: str, modelname: str, batch_size: int, num_iterations: int):
        """
        Args:
            classifier_type (str): Type of classifier to train ("linear", etc.)
            modelname (str): Pretrained model name for SetFitModel
            batch_size (int): Batch size for training
            num_iterations (int): Number of iterations for SetFit fine-tuning
        """
        self.classifier_type = classifier_type.lower()
        self.modelname = modelname
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __call__(self,cfg, ds, SYNQ,device):

        for lang in langs:
            model = SetFitModel.from_pretrained(
                self.modelname, multi_target_strategy="multi-output"
            )

            output_dir = f"{cfg.paths.res_dir}/models/classifier/{lang}-classifier-SetFit-{cfg.component.classifier.classifier_type}/{SYNQ}"
            os.makedirs(output_dir, exist_ok=True)

            args = TrainingArguments(
                output_dir=output_dir,
                num_epochs=5 if lang == "java" else 10,
                batch_size=self.batch_size,
                num_iterations=self.num_iterations,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds[f"{lang}_train"],
                eval_dataset=ds[f"{lang}_test"],
                column_mapping={"combo": "text", "labels": "label"},
            )

            trainer.train()
            trainer.model.save_pretrained(output_dir)
            # Explicit cleanup
            del trainer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
