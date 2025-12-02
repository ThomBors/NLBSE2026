# multitask_setfit_trainer.py
import os
import gc
import torch
import torch.nn as nn
from setfit import Trainer, TrainingArguments
from setfit import SetFitModel as BaseSetFitModel
from src.GradSurg.weight_methods import WeightMethods
from typing import List, Union, Optional
from adabelief_pytorch import AdaBelief
import setfit
from tqdm.asyncio import trange, tqdm

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
    def __init__(self, input_dim, n_task, init="xavier", generator=None):
        super().__init__()
        self.n_task = n_task
        self.init = init
        self.generator = generator

        self.shared_base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            )
        
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(128, 128),
                    nn.BatchNorm1d(num_features=128),
                    nn.LeakyReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 2),  # binary classification per task
                )
                for _ in range(n_task)
            ]
        )
    # apply init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init == "orthogonal":
                nn.init.orthogonal_(m.weight, generator=self.generator)
            elif self.init == "xavier":
                nn.init.xavier_uniform_(m.weight, generator=self.generator)
            elif self.init == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu", generator=self.generator)
            else:
                nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: dict) -> dict:
        """
        x: dict output from SetFit encoder, expects 'sentence_embedding' key
        Returns: dict with 'logits' key, shape [batch_size, n_tasks, n_classes]
        """
        embeddings = x["sentence_embedding"]
        h = self.shared_base(embeddings)
        logits = torch.stack([head(h) for head in self.task_heads], dim=1)
        return {"logits": logits}  # shape [batch, n_tasks, n_classes]
    
    def shared_parameters(self):
        return (p for p in self.shared_base.parameters())

    def shared_parameters_named(self):
        return ((name, p) for name, p in self.shared_base.named_parameters())

    def task_specific_parameters(self):
        return_list = []
        for task in range(len(self.task_heads)):
            return_list += [p for p in self.task_heads[task].parameters()]
        return return_list


    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.shared_base(embeddings)
            probs = torch.stack([head(h) for head in self.task_heads], dim=1)
        return torch.softmax(probs,dim=2)  # probabilities

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            proba = self.predict_proba(embeddings)
        return proba




# ----------------------------------------------------------------------
# 2. Custom SetFitModel Trainer loop
# ----------------------------------------------------------------------
class MTLSetFitModel(setfit.SetFitModel):
    def __init__(
        self,
        *args,
        weight_method=None,
        n_task=1,
        **kwargs,
    ):
        """
        Initializes a model class with support for static or dynamic weighted loss strategies.

        This constructor:
            - Passes all positional and keyword arguments to the superclass
            - Stores the loss strategy and whether it is dynamic
            - Initializes and moves the custom class weights tensor to the appropriate device
            - Stores the number of output classes

        Parameters:
            *args: Positional arguments passed to the parent class.
            custom_loss_weight (Iterable[float] or Tensor): Class-specific weights for the loss function.
            weighted_loss_strategy (str or Callable): The selected loss weighting strategy (e.g., "EW", "ICF", or a callable).
            num_classes (int): The number of output classes.
            isDynamic (bool): Whether the model uses a dynamic weighting strategy.
            **kwargs: Additional keyword arguments passed to the parent class.

        Returns:
            None
        """

        super().__init__(*args, **kwargs)
        self.weight_method = weight_method
        self.n_task = n_task


    def fit(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        num_epochs: int,
        batch_size: Optional[int] = None,
        body_learning_rate: Optional[float] = None,
        head_learning_rate: Optional[float] = None,
        end_to_end: bool = True,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """Train the classifier head, only used if a differentiable PyTorch head is used.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            num_epochs (`int`): The number of epochs to train for.
            batch_size (`int`, *optional*): The batch size to use.
            body_learning_rate (`float`, *optional*): The learning rate for the `SentenceTransformer` body
                in the `AdamW` optimizer. Disregarded if `end_to_end=False`.
            head_learning_rate (`float`, *optional*): The learning rate for the differentiable torch head
                in the `AdamW` optimizer.
            end_to_end (`bool`, defaults to `False`): If True, train the entire model end-to-end.
                Otherwise, freeze the `SentenceTransformer` body and only train the head.
            l2_weight (`float`, *optional*): The l2 weight for both the model body and head
                in the `AdamW` optimizer.
            max_length (`int`, *optional*): The maximum token length a tokenizer can generate. If not provided,
                the maximum length for the `SentenceTransformer` body is used.
            show_progress_bar (`bool`, defaults to `True`): Whether to display a progress bar for the training
                epochs and iterations.
        """
        
        self.model_body.train()
        self.model_head.train()
        if not end_to_end:
            self.freeze("body")

        dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
        criterion = nn.CrossEntropyLoss()
        optimizer = self.__prepare_optimizer__(head_learning_rate, body_learning_rate, l2_weight)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch_idx in trange(num_epochs, desc="Epoch", disable=not show_progress_bar):
            for batch in tqdm(dataloader, desc="Iteration", disable=not show_progress_bar, leave=False):
                features, labels = batch
                optimizer.zero_grad()

                # to model's device
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device).long()

                outputs = self.model_body(features)
                if self.normalize_embeddings:
                    outputs["sentence_embedding"] = nn.functional.normalize(
                        outputs["sentence_embedding"], p=2, dim=1
                    )
                outputs = self.model_head(outputs)
                logits = outputs["logits"]

                losses = torch.stack([criterion(logits[:, i], labels[:, i]) for i in range(self.n_task)])
                # Custom weighting/backprop
                loss, _ = self.weight_method.backward(
                    losses=losses,
                    shared_parameters=list(self.model_head.shared_parameters()),
                    task_specific_parameters=list(self.model_head.task_specific_parameters())
                )
                optimizer.step()
            scheduler.step()

        if not end_to_end:
            self.unfreeze("body")

    def __prepare_optimizer__(
        self,
        head_learning_rate: float,
        body_learning_rate: Optional[float],
        l2_weight: float,
    ) -> torch.optim.Optimizer:
        body_learning_rate = body_learning_rate or head_learning_rate
        l2_weight = l2_weight or 1e-2
        optimizer = AdaBelief(
            [
                {
                    "params": self.model_body.parameters(),
                    "lr": body_learning_rate,
                    "weight_decay": l2_weight,
                },
                {"params": self.model_head.parameters(), "lr": head_learning_rate, "weight_decay": l2_weight},
            ],
        )

        return optimizer
    

# ----------------------------------------------------------------------
# 3. Main classifier training entrypoint
# ----------------------------------------------------------------------
class ClassifierMtl:
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

    def __call__(self,cfg, ds, SYNQ,device='cpu'):
        for lang in langs: 
            print(f"\n--- Training language: {lang} ---")
            n_tasks = len(labels[lang])
            weight_method = WeightMethods(
                method=cfg.optimization.method,
                n_tasks=n_tasks,
                device=device,
            )
            base_model = BaseSetFitModel.from_pretrained(
                self.modelname,
                multi_target_strategy="multi-output",
            )

            # Example multi-task setup (customize to your case)
            input_dim = base_model.model_body.get_sentence_embedding_dimension()
            model = MTLSetFitModel(
                model_body=base_model.model_body,
                model_head=MultiTaskHead(input_dim, n_tasks).to(device),
                weight_method=weight_method,
                n_task=n_tasks
            )
           
            
            output_dir = f"{cfg.paths.res_dir}/models/classifier/{lang}-classifier-SetFit-{cfg.component.classifier.classifier_type}/{SYNQ}"
            os.makedirs(output_dir, exist_ok=True)


            args = TrainingArguments(
                output_dir=output_dir,
                num_epochs=5 if lang == "java" else 10, 
                batch_size=self.batch_size,
                num_iterations=self.num_iterations,
                head_learning_rate=1e-3
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds[f"{lang}_train"],
                eval_dataset=ds[f"{lang}_test"],
                column_mapping={"combo": "text", "labels": "label"}
            )
 
            trainer.train()
            trainer.model.save_pretrained(output_dir)

            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

