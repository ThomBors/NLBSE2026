import logging
import time

import numpy as np
import pandas as pd
import torch
from src.classifiers.MtlClassifier import MTLSetFitModel




class EvaluationMtl:
    def __init__(self,evaluation_type):
        self.langs = ["java", "python", "pharo"] 
        self.labels = {
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


    def __call__(self, cfg, ds, SYNQ):
        total_flops = 0
        total_time = 0
        scores = []
        for lan in self.langs:
            # to load trained models:
            model = MTLSetFitModel.from_pretrained(
                f"{cfg.paths.res_dir}/models/classifier/{lan}-classifier-SetFit/{SYNQ}"
            )
            # Check encoder weights
            for name, param in model.model_body.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in {name}")

            # Check head weights
            for i, head in enumerate(model.model_head.task_heads):
                for name, param in head.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN in head {i}, {name}")

            with torch.profiler.profile(with_flops=True) as p:
                x = ds[f"{lan}_test"][:]["combo"]
                begin = time.time()
                for i in range(10):
                    y_pred = model(x)
                    y_pred = [torch.argmax(out, dim=1) for out in y_pred] #-->check for mtl
                    y_pred = torch.stack(y_pred, dim=0)
                total = time.time() - begin
                total_time = total_time + total
            total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)
            y_true = np.array(ds[f"{lan}_test"]["labels"]).T
            y_pred= y_pred.detach().cpu().numpy().T
            for i in range(len(y_pred)):
                assert len(y_pred[i]) == len(y_true[i])
                tp = sum([true == pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
                tn = sum([true == pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])
                fp = sum(
                    [true == 0 and pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])]
                )
                fn = sum(
                    [true == 1 and pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])]
                )
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = (2 * tp) / (2 * tp + fp + fn)
                scores.append(
                    {
                        "lan": lan,
                        "cat": self.labels[lan][i],
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "SYNQ": SYNQ,
                    }
                )
        logging.info(f"Compute in GFLOPs: {total_flops/10}")
        logging.info(f"Avg runtime in seconds: {total_time/10}")
        compute = {"GFLOPs": total_flops / 10, "runtime": total_time / 10}

        return pd.DataFrame(scores), pd.DataFrame([compute])