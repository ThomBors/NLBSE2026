from datasets import Dataset, DatasetDict, load_dataset,load_from_disk
from setfit import SetFitModel, Trainer, TrainingArguments
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import rootutils
import logging
import torch
import hydra
import time
tqdm.pandas()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import set_seed,set_logger
from src.finetune import createMLforWCft
from src.augment import run_augmentation_pipeline
from src.classification import classifiers
from src.utils import filter_synthetic

set_logger()

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() & cfg.trainerHardwer.use_cuda else "cpu")
    logging.info("Device:", device)

    set_seed(cfg.seed)

    langs = ['java', 'python', 'pharo']
    labels = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Intent', 'Keymessages', 'Collaborators']
    }
    ds = load_dataset('NLBSE/nlbse26-code-comment-classification')

    # --- fine tune ModernBERT for augmentation --- #
    createMLforWCft(cfg,ds,langs,device,batch_size = 64)

    # --- Synthetic Augmentation --- #
    run_augmentation_pipeline(cfg,ds)

    # --- Load new Augmentd Data --- #
    dsplus = load_from_disk(f"{cfg.paths.data_dir}/augmented_datasets")

    # --- Set Syntetic Quality --- #
    SYNQ = cfg.experiment.setfit.trainer.SYNQ
    for split_name in dsplus.keys():
        if split_name.endswith("_train"):
            dsplus[split_name] = dsplus[split_name].filter(filter_synthetic(SYNQ))

    # --- Code Commente Classification --- #
    classifiers(cfg,dsplus)

    # --- Test Pipeline --- #
    total_flops = 0
    total_time = 0
    scores = []
    for lan in langs:
        # to load trained models:
        model = SetFitModel.from_pretrained(f'{cfg.paths.res_dir}/models/{lan}-SetFit')
        # to load pretrained models from Hub:
        # model = SetFitModel.from_pretrained(f'NLBSE/nlbse26_{lan}')
        with torch.profiler.profile(with_flops=True) as p:
            x = dsplus[f'{lan}_test'][:]["combo"]
            begin = time.time()
            for i in range(10):
            y_pred = model(x)
            y_pred = np.asarray(y_pred).T 
            total = time.time() - begin
            total_time = total_time + total
        total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)
        y_true = np.array(dsplus[f'{lan}_test']['labels']).T
        for i in range(len(y_pred)):
            assert(len(y_pred[i]) == len(y_true[i]))
            tp = sum([true == pred == 1 for (true,pred) in zip(y_true[i], y_pred[i])])
            tn = sum([true == pred == 0 for (true,pred) in zip(y_true[i], y_pred[i])])
            fp = sum([true == 0 and pred == 1 for (true,pred) in zip(y_true[i], y_pred[i])])
            fn = sum([true == 1 and pred == 0 for (true,pred) in zip(y_true[i], y_pred[i])])
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2*tp) / (2*tp + fp + fn)
            scores.append({'lan': lan, 'cat': labels[lan][i],'precision': precision,'recall': recall,'f1': f1, 'SYNQ': SYNQ})
    print("Compute in GFLOPs:", total_flops/10)
    print("Avg runtime in seconds:", total_time/10)
    scores = pd.DataFrame(scores)
    scores.to_csv("scores.csv", index=False)

    # max_avg_runtime = 5
    # max_avg_flops = 5000
    # # s𝑢𝑏𝑚𝑖𝑠𝑠𝑖𝑜𝑛_𝑠𝑐𝑜𝑟𝑒(𝑚𝑜𝑑𝑒𝑙)=(𝑎𝑣𝑔. 𝐹1)×0.60+max((𝑚𝑎𝑥_𝑎𝑣𝑔_𝑟𝑢𝑛𝑡𝑖𝑚𝑒−𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑_𝑎𝑣𝑔_𝑟𝑢𝑛𝑡𝑖𝑚𝑒)/𝑚𝑎𝑥_𝑎𝑣𝑔_𝑟𝑢𝑛𝑡𝑖𝑚𝑒),0)×0.2+max(((𝑚𝑎𝑥_GFLOPs−𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑_GFLOPs)/𝑚𝑎𝑥_GFLOPs), 0)×0.2
    # def score(avg_f1, avg_runtime, avg_flops):
    #     return (0.6 * avg_f1 +
    #     0.2 * max((max_avg_runtime - avg_runtime) / max_avg_runtime, 0) +
    #     0.2 * max((max_avg_flops - avg_flops) / max_avg_flops), 0)

    # avg_f1 = scores.f1.mean()
    # avg_runtime = total_time/10
    # avg_flops = total_flops/10

    # round(score(avg_f1, avg_runtime, avg_flops), 2)

if __name__ == "__main__":
    main()