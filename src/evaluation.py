from setfit import SetFitModel
import torch
import time
import numpy as np
import pandas as pd
import logging

def evaluation(cfg,ds,langs,labels,SYNQ):
    total_flops = 0
    total_time = 0
    scores = []
    for lan in langs:
        # to load trained models:
        model = SetFitModel.from_pretrained(f'{cfg.paths.res_dir}/models/{lan}-SetFit')
        # to load pretrained models from Hub:
        # model = SetFitModel.from_pretrained(f'NLBSE/nlbse26_{lan}')
        with torch.profiler.profile(with_flops=True) as p:
            x = ds[f'{lan}_test'][:]["combo"]
            begin = time.time()
            for i in range(10):
            y_pred = model(x)
            y_pred = np.asarray(y_pred).T 
            total = time.time() - begin
            total_time = total_time + total
        total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)
        y_true = np.array(ds[f'{lan}_test']['labels']).T
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
    logging.info("Compute in GFLOPs:", total_flops/10)
    logging.info("Avg runtime in seconds:", total_time/10)
    return pd.DataFrame(scores)