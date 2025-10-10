import logging
import os
import torch
from datasets import load_dataset, load_from_disk
import gc
from src.augment import run_augmentation_pipeline
from src.multitask_setfit_trainer import classifiers
from src.evaluation import evaluation
from src.finetune import createMLforWCft
from src.oversampling import oversampling
from src.utils import set_seed, title, generate_label_statistics


class pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
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

    def __call__(self):
        title()
        cfg = self.cfg
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.trainerHardwer.use_cuda
            else "cpu"
        )
        logging.info(f"Device: {device}")

        set_seed(cfg.seed)

        ds = load_dataset("NLBSE/nlbse26-code-comment-classification")

        # --- fine tune ModernBERT for augmentation --- #
        createMLforWCft(cfg, ds, self.langs, device)
        self.cleanup()

        # --- Synthetic Augmentation --- #
        run_augmentation_pipeline(cfg, ds)
        self.cleanup()

        # --- Load new Augmentd Data --- #
        dsplus = load_from_disk(f"{cfg.paths.data_dir}/augmented_datasets")

        # --- Set Syntetic Quality and Number of Observation--- #
        SYNQ = cfg.trainer.SYNQ
        dsplus, strategy = oversampling(cfg, dsplus, SYNQ)

        # --- Report Syntetic Quality and Number of Observation--- #
        generate_label_statistics(cfg, dsplus, SYNQ,strategy, "oversampling_report_complete.csv")

        # --- Code Commente Classification --- #
        classifiers(cfg, dsplus, SYNQ)
        self.cleanup()

        # --- Test Pipeline --- #
        scores, compute = evaluation(cfg, dsplus, self.langs, self.labels, SYNQ)

        output_dir = f"{cfg.paths.res_dir}/performance/{strategy}/{SYNQ}"
        os.makedirs(output_dir, exist_ok=True)

        scores.to_csv(f"{output_dir}/scores.csv", index=False)
        compute.to_csv(f"{output_dir}/compute.csv", index=False)

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

    def cleanup(self):

        # Free CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Clear any remaining references
        gc.collect()
