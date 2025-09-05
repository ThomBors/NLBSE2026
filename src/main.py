import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import set_logger

set_logger()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):

    # save_path = get_save_path(cfg)

    OmegaConf.resolve(cfg)

    base_cfg = OmegaConf.to_container(cfg, resolve=False)
    base_cfg = {"_target_": base_cfg.get("_target_")}
    pipiliner = hydra.utils.instantiate(base_cfg, cfg)

    # Run only if stats file is missing
    # if not save_path.exists():
    #     logging.info(f"run: {save_path}")
    #     return trainer()
    # else:
    #     logging.info(f"Skipping training: stats already exist at {save_path}")
    #     return None
    return pipiliner()


if __name__ == "__main__":
    main()
