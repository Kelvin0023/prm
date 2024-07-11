import os
import hydra
import datetime
from termcolor import cprint
import multiprocessing as mp

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
import wandb
import time


# OmegaConf & HydraConfig
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower(), replace=True)
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower(), replace=True)
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b, replace=True)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg, replace=True)


@hydra.main(config_name="config", config_path="cfg", version_base="1.2")
def main(cfg: DictConfig):
    """Main entry point for training/testing
    Launches a new process to create sandbox for each training run
    """

    print(OmegaConf.to_yaml(cfg))

    import isaacgym
    import torch
    from algo.rrl import RRL
    from utils.reformat import omegaconf_to_dict, print_dict
    from utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config

    # TODO: Resolve config before moving it to a dict

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed)

    output_dir = HydraConfig.get().runtime.output_dir
    agent = RRL(cfg=omegaconf_to_dict(cfg), output_dir=output_dir)

    if cfg.test:
        if cfg.test in ["random", "zero"]:
            env = agent.env
            while True:
                if cfg.test == "random":
                    actions = env.random_actions()
                elif cfg.test == "zero":
                    actions = env.zero_actions()
                _ = env.step(actions)
                if not cfg.headless:
                    env.render()
        else:
            agent.restore_test(cfg.checkpoint)
            agent.test()

    wandb.init(
        project="RRL",  # set the wandb project where this run will be logged
        # config=OmegaConf.to_container(cfg, resolve=False),  # track hyperparameters and run metadata
        name=output_dir,
    )

    agent.train()


if __name__ == "__main__":
    main()
