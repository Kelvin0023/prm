import logging
import os
import time
from termcolor import cprint

import hydra
import numpy as np
import torch

import wandb

from utils.hydra import get_class
from utils.misc import AverageScalarMeter
from torch import nn

logger = logging.getLogger(__name__)


class RRL:

    """Rapidly Exploring Reinforcement Learning (RRL) algorithm

    RRL combines model based planning with model free reinforcement learning to
    achieve rapid exploration and learning in sparse reward environments.

    """

    def __init__(self, cfg, output_dir="debug"):
        self.cfg = cfg
        self.device = self.cfg.get("rl_device", "cuda:0")


        # ---- Environment ----
        logger.log(logging.INFO, "Creating environment")

        self.env = hydra.utils.get_class(self.cfg["rrl"]["task"]["_target_"])(
            cfg=self.cfg["rrl"]["task"],
            rl_device=self.cfg["rl_device"],
            sim_device=self.cfg["sim_device"],
            graphics_device_id=self.cfg["graphics_device_id"],
            headless=self.cfg["headless"],
            virtual_screen_capture=False,
            force_render=not self.cfg["headless"],
        )

        logger.log(logging.INFO, "Extracting shapes")
        self.num_envs = self.env.num_envs

        # Fetch dimension info from env
        self.state_dim = self.env.num_states
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # ---- Play parameters ----
        self.play_horizon_length = self.cfg["rrl"]["play_horizon_length"]
        self.plan_horizon_length = self.cfg["rrl"]["planner"]["rollout_len"]

        # ---- Sampling based-planner ----
        logger.log(logging.INFO, "Creating planner")
        planner_cfg = self.cfg["rrl"]["planner"]
        self.planner = hydra.utils.get_class(planner_cfg["_target_"])(
            cfg=planner_cfg,
            env=self.env,
            model=None,
            obs_rms=None,
            state_rms=None,
            value_rms=None,
            device=self.device,
        )

        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        # dev_output is the temporary output directory for development
        # TODO: configure output dir from hydra config
        logger.log(logging.INFO, "Creating output directory")
        # output_dir = os.path.join(os.path.curdir, output_dir)
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, f"nn")
        self.tb_dif = os.path.join(self.output_dir, f"tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Value and advantage estimation  ----
        self.gamma = self.cfg["rrl"]["gamma"]
        self.tau = self.cfg["rrl"]["tau"]
        self.value_bootstrap = self.cfg["rrl"]["value_bootstrap"]
        self.normalize_input = self.cfg["rrl"]["normalize_input"]
        self.normalize_value = self.cfg["rrl"]["normalize_value"]
        self.normalize_advantage = self.cfg["rrl"]["normalize_advantage"]
        # Normalize the advantage is not respected as it hard coded to be True  in ExperienceBuffer
        # where the adavatage is computed.

        # ---- Snapshot
        self.save_freq = self.cfg["rrl"]["save_frequency"]
        self.save_best_after = self.cfg["rrl"]["save_best_after"]

        # ---- Logging ----
        self.eval_freq = self.cfg["rrl"]["eval_frequency"]
        self.extra_info = {}
        # writer = SummaryWriter(self.tb_dif)
        # self.writer = writer
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.eval_episode_rewards = AverageScalarMeter(self.num_envs)
        self.eval_episode_lengths = AverageScalarMeter(self.num_envs)

        # Compute success rate during training
        self.num_train_success = AverageScalarMeter(100)
        self.num_train_episodes = AverageScalarMeter(100)
        # Compute success rate during evaluation
        self.num_eval_success = AverageScalarMeter(100)
        self.num_eval_episodes = AverageScalarMeter(100)

        # ---- Training ----
        self.obs = None
        self.epoch_num = 0
        self.current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.eval_current_rewards = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.eval_current_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((self.num_envs,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.cfg["rrl"]["max_agent_steps"]
        # TODO: make sure we count the planner steps correctly and add them to the agent steps
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

        # set up decaying parameters for DAPG
        self.lambda_1 = self.cfg["rrl"]["lambda_1"]  # decay rate for the BC loss
        self.lambda_1_k = self.cfg["rrl"]["lambda_1_k"]  # start value (usually set as 1)
        self.reach_maximum_nodes = False

        # load RRT from file
        self.load_rrt = self.cfg["rrl"]["saved_tree"]
        if self.load_rrt:
            cprint("Loading RRT from file", "red")
            self.planner.load_tree(self.cfg["rrl"]["tree_file"])



    def train(self):
        logger.log(logging.INFO, "Starting training")
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()

        self.planner.runPRM()





