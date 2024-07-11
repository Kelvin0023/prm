import logging
import os
import time
from termcolor import cprint

import hydra
import numpy as np
import torch

import wandb
from algo.policy.experience import ExperienceBuffer
from algo.policy.running_mean_std import RunningMeanStd
from algo.policy.actor_critic import ActorCritic
from algo.policy.actor_critic_cvae import ActorCritic_CVAE
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

        # ---- Algorithm ----
        self.algo = self.cfg["algo"]
        assert self.algo in ["PPO", "RRL-RL", "RRL-BC", "RRL-BC-CVAE", "RRL-BC-RL", "RRL-DAPG","RRL-DAPG-CVAE"], \
            "Invalid algorithm type"

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
        if self.algo == "RRL-RL":
            self.batch_size = (self.play_horizon_length + self.plan_horizon_length) * self.num_envs
        elif self.algo == "PPO":
            self.batch_size = self.play_horizon_length * self.num_envs
        elif self.algo == "RRL-BC" or self.algo == "RRL-BC-CVAE":
            self.batch_size = self.plan_horizon_length * self.num_envs
        elif self.algo == "RRL-BC-RL" or self.algo == "RRL-DAPG" or self.algo == "RRL-DAPG-CVAE":
            self.batch_size = self.play_horizon_length * self.num_envs

        # ---- Storage ----
        # Create storage for the play steps
        self.storage_play = ExperienceBuffer(
            self.num_envs,
            self.play_horizon_length,
            self.state_dim,
            self.obs_dim,
            self.act_dim,
            self.device,
        )
        # Create storage for the RRT plan steps
        self.storage_plan = ExperienceBuffer(
            self.num_envs,
            self.plan_horizon_length,
            self.state_dim,
            self.obs_dim,
            self.act_dim,
            self.device,
        )

        # ---- Models ----
        # Store the modules in a ModuleList for convenience funtions like .train() and .eval()
        self.models = nn.ModuleList()

        net_config = {
            "actor_units": self.cfg["rrl"]["actor_critic"]["network"]["actor_units"],
            "critic_units": self.cfg["rrl"]["actor_critic"]["network"]["critic_units"],
            "actions_num": self.act_dim,
            "obs_shape": self.env.observation_space.shape,
            "state_shape": (self.env.num_states, 1),
        }
        if self.algo == "RRL-BC-CVAE" or self.algo == "RRL-DAPG-CVAE":
            net_config["device"] = self.device
            net_config["latent_dim"] = 32
            self.model = ActorCritic_CVAE(net_config).to(self.device)
        else:
            self.model = ActorCritic(net_config).to(self.device)

        self.models.append(self.model)

        #  ---- Normalization ----
        logger.log(logging.INFO, "Creating normalization modules")
        # The normalization can be turned off by setting these modules to eval() mode
        # normalization for actor
        self.obs_rms = RunningMeanStd((self.obs_dim,)).to(self.device)
        # normalization for critic
        self.c1_state_rms = RunningMeanStd((self.state_dim,)).to(self.device)
        self.c1_value_rms = RunningMeanStd((1,)).to(self.device)

        # ---- Trainers ----
        logger.log(logging.INFO, "Creating trainers")
        model_trainer_cfg = self.cfg["rrl"]["actor_critic"]
        self.model_trainer = hydra.utils.get_class(model_trainer_cfg["_target_"])(
            cfg=model_trainer_cfg,
            model=self.model,
            obs_rms=self.obs_rms,
            state_rms=self.c1_state_rms,
            value_rms=self.c1_value_rms,
            device=self.device
        )

        # ---- Sampling based-planner ----
        logger.log(logging.INFO, "Creating planner")
        planner_cfg = self.cfg["rrl"]["planner"]
        self.planner = hydra.utils.get_class(planner_cfg["_target_"])(
            cfg=planner_cfg,
            env=self.env,
            model=self.model,
            obs_rms=self.obs_rms,
            state_rms=self.c1_state_rms,
            value_rms=self.c1_value_rms,
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

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, last_lr):
        if self.algo == "RRL-RL" or self.algo == "PPO" or self.algo == "RRL-BC-RL" or self.algo == "RRL-DAPG" or self.algo == "RRL-DAPG-CVAE":
            wandb.log({"performance/RLTrainFPS": self.agent_steps / self.rl_train_time}, step=self.agent_steps)
            wandb.log({"performance/EnvStepFPS": self.agent_steps / self.data_collect_time}, step=self.agent_steps)
            wandb.log({"losses/actor_loss": torch.mean(torch.stack(a_losses)).item()}, step=self.agent_steps)
            wandb.log({"losses/bounds_loss": torch.mean(torch.stack(b_losses)).item()}, step=self.agent_steps)
            wandb.log({"losses/critic_loss": torch.mean(torch.stack(c_losses)).item()}, step=self.agent_steps)
            wandb.log({"losses/entropy": torch.mean(torch.stack(entropies)).item()}, step=self.agent_steps)
            wandb.log({"info/last_lr": last_lr}, step=self.agent_steps)
            # wandb.log({"info/e_clip": self.e_clip}, step=self.agent_steps)
            wandb.log({"info/kl": torch.mean(torch.stack(kls)).item()}, step=self.agent_steps)
            for k, v in self.extra_info.items():
                wandb.log({f"{k}": v}, step=self.agent_steps)
        elif self.algo == "RRL-BC" or self.algo == "RRL-BC-CVAE":
            wandb.log({"performance/RLTrainFPS": self.agent_steps / self.rl_train_time}, step=self.agent_steps)
            wandb.log({"performance/EnvStepFPS": self.agent_steps / self.data_collect_time}, step=self.agent_steps)
            wandb.log({"losses/mse_loss": torch.mean(torch.stack(a_losses)).item()}, step=self.agent_steps)
            for k, v in self.extra_info.items():
                wandb.log({f"{k}": v}, step=self.agent_steps)

    def set_eval(self):
        self.models.eval()
        if self.normalize_input:
            self.obs_rms.eval()
            self.c1_state_rms.eval()
        if self.normalize_value:
            self.c1_value_rms.eval()

    def set_train(self):
        self.models.train()
        if self.normalize_input:
            self.obs_rms.train()
            self.c1_state_rms.train()
        if self.normalize_value:
            self.c1_value_rms.train()

    def model_act(self, obs_dict):
        processed_obs = self.obs_rms(obs_dict['obs'])
        processed_states = self.c1_state_rms(obs_dict['states'])
        input_dict = {'obs': processed_obs, 'states': processed_states}
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.c1_value_rms(res_dict['values'], True)
        return res_dict

    def model_act_inference(self, obs_dict):
        processed_obs = self.obs_rms(obs_dict["obs"])
        processed_state = self.c1_state_rms(obs_dict['states'])
        input_dict = {"obs": processed_obs, "states": processed_state}
        mu = self.model.act_inference(input_dict)
        return {"actions": mu}

    def train(self):
        logger.log(logging.INFO, "Starting training")
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()

    def train(self):
        logger.log(logging.INFO, "Starting training")
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch_size

        # initialize PRM
        self.planner.initPRM()

        while self.agent_steps < self.max_agent_steps:
            # Run PRM for one epoch
            self.planner.runPRM()

            if self.epoch_num == 0 or (self.eval_freq > 0 and self.epoch_num % self.eval_freq == 0):
                # Evaluate the model
                self.eval_steps()
                eval_mean_rewards = self.eval_episode_rewards.get_mean()
                eval_mean_lengths = self.eval_episode_lengths.get_mean()
                wandb.log({"eval/episode_rewards": eval_mean_rewards}, step=self.agent_steps)
                wandb.log({"eval/episode_lengths": eval_mean_lengths}, step=self.agent_steps)
                print(f"Eval rewards: {eval_mean_rewards:.2f}")
                if self.epoch_num == 0:
                    self.best_rewards = eval_mean_rewards

                # update evaluation success rate if environment has returned such data
                if self.num_eval_success.current_size > 0:
                    running_mean_success = self.num_eval_success.get_mean()
                    running_mean_term = self.num_eval_episodes.get_mean()
                    mean_success_rate = running_mean_success / running_mean_term
                    wandb.log({"eval_success_rate/step": mean_success_rate}, step=self.agent_steps)

            self.epoch_num += 1

            a_losses, c_losses, b_losses, entropies, kls, last_lr = self.train_epoch()

            # clear the experience buffer after training
            self.storage_play.clear()
            self.storage_plan.clear()

            self.agent_steps += self.batch_size

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | "
                f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                f"Best: {self.best_rewards:.2f}"
            )
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls, last_lr)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            wandb.log({"train/episode_rewards": mean_rewards}, step=self.agent_steps)
            wandb.log({"train/episode_lengths": mean_lengths}, step=self.agent_steps)
            checkpoint_name = f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}"

            # update training success rate if environment has returned such data
            if self.num_train_success.current_size > 0:
                running_mean_success = self.num_train_success.get_mean()
                running_mean_term = self.num_train_episodes.get_mean()
                mean_success_rate = running_mean_success / running_mean_term
                wandb.log({"train_success_rate/step": mean_success_rate}, step=self.agent_steps)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, "last"))

            if mean_rewards == -np.Inf:  # mean_rewards are -inf if training episodes never end, use eval metrics
                mean_rewards = eval_mean_rewards
                mean_lengths = eval_mean_lengths

            if eval_mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f"save current best reward: {eval_mean_rewards:.2f}")
                self.best_rewards = eval_mean_rewards
                self.save(os.path.join(self.nn_dir, "best"))

            # if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
            #     print(f"save current best reward: {mean_rewards:.2f}")
            #     self.best_rewards = mean_rewards
            #     self.save(os.path.join(self.nn_dir, "best"))

            wandb.log({"agent_steps": self.agent_steps}, step=self.epoch_num)

        logger.log(logging.INFO, "Finished training")

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "c1_state_rms": self.c1_state_rms.state_dict(),
            "c1_value_rms": self.c1_value_rms.state_dict(),
        }
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        self.c1_state_rms.load_state_dict(checkpoint["c1_state_rms"])
        self.c1_value_rms.load_state_dict(checkpoint["c1_value_rms"])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])

    def test(self):
        self.env.reset_dist_type = "eval"
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                "obs": self.obs_rms(obs_dict["obs"]),
                "states": self.c1_state_rms(obs_dict["states"])
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def _train_rrl_bc(self):
        _t = time.time()
        self.set_eval()
        ep_ctxt = self.env.save_episode_context()
        self.env.reset_dist_type = "train"
        self.env.success_rate_mode = "plan"

        self.planner.plan(storage=self.storage_plan)
        self.obs = self.env.restore_episode_context(ep_ctxt)
        self.data_collect_time += time.time() - _t

        info_string = (
            f"Tree size: {len(self.planner.rrt_q)} |"
            f"Best dist: {self.planner.rrt_best_dist:.1f}"
        )
        print(info_string)

        _t = time.time()
        self.set_train()

        obs_buf, act_buf = self.planner.extract_demonstrations(500)
        if self.algo == "RRL-BC-CVAE":
            loss = self.model_trainer.train_bc_cvae(obs_buf=obs_buf, act_buf=act_buf)
        elif self.algo == "RRL-BC":
            loss = self.model_trainer.train_bc(obs_buf=obs_buf, act_buf=act_buf)
        self.rl_train_time += time.time() - _t

        return loss, None, None, None, None, None

    def _train_rrl_dapg(self):
        # train with optimal plan (demonstrations) collected in RRT
        self.lambda_1_k *= self.lambda_1
        _t = time.time()
        self.set_eval()
        self.env.reset_dist_type = "train"
        print("debug:", len(self.planner.rrt_states), "states found.")
        print("debug: value of lambda1 is", self.lambda_1_k)

        if self.reach_maximum_nodes is False and len(self.planner.rrt_states) > self.cfg["rrl"]["planner"]["num_reset_states"]:
            self.reach_maximum_nodes = True
            self.steps_condition_1 = int(self.agent_steps // 1e6)

        if self.reach_maximum_nodes is False:
            self.env.success_rate_mode = "plan"
            ep_ctxt = self.env.save_episode_context()
            self.planner.plan(storage=self.storage_plan)
            self.obs = self.env.restore_episode_context(ep_ctxt)
            self.data_collect_time += time.time() - _t

            info_string = (
                f"Tree size: {len(self.planner.rrt_q)} |"
                f"Best dist: {self.planner.rrt_best_dist:.1f}"
            )
            print(info_string)
        else:
            print("Step number when the first condition is met:", self.steps_condition_1)

        obs_buf, act_buf = self.planner.extract_demonstrations(500)

        # train with data collected from the environment (play steps)
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        _t = time.time()

        self.set_train()
        if self.algo == "RRL-DAPG-CVAE":
            bc_loss, a_losses, c_losses, b_losses, entropies, kls, last_lr = self.model_trainer.train_dapg_cvae(
                storage=self.storage_play, obs_buf=obs_buf, act_buf=act_buf, lambda1=self.lambda_1_k)
        elif self.algo == "RRL-DAPG":
            bc_loss, a_losses, c_losses, b_losses, entropies, kls, last_lr = self.model_trainer.train_dapg(
                storage=self.storage_play, obs_buf=obs_buf, act_buf=act_buf, lambda1=self.lambda_1_k)
        self.rl_train_time += time.time() - _t

        # write bc loss to wandb
        wandb.log({"losses/bc_mse_loss": torch.mean(torch.stack(bc_loss)).item()}, step=self.agent_steps)

        return a_losses, c_losses, b_losses, entropies, kls, last_lr

    def _train_rrl_bc_rl(self):
        # train with optimal plan (demonstrations) collected in RRT
        _t = time.time()
        self.set_eval()
        ep_ctxt = self.env.save_episode_context()
        self.env.reset_dist_type = "train"
        self.env.success_rate_mode = "plan"

        self.planner.plan(storage=self.storage_plan)
        self.obs = self.env.restore_episode_context(ep_ctxt)
        self.data_collect_time += time.time() - _t

        info_string = (
            f"Tree size: {len(self.planner.rrt_q)} |"
            f"Best dist: {self.planner.rrt_best_dist:.1f}"
        )
        print(info_string)

        _t = time.time()
        self.set_train()

        obs_buf, act_buf = self.planner.extract_demonstrations(500)
        loss = self.model_trainer.train_bc(obs_buf=obs_buf, act_buf=act_buf)
        self.rl_train_time += time.time() - _t

        # write bc loss to wandb
        wandb.log({"losses/bc_mse_loss": torch.mean(torch.stack(loss)).item()}, step=self.agent_steps)

        # train with data collected from the environment (play steps)
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        _t = time.time()

        self.set_train()
        a_losses, c_losses, b_losses, entropies, kls, last_lr = self.model_trainer.train_rl(storage=self.storage_play)
        self.rl_train_time += time.time() - _t

        return a_losses, c_losses, b_losses, entropies, kls, last_lr

    def _train_rrl_rl(self):
        # train with data collected from the planner (plan steps)
        _t = time.time()
        self.set_eval()
        self.env.reset_dist_type = "train"
        print("debug:", len(self.planner.rrt_states), "states found.")

        if self.reach_maximum_nodes is False and len(self.planner.rrt_states) > self.cfg["rrl"]["planner"][
            "num_reset_states"]:
            self.reach_maximum_nodes = True
            self.steps_condition_1 = int(self.agent_steps // 1e6)

        if self.reach_maximum_nodes is False:
            ep_ctxt = self.env.save_episode_context()
            self.plan_steps()
            self.obs = self.env.restore_episode_context(ep_ctxt)
            self.data_collect_time += time.time() - _t

            info_string = (
                f"Tree size: {len(self.planner.rrt_q)} |"
                f"Best dist: {self.planner.rrt_best_dist:.1f}"
            )
            print(info_string)
        else:
            print("Step number when the first condition is met:", self.steps_condition_1)

        # train with data collected from the environment (play steps)
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        _t = time.time()

        self.set_train()
        a_losses, c_losses, b_losses, entropies, kls, last_lr = self.model_trainer.train_rl(storage=self.storage_play)
        self.rl_train_time += time.time() - _t

        return a_losses, c_losses, b_losses, entropies, kls, last_lr

    def _train_rl(self):
        # print("debug: called train_epoch()")
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        _t = time.time()

        self.set_train()
        a_losses, c_losses, b_losses, entropies, kls, last_lr = self.model_trainer.train_rl(storage=self.storage_play)
        self.rl_train_time += time.time() - _t

        return a_losses, c_losses, b_losses, entropies, kls, last_lr

    def train_epoch(self):
        if self.algo == "RRL-BC":
            return self._train_rrl_bc()
        elif self.algo == "RRL-RL":
            return self._train_rrl_rl()
        elif self.algo == "PPO":
            return self._train_rl()
        elif self.algo == "RRL-BC-RL":
            return self._train_rrl_bc_rl()
        elif self.algo == "RRL-DAPG":
            return self._train_rrl_dapg()
        elif self.algo == "RRL-BC-CVAE":
            return self._train_rrl_bc()
        elif self.algo == "RRL-DAPG-CVAE":
            return self._train_rrl_dapg()

    def play_steps(self, enable_reset=True, reset_dist_type="train"):
        # Prepare for play
        # First, enable reset as it is disabled during plan_steps in the previous epoch
        # Then, set the reset_dist_type to train instead of eval in case it was set to eval previously
        self.env.enable_reset = enable_reset
        self.env.reset_dist_type = reset_dist_type
        self.env.success_rate_mode = "train"

        # Store the rollout data in storage_play
        storage = self.storage_play
        for n in range(self.play_horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            storage.update_data("obses", n, self.obs["obs"])
            storage.update_data("states", n, self.obs["states"])
            for k in ["actions", "neglogpacs", "values", "mus", "sigmas"]:
                storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict["actions"], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            storage.update_data("dones", n, self.dones)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += self.gamma * res_dict["values"] * infos["time_outs"].unsqueeze(1).float()
            storage.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            # # if success in info, then update success rate
            if 'success' in infos:
                num_train_success = infos['success']
                self.num_train_success.update(num_train_success)
                num_train_terminations = self.dones
                self.num_train_episodes.update(num_train_terminations)
            assert isinstance(infos, dict), "Info Should be a Dict"
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict["values"]

        storage.computer_return(last_values, self.gamma, self.tau)
        storage.prepare_training()

        returns = storage.data_dict["returns"]
        values = storage.data_dict["values"]
        if self.normalize_value:
            self.c1_value_rms.train()
            values = self.c1_value_rms(values)
            returns = self.c1_value_rms(returns)
            self.c1_value_rms.eval()
        storage.data_dict["values"] = values
        storage.data_dict["returns"] = returns

    def eval_steps(self):
        self.set_eval()
        self.env.enable_reset = True
        self.env.reset_dist_type = "eval"
        self.env.success_rate_mode = "eval"
        ep_ctxt = self.env.save_episode_context()
        obs = self.env.reset()
        eval_current_rewards = torch.zeros(size=(self.num_envs, 1), dtype=torch.float32, device=self.device)
        eval_current_lengths = torch.zeros(size=(self.num_envs,), dtype=torch.float32, device=self.device)
        count = 0  # evaluate once for each env
        for n in range(self.env.max_episode_length):
            res_dict = self.model_act_inference(obs)
            actions = torch.clamp(res_dict["actions"], -1.0, 1.0)
            obs, rewards, dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            eval_current_rewards += rewards
            eval_current_lengths += 1

            # if success in info, then update success rate
            if 'success' in infos:
                num_eval_success = infos['success']
                self.num_eval_success.update(num_eval_success)
                num_eval_terminations = dones
                self.num_eval_episodes.update(num_eval_terminations)

            done_indices = dones.nonzero(as_tuple=False)
            count += len(done_indices)
            self.eval_episode_rewards.update(eval_current_rewards[done_indices])
            self.eval_episode_lengths.update(eval_current_lengths[done_indices])
            not_dones = 1.0 - dones.float()
            eval_current_rewards = eval_current_rewards * not_dones.unsqueeze(1)
            eval_current_lengths = eval_current_lengths * not_dones

            if count >= self.env.num_envs:
                break
        self.obs = self.env.restore_episode_context(ep_ctxt)

    def plan_steps(self):
        self.env.reset_dist_type = "train"
        self.env.success_rate_mode = "plan"

        storage = self.storage_plan
        self.planner.plan(storage)
        states = self.c1_state_rms(self.env.get_state())
        last_values = self.planner.model.critic_sample(states)
        storage.computer_return(last_values, self.gamma, self.tau)
        storage.prepare_training()
        returns = storage.data_dict["returns"]
        values = storage.data_dict["values"]
        if self.normalize_value:
            self.c1_value_rms.train()
            values = self.c1_value_rms(values)
            returns = self.c1_value_rms(returns)
            self.c1_value_rms.eval()
        storage.data_dict["values"] = values
        storage.data_dict["returns"] = returns





