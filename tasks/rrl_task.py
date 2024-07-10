import os.path
import numpy as np
from tasks.vec_task import VecTask
import torch
import picologging
from termcolor import cprint

logger = picologging.getLogger("root")


class RRLTask(VecTask):
    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
    ):
        super().__init__(
            config,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self.reset_state_buf = None
        self.reset_state_dim = None
        if self.cfg["env"].get("use_saved_reset_states", False):
            logger.info("use_saved_reset_states=True")
            self.reset_state_buf = self.load_reset_states()
            self.reset_state_dim = self.reset_state_buf.shape[1]
        self.reset_select_style = self.cfg.get("reset_select_style", "naive")
        if self.reset_select_style:
            self.start_state_bias = self.cfg.get("start_state_bias", 0.0)

        # Flags to control reset behavior as the reset behavior needs
        # to change between RRT and RL
        self.enable_reset = True
        self.reset_dist_type = "train"
        if self.cfg["env"].get("staggered_progress", False):
            self.progress_buf = torch.randint_like(self.progress_buf, self.max_episode_length - 2)

    def refresh_tensors(self):
        raise NotImplementedError

    def compute_obs(self):
        """Compute obs and state and update obs_buf and states_buf respectively"""
        raise NotImplementedError

    def compute_reward(self):
        """Compute reward and update rew_buf"""
        raise NotImplementedError

    def compute_reward_from_states(self, state, prev_state=None):
        """Compute reward for prev_state -> state transition"""
        raise NotImplementedError

    def check_constraints(self):
        """Check state-space constraints and return a boolean array True for invalid states"""
        raise NotImplementedError

    def check_termination(self):
        """Check the environments that need to be reset after taking max_episode_steps"""
        reset = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )
        return reset

    def load_reset_states(self):
        file = os.path.join(self.cfg["env"]["saved_reset_states_file"])
        logger.info(f"Loading reset states form {file}")
        assert os.path.exists(file), f"{file} does not exist"
        cprint(f"Loading grasps from {file}", "blue", attrs=["bold"])
        return torch.from_numpy(np.load(file)).to(self.device)

    def post_physics_step(self):
        self.progress_buf += 1
        self.refresh_tensors()
        # self.compute_obs()
        self.compute_reward()
        if self.enable_reset:
            self.reset_done()
            self.reset_buf[:] = self.check_termination()
        self.compute_obs()

    def reset_idx(self, env_idx):
        if self.reset_dist_type == "train":
            if self.reset_select_style == "naive":
                idx = torch.randint_like(env_idx, len(self.reset_state_buf), device=self.device)
                states = self.reset_state_buf[idx]

            elif self.reset_select_style == "nearest":
                q_sample = self.sample_q(len(env_idx))
                states = torch.zeros((len(env_idx), self.reset_state_dim), device=self.device)
                for i in range(len(env_idx)):
                    dist = torch.linalg.norm(self.reset_state_buf[:, :3] - q_sample[i, :], dim=1)
                    nearest_idx = int(torch.argmin(dist))
                    nearest_state = self.reset_state_buf[nearest_idx]
                    states[i] = nearest_state

        elif self.reset_dist_type == "eval":
            states = self.get_env_root_state()
        self.set_env_states(states, env_idx)
        self.reset_buf[env_idx] = 0.0
        self.progress_buf[env_idx] = 0.0

    def reset(self):
        env_idx = torch.tensor(list(range(self.num_envs)), device=self.device)
        self.reset_idx(env_idx)
        return super().reset()

    def get_env_states(self):
        """Gets state of the environment"""
        raise NotImplementedError

    def set_env_states(self, states, env_idx):
        """Set env state
        Args:
            states: torch.Tensor (num_envs, state_dim) or (state_dim,) tensor consisting of state(s).
            env_idx: torch.IntTensor env_idx for which will be set to state(s) provided.
        """
        raise NotImplementedError

    def get_env_root_q(self):
        raise NotImplementedError

    def get_env_q(self):
        raise NotImplementedError

    def get_env_root_state(self):
        raise NotImplementedError

    def set_reset_state_buf(self, buf):
        self.reset_state_buf = buf

    def sample_q(self, num_samples):
        """Sample from q-space"""
        raise NotImplementedError

    def check_constraints(self):
        raise NotImplementedError

    def dist(self, q1, q2):
        """Computes distance between points in state-space"""
        return torch.norm(q1 - q2, dim=1)

    def save_episode_context(self):
        """Saves episode context to switch to planner"""

        context = {
            "progress_buf": self.progress_buf.detach().clone(),
            "reset_buf": self.reset_buf.detach().clone(),
            "dones": self.reset_buf.detach().clone(),
            "env_states": self.get_env_states(),
            "goal": self.goal.detach().clone() if hasattr(self, "goal") else "None",
        }

        return context

    def restore_episode_context(self, context):
        """Restore episode context from planner to RL"""

        self.progress_buf = context["progress_buf"]
        self.reset_buf = context["reset_buf"]
        self.dones = context["dones"]
        self.set_env_states(context["env_states"], torch.arange(self.num_envs, device=self.device))
        if hasattr(self, "goal"):
            self.goal = context["goal"]
        self.compute_obs()

        return {"obs": self.get_obs(), "states": self.get_state()}
