import os
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from tasks.rrl_task import RRLTask
from utils.torch_jit_utils import tensor_clamp, to_torch
from utils.misc import AverageScalarMeter


class MazeBot(RRLTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
    ):
        self.cfg = cfg
        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 2
        self.cfg["env"]["numStates"] = 4
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.up_axis = "z"

        # set controller params
        self.kp = 5
        self.kd = 2
        self.dof_vel_lim = cfg["env"]["dof_vel_lim"]

        # set maze specific params
        self.maze = self.cfg["env"]["maze"]
        self.dof_pos_lim = cfg["env"][self.maze]["dof_pos_lim"]
        self.x_dof_lim = self.dof_pos_lim["x"]
        self.y_dof_lim = self.dof_pos_lim["y"]

        self.dof_pos_start = cfg["env"][self.maze]["dof_pos_start"]
        # set the file so that the reset states are loaded from the correct file in _post_init()
        self.cfg["env"]["saved_reset_states_file"] = self.cfg["env"][self.maze]["saved_reset_states_file"]

        print(f'Using {self.cfg["env"]["saved_reset_states_file"]}')

        # set reset and reward params
        self.state_dim = 4
        self.reset_at_goal = cfg["reset_at_goal"]
        self.bonus_at_goal = cfg["bonus_at_goal"]
        self.success_threshold = cfg["success_threshold"]
        self.at_goal_threshold = cfg["at_goal_threshold"]
        self.reset_select_style = cfg["reset_select_style"]
        if self.reset_select_style:
            self.start_state_bias = cfg["start_state_bias"]

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        # Create tensor view for computing obs, states and reward
        self._create_tensor_views()

        # Setup viewer
        if self.viewer is not None:
            a = 0.0001
            cam_pos = gymapi.Vec3(-0.4, 0.0, 2)
            cam_target = gymapi.Vec3(-0.4, a, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        if self.reset_state_buf is None:
            self.reset_state_buf = self.get_env_root_state()
            self.reset_state_dim = self.reset_state_buf.shape[1]

        # Setup goal buffer
        # If goal is specified in the config, use that, else use the reset states
        if "goal" in self.cfg["env"][self.maze]:
            self.goal_buf = to_torch(self.cfg["env"][self.maze]["goal"], device=self.device).unsqueeze(0)
        else:
            self.goal_buf = self.load_reset_states()[:, :2]

        # Flags to control reset behavior as the reset behavior needs
        # to change between RRT and RL rollouts
        self.enable_reset = True
        self.reset_dist_type = "eval"

        self.reward_type = self.cfg["env"].get("reward_type", "dense")
        self.target_limit_lower = torch.tensor([self.x_dof_lim[0], self.y_dof_lim[0]], device=self.device)
        self.target_limit_upper = torch.tensor([self.x_dof_lim[1], self.y_dof_lim[1]], device=self.device)
        self.reset_state_noise = torch.tensor(self.cfg["reset_state_noise"], device=self.device)
        self.goal = torch.zeros((self.num_envs, 2), device=self.device)

        # Logging success rate
        self.success = torch.zeros_like(self.reset_buf)
        self.success_rate = AverageScalarMeter(100)
        self.extras["success_rate"] = 0.0

        self._setup_rrt_config()

    def _setup_rrt_config(self):
        # create sim state map for parsing states
        self.state_space_map = {
            "pos": (0, 2),
            "vel": (2, 4),
        }
        self.sample_space_map = {
            "pos": (0, 2),
        }
        self.goal_map = (2, 4)
        self.selected_goal_map = (0, 2)

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == "z" else 1  # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets"))
        maze_urdf = self.cfg["env"]["maze"] + ".urdf"
        maze_asset_file = os.path.join("maze", "urdf", maze_urdf)
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.vhacd_enabled = False
        maze_asset = self.gym.load_asset(self.sim, asset_root, maze_asset_file, asset_options)
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = False

        dof_props = self.gym.get_asset_dof_properties(maze_asset)
        self.num_dofs = self.gym.get_asset_dof_count(maze_asset)
        self.envs = []
        dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
        dof_props["damping"].fill(0.0)

        # Does NOT work in enforcing a limit! Make desired change in URDF
        # dof_props["lower"][0] = self.xlim[0]
        # dof_props["lower"][0] = self.ylim[0]
        # dof_props["upper"][1] = self.xlim[1]
        # dof_props["upper"][1] = self.ylim[1]
        if self.dof_vel_lim is not None:
            dof_props["velocity"].fill(self.dof_vel_lim)

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            maze_poze = gymapi.Transform()
            maze_poze.p = gymapi.Vec3(0, 0, 0.0)
            maze_actor = self.gym.create_actor(env, maze_asset, maze_poze, "maze", i, 0, 0)

    def _create_tensor_views(self):
        dof_state_ = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_).view(self.num_envs, -1, 2)
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]

        # for controller
        self.dof_pos_target = self.dof_pos.detach().clone()

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)

    """ RL methods """

    def pre_physics_step(self, actions: torch.Tensor):
        self.set_actions(actions)

    def set_actions(self, actions: torch.Tensor):
        """In-built position controlled didn't work on this problem due to some bug.
        Implemented a controller myself."""
        self.dof_pos_target = tensor_clamp(self.dof_pos + actions, self.target_limit_lower, self.target_limit_upper)
        force = (self.dof_pos_target - self.dof_pos) * self.kp - self.dof_vel * self.kd
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(force))

    def compute_obs(self):
        pos = self.dof_pos.view(self.num_envs, -1).detach().clone()
        vel = self.dof_vel.view(self.num_envs, -1).detach().clone()
        goal = self.goal
        self.states_buf[:] = torch.cat([pos, goal], dim=1)
        self.obs_buf[:] = torch.cat([pos, goal], dim=1)

    def compute_reward(self, actions=None):
        dist = torch.linalg.norm(self.dof_pos - self.goal, dim=1)
        if self.reward_type == "dense":
            reward = -(dist**2) + (dist < self.at_goal_threshold) * self.bonus_at_goal
        elif self.reward_type == "sparse":
            reward = (dist < self.at_goal_threshold) * 1.0
        if self.reset_dist_type == "eval":
            self.success = torch.logical_or(self.success > 0, (dist < 0.2))
        self.rew_buf = reward

    def compute_reward_from_states(self, state, prev_state=None):
        # state = state.detach().clone()
        # pos = state[0, :2].view(-1, 1)
        # dist = torch.linalg.norm(pos - self.goals, dim=1)
        # if self.reward_type == "dense":
        #     reward = - dist**2 + (dist < 0.1) * self.bonus_at_goal
        # elif self.reward_type == "sparse":
        #     reward = (dist < 0.1) * 1.0
        # reward = reward[0]
        return 0.0

    def check_constraints(self):
        return torch.zeros_like(self.reset_buf)

    def check_termination(self):
        reset = torch.zeros_like(self.reset_buf)
        if self.reset_at_goal:
            dist = torch.linalg.norm(self.dof_pos - self.goal, dim=1)
            reset = torch.where(dist < self.at_goal_threshold, torch.ones_like(self.reset_buf), reset)
        reset = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            reset,
        )
        return reset

    def reset_idx(self, env_idx):
        self.goal[env_idx] = self.goal_buf[torch.randint_like(env_idx, len(self.goal_buf))]
        if self.reset_dist_type == "train":
            # print("reset dist = train")
            if self.reset_select_style == "naive":
                # print("reset select style = train")
                buf_idx = torch.randint_like(env_idx, len(self.reset_state_buf))
                states = self.reset_state_buf[buf_idx]
                for i in range(len(env_idx)):
                    u = np.random.uniform()
                    if u < self.start_state_bias:
                        states[i, 0], states[i, 1] = (
                            self.dof_pos_start[0],
                            self.dof_pos_start[1],
                        )
                        states[i, 2:] = 0.0
                states = torch.zeros((len(env_idx), 4), device=self.device)
            elif self.reset_select_style == "nearest":
                # print("reset select style = train")
                q_sample = self.sample_q(len(env_idx))
                states = torch.zeros((len(env_idx), 4), device=self.device)
                for i in range(len(env_idx)):
                    u = np.random.uniform()
                    if u > self.start_state_bias:
                        dist = torch.linalg.norm(self.reset_state_buf[:, :2] - q_sample[i, :], dim=1)
                        nearest_idx = int(torch.argmin(dist))
                        nearest_state = self.reset_state_buf[nearest_idx]
                        states[i] = nearest_state
                    else:
                        states[i, 0], states[i, 1] = (
                            self.dof_pos_start[0],
                            self.dof_pos_start[1],
                        )
                # print(states)
        elif self.reset_dist_type == "eval":
            self.success_rate.update(self.success[env_idx])
            self.extras["success_rate"] = self.success_rate.get_mean()
            self.success[env_idx] = 0.0

            states = torch.zeros((len(env_idx), 4), device=self.device)
            states[:, 0] = self.dof_pos_start[0]
            states[:, 1] = self.dof_pos_start[1]
            # q_sample = self.sample_q(len(env_idx))
            # states = torch.zeros((len(env_idx),4), device=self.device)
            # states[:, :2] = q_sample

        if not self.headless:
            pass
            # self.gym.clear_lines(self.viewer)
            # for state in states:
            #     p = state.squeeze(0).cpu().numpy()
            #     self.gym.add_lines(self.viewer, self.envs[0], 1, [p[0]-0.01, p[1], 0.01, p[0] + 0.01, p[1], 0.01], [0, 0, 1])
            #     self.gym.add_lines(self.viewer, self.envs[0], 1, [p[0], p[1]- 0.01, 0.01, p[0], p[1] + 0.01, 0.01], [0, 0, 1])

            # self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                g = self.goal[i].cpu().numpy()
                draw_plus(self.gym, self.viewer, self.envs[i], g)

        self.set_env_states(states, env_idx)
        self.reset_buf[env_idx] = 0.0
        self.progress_buf[env_idx] = 0.0

    """ RRT methods """

    def sample_q(self, num_samples):
        """ Sampling space now contains both position and velocity"""
        # Sample random positions
        u = torch_rand_float(0, 1.0, (num_samples, 2), device=self.device)
        w_max = to_torch([self.x_dof_lim[1], self.y_dof_lim[1]])
        w_min = to_torch([self.x_dof_lim[0], self.y_dof_lim[0]])
        q = (w_max - w_min) * u + w_min
        # Sample random velocities
        u_dot = torch_rand_float(0, 0.5, (num_samples, 2), device=self.device)
        q_dot = u_dot - 0.25
        samples = torch.cat([q, q_dot], dim=1)
        return samples

    def get_env_root_q(self):
        return torch.tensor([self.dof_pos_start[0], self.dof_pos_start[1], 0, 0], device=self.device).unsqueeze(0)

    def get_env_q(self):
        return torch.cat([self.dof_pos.detach().clone(), self.dof_vel.detach().clone()], dim=1)

    def get_env_root_state(self):
        return self.get_env_root_q()

    def get_env_states(self):
        return self.get_env_q()

    def set_env_states(self, states, env_idx: torch.Tensor):
        """Sets the state of the envs specified by env_idx"""

        if len(states.shape) == 2:
            dof_pos, dof_vel, dof_pos_target = (
                states[:, :2],
                states[:, 2:4],
                states[:, :2],
            )

        elif len(states.shape) == 1:
            dof_pos = states[:2].repeat(len(env_idx), 1)
            dof_vel = states[2:4].repeat(len(env_idx), 1)
            dof_pos_target = states[:2].repeat(len(env_idx), 1)

        self.dof_pos[env_idx, :] = dof_pos
        self.dof_vel[env_idx, :] = dof_vel
        self.dof_pos_target[env_idx, :] = dof_pos_target

        env_idx_int32 = env_idx.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_idx_int32),
            len(env_idx),
        )

    def set_goal(self, goals):
        self.goal[:] = goals

    def q_to_goal(self, q):
        goal = q[:2]
        return goal

    def sample_random_config(self, n_samples):
        return self.sample_q(n_samples)

    def compute_distances_in_goal(self, rrt_states, goal=None):
        node_pos = rrt_states[:, 0:2].view(-1, 2)  # extract pos
        goal_pos = node_pos[0].unsqueeze(0).repeat(len(node_pos), 1) if goal is None else goal
        distances = torch.linalg.norm(node_pos - goal_pos[:2], dim=1)
        return distances

    def compute_distance(self, node, node_set):
        """ computes distance from node to each element in node set """
        # Position distance
        pos_dist = 1.0 * torch.linalg.norm(node_set[:, :2] - node[:2], dim=1)
        # Velocity distance
        vel_dist = 0.05 * torch.linalg.norm(node_set[:, 2:4] - node[2:4], dim=1)
        dist = pos_dist + vel_dist
        return dist

    # PRM methods

    def check_collision(self, q_sample):
        pass

    def sample_initial_nodes(self, select_style="naive", num_init_nodes=32):
        print("Debug: reset_state_buf shape: ", self.reset_state_buf.shape)
        num_init_nodes = torch.tensor(list(range(num_init_nodes)))
        if select_style == "naive":
            buf_idx = torch.randint_like(num_init_nodes, len(self.reset_state_buf))
            states = self.reset_state_buf[buf_idx]
        elif select_style == "nearest":
            q_sample = self.sample_q(len(num_init_nodes))
            states = torch.zeros((len(num_init_nodes), 4), device=self.device)
            for i in range(len(num_init_nodes)):
                u = np.random.uniform()
                if u > self.start_state_bias:
                    dist = torch.linalg.norm(self.reset_state_buf[:, :2] - q_sample[i, :], dim=1)
                    nearest_idx = int(torch.argmin(dist))
                    nearest_state = self.reset_state_buf[nearest_idx]
                    states[i] = nearest_state
                else:
                    states[i, 0], states[i, 1] = (
                        self.dof_pos_start[0],
                        self.dof_pos_start[1],
                    )
        return states

    def sample_close_nodes(self, node_set):
        sampled_nodes = []
        for node in node_set:
            # random_idx = torch.randint(0, len(self.reset_state_buf), (1,)).item()
            # sampled_nodes.append(self.reset_state_buf[random_idx][:2])
            dist = self.compute_distances_in_goal(self.reset_state_buf, node)
            close_idx = torch.where(dist < 0.2)[0]
            # Randomly select one of the close nodes
            random_idx = close_idx[torch.randint(0, len(close_idx), (1,)).item()]
            sampled_nodes.append(self.reset_state_buf[random_idx])

        return sampled_nodes


def states_from_path(path: np.ndarray, delta=0.01, device="cuda:0"):
    delta = 0.01
    states = []
    for i in range(len(path) - 1):
        wp1 = path[i + 1]
        wp0 = path[i]
        seglen = np.linalg.norm(wp1 - wp0)
        num_states_seg = int(seglen / delta)
        states.append([wp0[0], wp0[1], 0, 0])
        for j in range(num_states_seg):
            point = j * (wp1 - wp0) / num_states_seg + wp0
            state = [point[0], point[1], 0, 0]
            states.append(state)
        states.append([wp1[0], wp1[1], 0, 0])
    states = to_torch(states, device=device)
    return states


def draw_plus(gym, viewer, env, p, color="r"):
    cmap = {"r": [1, 0, 0], "g": [0, 1, 0], "b": [0, 0, 1]}
    c = cmap[color]
    gym.add_lines(viewer, env, 1, [p[0] - 0.01, p[1], 0.01, p[0] + 0.01, p[1], 0.01], c)
    gym.add_lines(viewer, env, 1, [p[0], p[1] - 0.01, 0.01, p[0], p[1] + 0.01, 0.01], c)
