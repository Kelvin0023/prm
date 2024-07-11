import torch


class PRM:
    def __init__(self, cfg, env, model, obs_rms, state_rms, value_rms, device):
        self.cfg = cfg
        self.env = env
        self.model = model
        self.obs_rms = obs_rms
        self.state_rms = state_rms
        self.value_rms = value_rms
        self.device = device

        # PRM config
        self.init_samples = 64
        self.prm_samples_per_epoch = 2048 # 32
        assert self.env.num_envs % self.prm_samples_per_epoch == 0
        # self.envs_per_sample = self.env.num_envs // self.prm_samples_per_epoch
        self.envs_per_sample = 1 # Only one environment per sample
        self.prm_rollout_len = 8
        self.prm_local_planner = "random"  # "random" or "policy
        self.visualize_prm = True


        self.prm_q = self.env.get_env_root_q()
        self.prm_states = self.env.get_env_root_state()
        self.prm_children = [[]]  # List of "list of children" (index) for each node
        self.prm_parents = [[]]  # List of parent (index) of each node
        self.prm_actions = torch.zeros((1, self.env.num_actions), device=self.device)  # store actions
        self.prm_rewsum = torch.zeros((1, 1), device=self.device)
        self.prm_pathlen = torch.ones((1, 1), device=self.device)

        self.invalid_buf = torch.zeros(self.env.num_envs, device=self.device)

        # Temporary variables for building tree
        self.nearest_idx = [0] * self.prm_samples_per_epoch
        self.sampled_actions = None
        self.node_distances_to_root = [0.0]

    def initPRM(self):
        self.env.enable_reset = False

        # Sample initial state in the task space
        states = self.env.sample_initial_nodes(num_init_nodes=self.init_samples)
        self.prm_q = states
        self.prm_states = states

        for _ in range(self.init_samples):
            self.prm_parents.append([])
            self.prm_children.append([])

    def runPRM(self):
        # reset invalid buffer
        self.invalid_buf[:] = 0.0

        # print("*** PRM step: ", steps, "***")
        print("PRM states: ", self.prm_states.shape[0])
        # Sample new nodes and perform collision check
        self.sample_and_set()
        self.env.refresh_tensors()
        self.env.compute_obs()
        # Rollout for k steps
        self.plan_steps()
        self.add_nodes()

        # Update the reset distribution
        self.env.set_reset_state_buf(self.prm_states)


    def sample_and_set(self):
        self.nearest_idx = torch.randint_like(torch.tensor(list(range(self.prm_samples_per_epoch))), len(self.prm_states))

        chosen_node = self.prm_states[self.nearest_idx]
        self.q_sample = self.env.sample_close_nodes(node_set=chosen_node)
        # Check whether the new nodes lie in the free space or not
        self.env.check_collision(self.q_sample)

        chosen_states = torch.zeros((self.env.num_envs, self.env.reset_state_dim), device=self.device)

        if hasattr(self.env, "goal"):
            goals = torch.zeros_like(self.env.goal)
        for i in range(self.prm_samples_per_epoch):
            chosen_states[self.envs_per_sample * i : self.envs_per_sample * (i + 1)] = chosen_node[i]
            if hasattr(self.env, "goal"):
                goals[self.envs_per_sample * i : self.envs_per_sample * (i + 1)] = self.env.q_to_goal(self.q_sample[i])
        self.env.set_env_states(chosen_states, torch.arange(self.env.num_envs, device=self.device))
        if hasattr(self.env, "goal"):
            self.env.set_goal(goals)
        self.env.progress_buf[:] = 0
        self.env.reset_buf[:] = 0

    def model_act(self, obs_dict, actions=None):
        processed_obs = self.obs_rms(obs_dict["obs"])
        processed_states = self.state_rms(obs_dict["states"])
        input_dict = {"obs": processed_obs, "states": processed_states}
        res_dict = self.model.act(input_dict, actions)
        res_dict["values"] = self.value_rms(res_dict["values"], True)
        return res_dict

    def plan_steps(self):
        for k in range(self.prm_rollout_len):
            obses = self.env.get_obs()
            states = self.env.get_state()

            obs_dict = {"obs": obses, "states": states}

            # Sample actions from policy or randomly and get value
            actions = self.env.random_actions()

            # Clamp sampled actions and step the environment
            actions = torch.clamp(actions, -1.0, 1.0)

            # # store intermediate obs and actions
            # self.obs_buf[k] = obses
            # self.act_buf[k] = actions
            # self.states_buf[k] = states

            self.env.set_actions(actions)
            self.env.gym.simulate(self.env.sim)
            self.env.refresh_tensors()
            self.env.compute_obs()
            self.env.compute_reward()

            if not self.env.headless and self.env.force_render:
                self.env.render()

    def add_nodes(self):
        """Add nodes based on q-sampled"""
        # Collect the reached nodes
        reached_states = self.env.get_env_states()
        reached_q = self.env.get_env_q()
        # Evaluate if the final states is valid (all the intermediate state must be valid too)
        invalid = torch.logical_or(self.env.check_constraints(), self.invalid_buf)

        # Grow tree if the states reached satisfy constraints
        for sample_idx in range(self.prm_samples_per_epoch):
            result = self._add_nodes_for_sample(reached_states, reached_q, invalid, sample_idx)
            if result is not None:
                state_parent, state_best = result
                self._visualize_new_edges(state_parent, state_best)  # Debug visualization for MazeBot task only

    def _add_nodes_for_sample(self, reached_states, reached_q, invalid, sample_idx):
        i = sample_idx
        # Compute distance to the sampled q-space goals
        # batch = range(self.envs_per_sample * i, self.envs_per_sample * (i + 1))
        batch = i
        states_batch = reached_states[batch]
        # actions_batch = self.sampled_actions[batch]
        qbatch = reached_q[batch]

        # # Compute distance to the sampled q-space goals
        # dist = self.env.compute_distance(node=self.q_sample[i], node_set=qbatch)

        # Find the invalid index in the batch
        invalid_idx = invalid[batch].nonzero(as_tuple=False).squeeze(-1)

        # Now, finally we grow the tree
        # print("Adding nodes to tree")
        # if len(invalid_idx) < len(invalid[batch]):
        if len(invalid_idx) < 1:
            # dist[invalid_idx] = torch.inf
            # best_idx = torch.argmin(dist)
            # qbest = qbatch[best_idx]
            qbest = qbatch

            # compute distance from qbest to other nodes in PRM
            dist_to_qbest = self.env.compute_distance(node=qbest, node_set=self.prm_q)

            if torch.min(dist_to_qbest) < 0.05:
                closest_idx = torch.argmin(dist_to_qbest)
                parent_idx = self.nearest_idx[i]

                state_parent = self.prm_states[parent_idx].unsqueeze(0)
                state_best = self.prm_states[closest_idx].unsqueeze(0)
                self.prm_parents[closest_idx].append(parent_idx)

            else:
                parent_idx = self.nearest_idx[i]
                self.prm_q = torch.cat([self.prm_q, qbest.unsqueeze(0)])
                # state_best = states_batch[best_idx].unsqueeze(0)
                state_best = states_batch.unsqueeze(0)

                state_parent = self.prm_states[parent_idx].unsqueeze(0)
                self.prm_states = torch.cat([self.prm_states, state_best])
                self.prm_parents.append([parent_idx])
                self.prm_children.append([])
                self.prm_children[parent_idx].append(len(self.prm_q) - 1)

        return state_parent, state_best


    # def _add_nodes_for_sample(self, reached_states, reached_q, invalid, sample_idx):
    #     i = sample_idx
    #     # Compute distance to the sampled q-space goals
    #     batch = range(self.envs_per_sample * i, self.envs_per_sample * (i + 1))
    #     states_batch = reached_states[batch]
    #     # actions_batch = self.sampled_actions[batch]
    #     qbatch = reached_q[batch]
    #
    #     # Compute distance to the sampled q-space goals
    #     dist = self.env.compute_distance(node=self.q_sample[i], node_set=qbatch)
    #
    #     # Find the invalid index in the batch
    #     invalid_idx = invalid[batch].nonzero(as_tuple=False).squeeze(-1)
    #
    #     # Now, finally we grow the tree
    #     # print("Adding nodes to tree")
    #     if len(invalid_idx) < len(invalid[batch]):
    #         dist[invalid_idx] = torch.inf
    #         best_idx = torch.argmin(dist)
    #         qbest = qbatch[best_idx]
    #
    #         # compute distance from qbest to other nodes in PRM
    #         dist_to_qbest = self.env.compute_distance(node=qbest, node_set=self.prm_states)
    #
    #         if torch.min(dist_to_qbest) < 0.05:
    #             closest_idx = torch.argmin(dist_to_qbest)
    #             parent_idx = self.nearest_idx[i]
    #
    #             state_parent = self.prm_states[parent_idx].unsqueeze(0)
    #             state_best = self.prm_states[closest_idx].unsqueeze(0)
    #             # self.prm_states = torch.cat([self.prm_states, state_closest])
    #             self.prm_parents[closest_idx].append(parent_idx)
    #
    #         else:
    #             parent_idx = self.nearest_idx[i]
    #             self.prm_q = torch.cat([self.prm_q, qbest.unsqueeze(0)])
    #             state_best = states_batch[best_idx].unsqueeze(0)
    #
    #             state_parent = self.prm_states[parent_idx].unsqueeze(0)
    #             self.prm_states = torch.cat([self.prm_states, state_best])
    #             self.prm_parents.append([parent_idx])
    #             self.prm_children.append([])
    #             self.prm_children[parent_idx].append(len(self.prm_q) - 1)
    #
    #     return state_parent, state_best

    def _visualize_new_edges(self, state_parent, state_best):
        """
        Debug visualization: Draw edge for visualization (works only for maze env)
        """

        task_name = self.env.__class__.__name__
        if task_name == "MazeBot" and not self.env.headless and self.visualize_prm:
            p = state_best[0].cpu().numpy()
            parent = state_parent[0].cpu().numpy()

            # draw edge
            edge = [parent[0], parent[1], 0.01, p[0], p[1], 0.01]
            self.env.gym.add_lines(self.env.viewer, self.env.envs[0], 1, edge, [0.5, 0.5, 0.5])
            # draw "+" marker for the new node
            hline = [p[0] - 0.01, p[1], 0.01, p[0] + 0.01, p[1], 0.01]
            self.env.gym.add_lines(self.env.viewer, self.env.envs[0], 1, hline, [1, 1, 0])
            vline = [p[0], p[1] - 0.01, 0.01, p[0], p[1] + 0.01, 0.01]
            self.env.gym.add_lines(self.env.viewer, self.env.envs[0], 1, vline, [1, 1, 0])

        if task_name == "Ant" and not self.env.headless and self.visualize_prm:
            p = state_best[0].cpu().numpy()
            parent = state_parent[0].cpu().numpy()

            # draw edge
            edge = [parent[0], parent[1], 0.1, p[0], p[1], 0.1]
            self.env.gym.add_lines(self.env.viewer, self.env.envs[0], 1, edge, [0.5, 0.5, 0.5])
            # draw "+" marker for the new node
            hline = [p[0] - 0.1, p[1], 0.1, p[0] + 0.1, p[1], 0.1]
            self.env.gym.add_lines(self.env.viewer, self.env.envs[0], 1, hline, [1, 1, 0])
            vline = [p[0], p[1] - 0.1, 0.1, p[0], p[1] + 0.1, 0.1]
            self.env.gym.add_lines(self.env.viewer, self.env.envs[0], 1, vline, [1, 1, 0])
