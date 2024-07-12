import torch


class PRM:
    """ Prbabilistic Roadmap (PRM) planner for sampling and planning in the task space """
    def __init__(self, cfg, env, model, obs_rms, state_rms, value_rms, device):
        self.cfg = cfg
        self.env = env
        self.model = model
        self.obs_rms = obs_rms
        self.state_rms = state_rms
        self.value_rms = value_rms
        self.device = device

        # PRM config
        self.init_samples = cfg["init_samples"]  # number of initial samples
        self.prm_samples_per_epoch = cfg["samples_per_epoch"]  # number of samples per epoch
        assert self.env.num_envs % self.prm_samples_per_epoch == 0
        self.envs_per_sample = 1  # one environment per sample
        self.prm_rollout_len = cfg["rollout_len"]
        self.prm_local_planner = cfg["local_planner"]  # "random" or "policy
        self.visualize_prm = cfg["visualize_prm"]

        # PRM data
        self.prm_q = None
        self.prm_states = None
        self.prm_children = []  # List of "list of children" (index) for each node
        self.prm_parents = []  # List of "list of parent" (index) of each node
        self.prm_actions = torch.zeros((1, self.env.num_actions), device=self.device)  # store actions

        # Temporary variables for building the PRM
        self.chosen_nodes_idx = [0] * self.prm_samples_per_epoch
        self.num_new_nodes = 0

    def initPRM(self):
        self.env.enable_reset = False

        # Sample initial state in the task space
        states = self.env.sample_initial_nodes(num_init_nodes=self.init_samples)

        for state in states:
            if self.prm_q is None:  # Initialize the PRM if it is empty
                self.prm_q = state.unsqueeze(0)
                self.prm_states = state.unsqueeze(0)
                self.prm_parents.append([])
                self.prm_children.append([])
            else:
                # compute distance from qbest to other nodes in PRM
                dist_to_qbest = self.env.compute_distance(node=state, node_set=self.prm_q)
                if torch.min(dist_to_qbest) < 0.05:  # If the node is too close to the existing nodes, skip
                    continue
                else:  # Otherwise, add the node to PRM
                    self.prm_q = torch.cat([self.prm_q, state.unsqueeze(0)])
                    self.prm_states = torch.cat([self.prm_states, state.unsqueeze(0)])
                    self.prm_parents.append([])
                    self.prm_children.append([])

    def runPRM(self):
        print("*** PRM nodes: ", self.prm_states.shape[0], "***")
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
        # Randomly sample current nodes from the PRM
        self.chosen_nodes_idx = torch.randint_like(torch.tensor(list(range(self.prm_samples_per_epoch))), len(self.prm_states))
        chosen_nodes = self.prm_states[self.chosen_nodes_idx]

        # Sample goal states in the task space that are close to the chosen nodes
        self.q_sample = self.env.sample_close_nodes(node_set=chosen_nodes)

        # Check whether the new nodes lie in the free space or not
        self.env.check_collision(self.q_sample)
        self.num_new_nodes = len(self.q_sample)  # store the number of new nodes in case of collision

        # Set the environment states and goals
        # Initialize empty states and goals
        chosen_states = torch.zeros((self.env.num_envs, self.env.reset_state_dim), device=self.device)
        if hasattr(self.env, "goal"):
            goals = torch.zeros_like(self.env.goal)

        # Set the environment states to the chosen nodes
        for i in range(self.num_new_nodes):
            chosen_states[self.envs_per_sample * i : self.envs_per_sample * (i + 1)] = chosen_nodes[i]
            # Set the goals to the sampled q_space goals
            if hasattr(self.env, "goal"):
                goals[self.envs_per_sample * i : self.envs_per_sample * (i + 1)] = self.env.q_to_goal(self.q_sample[i])
        self.env.set_env_states(chosen_states, torch.arange(self.env.num_envs, device=self.device))
        if hasattr(self.env, "goal"):
            self.env.set_goal(goals)

        # Reset the environment
        self.env.progress_buf[:] = 0
        self.env.reset_buf[:] = 0

    def plan_steps(self):
        for k in range(self.prm_rollout_len):
            # Sample actions randomly
            actions = self.env.random_actions()

            # Clamp sampled actions and step the environment
            actions = torch.clamp(actions, -1.0, 1.0)

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

        # Grow PRM with the reached states (x_end)
        for sample_idx in range( self.num_new_nodes):
            result = self._add_nodes_for_sample(reached_states, reached_q, sample_idx)
            if result is not None:
                state_parent, state_best = result
                self._visualize_new_edges(state_parent, state_best)  # Debug visualization for MazeBot task only

    def _add_nodes_for_sample(self, reached_states, reached_q, sample_idx):
        i = sample_idx
        # Fetch the batch of reached states and q
        batch = range(self.envs_per_sample * i, self.envs_per_sample * (i + 1))
        states_batch = reached_states[batch]
        qbatch = reached_q[batch]

        # Compute distance to the sampled q-space goal
        dist = self.env.compute_distance(node=self.q_sample[i], node_set=qbatch)
        # Find the node with the minimum distance to goal
        best_idx = torch.argmin(dist)
        qbest = qbatch[best_idx]

        # compute distance from qbest to other nodes in PRM
        dist_to_qbest = self.env.compute_distance(node=qbest, node_set=self.prm_q)

        if torch.min(dist_to_qbest) < 0.05:  # if the node is too close to the existing nodes, we only add the edge
            closest_idx = torch.argmin(dist_to_qbest)
            parent_idx = self.chosen_nodes_idx[i]
            state_parent = self.prm_states[parent_idx].unsqueeze(0)
            state_best = self.prm_states[closest_idx].unsqueeze(0)
            # Update the parent list to add the new parent
            self.prm_parents[closest_idx].append(parent_idx)

        else:  # Otherwise, we add both the node and the edge
            parent_idx = self.chosen_nodes_idx[i]
            self.prm_q = torch.cat([self.prm_q, qbest.unsqueeze(0)])
            state_best = states_batch[best_idx].unsqueeze(0)
            self.prm_states = torch.cat([self.prm_states, state_best])
            state_parent = self.prm_states[parent_idx].unsqueeze(0)
            # Update the parent and children list
            self.prm_parents.append([parent_idx])
            self.prm_children.append([])
            self.prm_children[parent_idx].append(len(self.prm_q) - 1)

        return state_parent, state_best


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
