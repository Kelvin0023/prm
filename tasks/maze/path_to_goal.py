# Configure goal, reset and reward based on the difficulty of the maze
# self.reset_state_buf = torch.tensor([[self.dof_pos_start[0], self.dof_pos_start[1], 0, 0]], device=self.device)
# path_to_goal = None # custom reset distribution will be computed based on this
# if self.lvl == 0:
#     self.goal = to_torch([0.4, 0.4], device=self.device)
#     path_to_goal = np.array([[-0.4, -0.4], [-0.35, 0.4], [-0., 0.4], [-0., -0.4], [0., -0.4], [self.x_dof_lim[1]-0.1, 0.0]])
#
# if self.lvl == 1:
#     self.goal = to_torch([self.x_dof_lim[1]-0.1, 0.0], device=self.device)
#     path_to_goal = np.array([[-0.4,-0.4], [-0.35, 0.4], [-0.,0.4], [-0., -0.4], [0., -0.4], [self.x_dof_lim[1]-0.1, 0.0]])
#
# elif self.lvl == 2:
#     self.goal = to_torch([self.x_dof_lim[1]-0.1, 0.0], device=self.device)
#     path_to_goal = np.array([[-0.4,-0.4], [-0.35, 0.4], [-0.1, 0.4], [-0.1, -0.4], [0., -0.4], [0.15, -0.4], [0.15, 0.4], [0.2, 0.4], [self.x_dof_lim[1]-0.1, 0.0]])

# path_to_goal = None
# if self.use_custom_reset:
    # assert path_to_goal is not None, "path to goal is not defined but it is required \
    #     for computing the custom reset states"
    # self.reset_state_buf = states_from_path(path=path_to_goal, device=self.device)
    # self.reset_state_buf = torch.zeros((5000, self.reset_state_dim), device=self.device)
    # self.reset_state_buf[:, 0] = torch_rand_float(self.x_dof_lim[0], self.x_dof_lim[1], (5000,1), device=self.device)[:,0]
    # self.reset_state_buf[:, 1] = torch_rand_float(self.y_dof_lim[0], self.y_dof_lim[1], (5000,1), device=self.device)[:,0]
    # self.reset_state_buf[:, 2:4] = 0.0