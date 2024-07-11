import torch


class Trainer:
    def __init__(self, cfg, model, obs_rms, state_rms, value_rms, device):

        self.cfg = cfg
        self.model = model
        self.obs_rms = obs_rms
        self.state_rms = state_rms
        self.value_rms = value_rms
        self.device = device

        self.last_lr = cfg["learning_rate"]
        self.weight_decay = cfg["weight_decay"]
        self.kl_threshold = cfg["kl_threshold"]

        # epochs
        self.actor_mini_epochs = cfg["actor_mini_epochs"]
        self.critic_mini_epochs = cfg["critic_mini_epochs"]
        self.bc_mini_epochs = cfg["bc_mini_epochs"]

        # batch sizes
        self.num_mini_batches = cfg["num_mini_batches"]
        self.bc_batch_size = cfg["bc_batch_size"]

        self.e_clip = cfg["e_clip"]
        self.grad_norm = cfg["grad_norm"]
        self.truncate_grads = cfg["truncate_grads"]

        # loss coefficients
        self.bounds_loss_coef = cfg["bounds_loss_coef"]
        self.critic_coef = cfg["critic_coef"]
        self.entropy_coef = cfg["entropy_coef"]
        self.bc_coef = cfg["bc_coef"]

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.last_lr, weight_decay=self.weight_decay
        )
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        self.criterion = torch.nn.MSELoss()

    def train_rl(self, storage):
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        storage.set_minibatch_size(storage.batch_size // self.num_mini_batches)
        for _ in range(0, self.actor_mini_epochs):
            ep_kls = []
            for i in range(len(storage)):
                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    state,
                ) = storage[i]

                obs = self.obs_rms(obs)
                batch_dict = {"prev_actions": actions, "obs": obs, "states": state}
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                loss = (
                        a_loss
                        + 0.5 * c_loss * self.critic_coef
                        - entropy * self.entropy_coef
                        + b_loss * self.bounds_loss_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                # print("debug: computing kl distance")
                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.last_lr
            kls.append(av_kls)

        return a_losses, c_losses, b_losses, entropies, kls, self.last_lr

    def train_bc_cvae(self, obs_buf, act_buf):
        batch = torch.tensor(list(range(self.bc_batch_size)))
        losses = []
        for _ in range(self.bc_mini_epochs):
            # Sample obs and act from the buffer
            sampled_idx = torch.randint_like(batch, obs_buf.shape[0])
            sampled_obs = obs_buf[sampled_idx]
            sampled_acts = act_buf[sampled_idx]

            obs = self.obs_rms(sampled_obs)
            input_dict = {"obs": obs, "actions": sampled_acts}
            return_dict = self.model._actor(input_dict)
            recon_action, mu, logvar = return_dict["mus"], return_dict["fc_mu"], return_dict["fc_logvar"]
            bc_loss = self.criterion(recon_action, sampled_acts)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bc_loss + KLD
            loss = torch.mean(loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm
                )
            self.optimizer.step()
            losses.append(loss)

        return losses

    def train_bc(self, obs_buf, act_buf):
        batch = torch.tensor(list(range(self.bc_batch_size)))
        losses = []
        for _ in range(self.bc_mini_epochs):
            # Sample obs and act from the buffer
            sampled_idx = torch.randint_like(batch, obs_buf.shape[0])
            sampled_obs = obs_buf[sampled_idx]
            sampled_acts = act_buf[sampled_idx]

            obs = self.obs_rms(sampled_obs)
            input_dict = {"obs": obs}
            mu = self.model.act_bc(input_dict)
            loss = self.criterion(mu, sampled_acts)
            loss = torch.mean(loss)

            self.optimizer.zero_grad()
            loss.backward()
            if self.truncate_grads:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm
                )
            self.optimizer.step()
            losses.append(loss)

        return losses

    def train_dapg_cvae(self, storage, obs_buf, act_buf, lambda1):
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        bc_losses = []

        # set up batch for bc training
        bc_batch_size = torch.tensor(list(range(self.bc_batch_size)))

        storage.set_minibatch_size(storage.batch_size // self.num_mini_batches)
        for _ in range(self.bc_mini_epochs):
            ep_kls = []
            for i in range(len(storage)):
                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    state,
                ) = storage[i]

                obs = self.obs_rms(obs)
                batch_dict = {"prev_actions": actions, "obs": obs, "states": state}
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                # bc loss (MSE)
                # Sample obs and act from the buffer
                sampled_idx = torch.randint_like(bc_batch_size, obs_buf.shape[0])
                sampled_obs = obs_buf[sampled_idx]
                sampled_acts = act_buf[sampled_idx]

                input_obs = self.obs_rms(sampled_obs)
                input_dict = {"obs": input_obs, "actions": sampled_acts}
                return_dict = self.model._actor(input_dict)
                recon_action, bc_mu, bc_logvar = return_dict["mus"], return_dict["fc_mu"], return_dict["fc_logvar"]
                bc_loss = self.criterion(recon_action, sampled_acts)
                KLD = -0.5 * torch.sum(1 + bc_logvar - bc_mu.pow(2) - bc_logvar.exp())
                bc_loss = bc_loss + KLD
                bc_loss = torch.mean(bc_loss)

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                    + lambda1 * bc_loss * self.bc_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                # print("debug: computing kl distance")
                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                bc_losses.append(bc_loss)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.last_lr
            kls.append(av_kls)

        return bc_losses, a_losses, c_losses, b_losses, entropies, kls, self.last_lr

    def train_dapg(self, storage, obs_buf, act_buf, lambda1):
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        bc_losses = []

        # set up batch for bc training
        bc_batch_size = torch.tensor(list(range(self.bc_batch_size)))

        storage.set_minibatch_size(storage.batch_size // self.num_mini_batches)
        for _ in range(self.bc_mini_epochs):
            ep_kls = []
            for i in range(len(storage)):
                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    state,
                ) = storage[i]

                obs = self.obs_rms(obs)
                batch_dict = {"prev_actions": actions, "obs": obs, "states": state}
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                # bc loss (MSE)
                # Sample obs and act from the buffer
                sampled_idx = torch.randint_like(bc_batch_size, obs_buf.shape[0])
                sampled_obs = obs_buf[sampled_idx]
                sampled_acts = act_buf[sampled_idx]

                input_obs = self.obs_rms(sampled_obs)
                input_dict = {"obs": input_obs}
                bc_predicted_mu = self.model.act_bc(input_dict)
                bc_loss = self.criterion(bc_predicted_mu, sampled_acts)
                bc_loss = torch.mean(bc_loss)

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                    + lambda1 * bc_loss * self.bc_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                # print("debug: computing kl distance")
                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                bc_losses.append(bc_loss)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.last_lr
            kls.append(av_kls)

        return bc_losses, a_losses, c_losses, b_losses, entropies, kls, self.last_lr



def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr