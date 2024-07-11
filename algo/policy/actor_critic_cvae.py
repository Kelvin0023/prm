import inspect
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.policy.actor_critic import MLP


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, actor_units, latent_dim):
        super(Encoder, self).__init__()
        self.actor_units = actor_units
        out_size = self.actor_units[-1]
        self.fc1 = MLP(units=self.actor_units, input_size=state_dim + action_dim)
        self.fc_mu = nn.Linear(out_size, latent_dim)
        self.fc_logvar = nn.Linear(out_size, latent_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim, actor_units, latent_dim):
        super(Decoder, self).__init__()
        self.actor_units = actor_units
        out_size = self.actor_units[-1]
        self.fc1 = MLP(units=self.actor_units, input_size=state_dim + latent_dim)
        self.fc2 = nn.Linear(out_size, action_dim)
        self.sigma = nn.Parameter(
            torch.zeros(action_dim, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    def forward(self, state, z):
        x = torch.cat([state, z], dim=1)
        x = self.fc1(x)
        mu = self.fc2(x)
        sigma = torch.exp(self.sigma)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            "neglogpacs": -distr.log_prob(selected_action).sum(1),
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result


class CVAE(nn.Module):
    def __init__(self, state_dim, action_dim, actor_units, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(state_dim, action_dim, actor_units, latent_dim)
        self.decoder = Decoder(state_dim, action_dim, actor_units, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        mu, logvar = self.encoder(state, action)
        z = self.reparameterize(mu, logvar)
        return_dict = self.decoder(state, z)
        return_dict["fc_mu"] = mu
        return_dict["fc_logvar"] = logvar
        return return_dict


class ActorCritic_CVAE(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop("actions_num")
        obs_shape = kwargs.pop("obs_shape")
        state_shape = kwargs.pop("state_shape")
        self.actor_units = kwargs.pop("actor_units")
        self.critic_units = kwargs.pop("critic_units")
        self.device = kwargs.pop("device")
        self.latent_dim = kwargs.pop("latent_dim")

        out_size = self.actor_units[-1]
        # Actpr
        self.actor_vae = CVAE(state_dim=obs_shape[0], action_dim=actions_num, actor_units=self.actor_units,
                              latent_dim=self.latent_dim)

        # Critic
        self.critic_mlp = MLP(units=self.critic_units, input_size=state_shape[0])
        self.value = torch.nn.Linear(out_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

    @torch.no_grad()
    def act(self, obs_dict, actions=None):
        """used specifically to collect samples during training
        it contains exploration so needs to sample from distribution"""
        return_dict, value = self._actor_critic(obs_dict)
        return_dict["values"] = value
        if actions is not None:
            mu = return_dict["mus"]
            sigma = return_dict["sigmas"]
            distr = torch.distributions.Normal(mu, sigma)
            neglogpacs = -1.0 * distr.log_prob(actions).sum(dim=-1)
            return_dict["neglogpacs"] = neglogpacs
            return_dict["actions"] = actions
        return return_dict

    @torch.no_grad()
    def act_inference(self, obs_dict):
        """used for evaluation"""
        return_dict, _ = self._actor_critic(obs_dict)
        mu = return_dict["mus"]
        return mu

    def act_bc(self, obs_dict):
        """used for behavioral cloning"""
        return_dict = self._actor(obs_dict)
        mu = return_dict["mus"]
        return mu

    def _actor(self, obs_dict):
        obs = obs_dict["obs"]
        actions = obs_dict["actions"]
        return_dict = self.actor_vae(obs, actions)
        return return_dict

    def _actor_critic(self, obs_dict):
        obs = obs_dict["obs"]
        state = obs_dict["states"]
        z = torch.zeros(obs.shape[0], self.latent_dim).to(self.device)
        return_dict = self.actor_vae.decoder(obs, z)
        value = self.value(self.critic_mlp(state))
        return return_dict, value

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        return_dict, value = self._actor_critic(input_dict)
        mu = return_dict["mus"]
        sigma = return_dict["sigmas"]
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
        }
        return result