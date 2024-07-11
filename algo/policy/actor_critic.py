import inspect
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop("actions_num")
        obs_shape = kwargs.pop("obs_shape")
        state_shape = kwargs.pop("state_shape")
        self.actor_units = kwargs.pop("actor_units")
        self.critic_units = kwargs.pop("critic_units")

        out_size = self.actor_units[-1]
        self.actor_mlp = MLP(units=self.actor_units, input_size=obs_shape[0])
        self.critic_mlp = MLP(units=self.critic_units, input_size=state_shape[0])
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
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

    @torch.no_grad()
    def act(self, obs_dict, actions=None):
        """used specifically to collect samples during training
        it contains exploration so needs to sample from distribution"""
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        if actions is None:
            selected_action = distr.sample()
            neglogpacs = -1.0 * distr.log_prob(selected_action).sum(dim=-1)
        else:
            selected_action = actions
            neglogpacs = -1.0 * distr.log_prob(actions).sum(dim=-1)
        result = {
            "neglogpacs": neglogpacs,
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        """used for evaluation"""
        mu, _, _ = self._actor_critic(obs_dict)
        return mu

    def act_bc(self, obs_dict):
        """used for behavioral cloning"""
        mu, _ = self._actor(obs_dict)
        return mu

    def _actor(self, obs_dict):
        obs = obs_dict["obs"]
        x = self.actor_mlp(obs)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma

    def _actor_critic(self, obs_dict):
        obs = obs_dict["obs"]
        state = obs_dict["states"]
        x = self.actor_mlp(obs)
        value = self.value(self.critic_mlp(state))
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value

    @torch.no_grad()
    def critic_sample(self, state):
        value = self.value(self.critic_mlp(state))
        return value

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
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
