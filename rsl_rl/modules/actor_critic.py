# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import copy
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        actor_input_dim=None,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.obs_slices = {
            "goal": slice(0, 3),
            "curr_proprio_clean": slice(3, 45),
            "curr_proprio_noisy": slice(45, 87),
            "proprio_history": slice(87, 507),
            "base_lin_vel": slice(507, 510),
            "foot_contacts": slice(510, 514),
            "friction": slice(514, 515),
            "added_mass": slice(515, 516),
            "p_gain": slice(516, 517),
            "d_gain": slice(517, 518),
            "applied_force": slice(518, 521),
            "applied_torque": slice(521, 524),
            "feet_air_time": slice(524, 528),
            "phase_feet_ground_time": slice(528, 532),
            "height_scan": slice(532, 763),
            "height_scan_simplified": slice(763, 994),
            "ladder_info": slice(994, 999),
            "depth_image": slice(999, 3303),
        }

        self.proprio_dim = 42
        self.goal_dim = 3
        self.ladder_info_dim = 5
        self.privileged_dim = 30
        self.height_dim = 231
        self.height_latent_dim = 32

        self.critic_height_encoder = self._build_height_encoder(activation)

        if actor_input_dim is None:
            actor_input_dim = self.proprio_dim + self.goal_dim
        mlp_input_dim_a = actor_input_dim
        mlp_input_dim_c = self.proprio_dim + self.goal_dim + self.privileged_dim + self.height_latent_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Critic height encoder: {self.critic_height_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _build_height_encoder(self, activation):
        return nn.Sequential(
            nn.Linear(self.height_dim, 128),
            self._clone_activation(activation),
            nn.Linear(128, 64),
            self._clone_activation(activation),
            nn.Linear(64, self.height_latent_dim),
        )

    def _clone_activation(self, activation):
        if isinstance(activation, str):
            return get_activation(activation)
        return copy.deepcopy(activation)
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _split_observations(self, observations):
        return {name: observations[..., obs_slice] for name, obs_slice in self.obs_slices.items()}

    def _build_critic_input(self, observations):
        obs = self._split_observations(observations)
        privileged = self._build_privileged(obs)
        height_latent = self.critic_height_encoder(obs["height_scan"])
        return torch.cat([obs["curr_proprio_clean"], obs["goal"], privileged, height_latent], dim=-1)

    def _build_privileged(self, obs):
        return torch.cat(
            [
                obs["base_lin_vel"],
                obs["foot_contacts"],
                obs["friction"],
                obs["added_mass"],
                obs["p_gain"],
                obs["d_gain"],
                obs["applied_force"],
                obs["applied_torque"],
                obs["feet_air_time"],
                obs["phase_feet_ground_time"],
                obs["ladder_info"],
            ],
            dim=-1,
        )

    def update_distribution(self, observations):
        actor_input = self._build_actor_input(observations)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_input = self._build_actor_input(observations)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, **kwargs):
        critic_input = self._build_critic_input(critic_observations)
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories
            critic_input = unpad_trajectories(critic_input, masks)
        value = self.critic(critic_input)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
