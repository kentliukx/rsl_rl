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

from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        history_length=50,
                        history_latent_dim=32,
                        depth_latent_dim=32,
                        mixer_latent_dim=32,
                        rnn_type='gru',
                        rnn_hidden_size=32,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        **kwargs):
        self.history_length = history_length
        self.history_latent_dim = history_latent_dim
        self.depth_latent_dim = depth_latent_dim
        self.mixer_latent_dim = mixer_latent_dim
        self.estimator_dim = 7
        actor_input_dim = 42 + 3 + self.estimator_dim + rnn_hidden_size

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            actor_input_dim=actor_input_dim,
            **kwargs,
        )

        activation_module = get_activation(activation)
        self.proprio_history_len = 50
        self.depth_height = 36
        self.depth_width = 64

        self.history_encoder = self._build_history_encoder(activation_module)
        self.estimator = self._build_estimator(activation_module)
        self.depth_encoder = self._build_depth_encoder(activation_module)
        self.mixer = self._build_mixer(activation_module)
        self.height_decoder = self._build_height_decoder(activation_module, rnn_hidden_size)
        self.reconstructed_height_map = None
        self.memory_a = Memory(
            self.mixer_latent_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )

        print(f"History encoder: {self.history_encoder}")
        print(f"Estimator: {self.estimator}")
        print(f"Actor depth encoder: {self.depth_encoder}")
        print(f"Mixer: {self.mixer}")
        print(f"Actor GRU: {self.memory_a}")
        print(f"Height decoder: {self.height_decoder}")

    def _build_history_encoder(self, activation):
        return nn.Sequential(
            nn.Linear(self.history_length * self.proprio_dim, 256),
            self._clone_activation(activation),
            nn.Linear(256, 128),
            self._clone_activation(activation),
            nn.Linear(128, self.history_latent_dim),
            self._clone_activation(activation),
        )

    def _build_estimator(self, activation):
        return nn.Sequential(
            nn.Linear(self.history_latent_dim, 32),
            self._clone_activation(activation),
            nn.Linear(32, 16),
            self._clone_activation(activation),
            nn.Linear(16, self.estimator_dim),
        )

    def _build_depth_encoder(self, activation):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            self._clone_activation(activation),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            self._clone_activation(activation),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            self._clone_activation(activation),
            nn.Flatten(),
            nn.Linear(64 * 5 * 8, self.depth_latent_dim),
            self._clone_activation(activation),
        )

    def _build_mixer(self, activation):
        return nn.Sequential(
            nn.Linear(self.history_latent_dim + self.depth_latent_dim, 128),
            self._clone_activation(activation),
            nn.Linear(128, 64),
            self._clone_activation(activation),
            nn.Linear(64, self.mixer_latent_dim),
        )

    def _build_height_decoder(self, activation, rnn_hidden_size):
        return nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            self._clone_activation(activation),
            nn.Linear(64, self.height_dim),
        )

    def _encode_history(self, proprio_history):
        leading_shape = proprio_history.shape[:-1]
        history = proprio_history.reshape(*leading_shape, self.proprio_history_len, self.proprio_dim)
        recent_history = history[..., -self.history_length:, :].reshape(
            *leading_shape,
            self.history_length * self.proprio_dim,
        )
        return self.history_encoder(recent_history)

    def _encode_depth(self, depth_image):
        leading_shape = depth_image.shape[:-1]
        depth = depth_image.reshape(-1, 1, self.depth_height, self.depth_width)
        return self.depth_encoder(depth).reshape(*leading_shape, self.depth_latent_dim)

    def _build_actor_input(self, observations, masks=None, hidden_states=None):
        obs = self._split_observations(observations)
        history_latent = self._encode_history(obs["proprio_history"])
        estimator_output = self.estimator(history_latent)
        estimated_velocity_contact = torch.cat(
            [estimator_output[..., :3], torch.sigmoid(estimator_output[..., 3:])],
            dim=-1,
        )
        depth_latent = self._encode_depth(obs["depth_image"])
        mixer_latent = self.mixer(torch.cat([history_latent, depth_latent], dim=-1))
        z = self.memory_a(mixer_latent, masks, hidden_states)
        self.reconstructed_height_map = self.height_decoder(z)

        noisy_proprio = obs["curr_proprio_noisy"]
        goal = obs["goal"]
        if masks is not None:
            noisy_proprio = unpad_trajectories(noisy_proprio, masks)
            goal = unpad_trajectories(goal, masks)
            estimated_velocity_contact = unpad_trajectories(estimated_velocity_contact, masks)
        return torch.cat([noisy_proprio, goal, estimated_velocity_contact, z], dim=-1)

    def estimator_loss(self, observations, masks=None):
        obs = self._split_observations(observations)
        prediction = self.estimator(self._encode_history(obs["proprio_history"]))
        target = torch.cat([obs["base_lin_vel"], obs["foot_contacts"]], dim=-1)
        if masks is not None:
            prediction = unpad_trajectories(prediction, masks)
            target = unpad_trajectories(target, masks)
        velocity_loss = torch.mean((prediction[..., :3] - target[..., :3]) ** 2)
        contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[..., 3:],
            target[..., 3:],
        )
        return velocity_loss + contact_loss

    def height_reconstruction_loss(self, observations, masks=None):
        target = self._split_observations(observations)["height_scan"]
        if masks is not None:
            target = unpad_trajectories(target, masks)
        return torch.mean((self.reconstructed_height_map - target) ** 2)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        mean = self.actor(self._build_actor_input(observations, masks, hidden_states))
        self.distribution = torch.distributions.Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, masks=None, hidden_states=None):
        self.update_distribution(observations, masks, hidden_states)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor(self._build_actor_input(observations))

    def reset(self, dones=None):
        self.memory_a.reset(dones)

    def get_hidden_states(self):
        # RolloutStorage expects both actor and critic hidden states. The critic
        # is feed-forward, so this second copy is only an ignored placeholder.
        return self.memory_a.hidden_states, self.memory_a.hidden_states


class Memory(nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, inputs, masks=None, hidden_states=None):
        if masks is not None:
            if hidden_states is None:
                raise ValueError("Hidden states must be provided during recurrent policy updates.")
            outputs, _ = self.rnn(inputs, hidden_states)
            return unpad_trajectories(outputs, masks)

        outputs, self.hidden_states = self.rnn(inputs.unsqueeze(0), self.hidden_states)
        return outputs.squeeze(0)

    def reset(self, dones=None):
        if self.hidden_states is None or dones is None:
            return
        if isinstance(self.hidden_states, tuple):
            for hidden_state in self.hidden_states:
                hidden_state[..., dones.bool(), :] = 0.0
        else:
            self.hidden_states[..., dones.bool(), :] = 0.0


class TeacherPolicy(nn.Module):
    def __init__(self, num_actions, actor_hidden_dims=[256, 256, 256], activation='elu'):
        super().__init__()
        activation = get_activation(activation)
        self.std = nn.Parameter(torch.ones(num_actions))

        self.privileged_encoder = nn.Sequential(
            nn.Linear(27, 32),
            copy.deepcopy(activation),
            nn.Linear(32, 24),
            copy.deepcopy(activation),
            nn.Linear(24, 16),
        )
        self.height_encoder = nn.Sequential(
            nn.Linear(231, 128),
            copy.deepcopy(activation),
            nn.Linear(128, 64),
            copy.deepcopy(activation),
            nn.Linear(64, 32),
        )

        actor_layers = [nn.Linear(96, actor_hidden_dims[0]), copy.deepcopy(activation)]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(copy.deepcopy(activation))
        self.actor = nn.Sequential(*actor_layers)

    def load(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
        teacher_state_dict = {
            key: value for key, value in state_dict.items()
            if key == "std" or key.startswith(("privileged_encoder.", "height_encoder.", "actor."))
        }
        self.load_state_dict(teacher_state_dict)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def act_inference(self, observations, masks=None):
        goal = observations[..., 0:3]
        proprio = torch.cat([observations[..., 2187:2190], observations[..., 3:45]], dim=-1)
        privileged = torch.cat(
            [
                observations[..., 2190:2194],
                observations[..., 2204:2208],
                observations[..., 2208:2212],
                observations[..., 2443:2448],
                observations[..., 2194:2204],
            ],
            dim=-1,
        )
        privileged_latent = self.privileged_encoder(privileged)
        height_latent = self.height_encoder(observations[..., 2212:2443])
        actions = self.actor(torch.cat([proprio, goal, privileged_latent, height_latent], dim=-1))
        if masks is not None:
            actions = unpad_trajectories(actions, masks)
        return actions

    def distribution_parameters(self, observations, masks=None):
        mean = self.act_inference(observations, masks)
        return mean, self.std.expand_as(mean)
