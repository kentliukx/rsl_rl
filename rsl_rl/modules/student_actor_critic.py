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
from rsl_rl.utils import split_and_pad_trajectories, unpad_trajectories

class StudentActorCritic(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        history_length=10,
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
        # Keep the actor input layout identical to TeacherActorCritic so its
        # actor weights can initialize the student policy directly.
        self.estimator_dim = 15
        self.ladder_estimator_dim = 2 * 4 + 5
        actor_input_dim = 42 + 3 + self.estimator_dim + self.ladder_estimator_dim + rnn_hidden_size

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
        self.proprio_history_len = history_length
        self.depth_height = 36
        self.depth_width = 54

        self.history_encoder = self._build_history_encoder(activation_module)
        self.estimator = self._build_estimator(activation_module)
        self.depth_encoder = self._build_depth_encoder(activation_module)
        self.mixer = self._build_mixer(activation_module)
        self.ladder_estimator = self._build_ladder_estimator(activation_module, rnn_hidden_size)
        self.height_decoder = self._build_height_decoder(activation_module, rnn_hidden_size)
        self.reconstructed_ladder_obs = None
        self.reconstructed_height_obs = None
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
        print(f"GRU ladder estimator: {self.ladder_estimator}")
        print(f"Height decoder: {self.height_decoder}")

    def _build_history_encoder(self, activation):
        return nn.Sequential(
            nn.Linear(self.history_length * self.proprio_dim, 128),
            self._clone_activation(activation),
            nn.Linear(128, 64),
            self._clone_activation(activation),
            nn.Linear(64, self.history_latent_dim),
            self._clone_activation(activation),
        )

    def _build_estimator(self, activation):
        return nn.Sequential(
            nn.Linear(self.history_latent_dim, 64),
            self._clone_activation(activation),
            nn.Linear(64, 32),
            self._clone_activation(activation),
            nn.Linear(32, self.estimator_dim),
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
            nn.Linear(64 * 5 * 7, self.depth_latent_dim),
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

    def _build_ladder_estimator(self, activation, rnn_hidden_size):
        return nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            self._clone_activation(activation),
            nn.Linear(64, self.ladder_estimator_dim),
        )

    def _build_height_decoder(self, activation, rnn_hidden_size):
        return nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            self._clone_activation(activation),
            nn.Linear(64, 128),
            self._clone_activation(activation),
            nn.Linear(128, self.height_dim),
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

    def _build_actor_input(self, observations, masks=None, hidden_states=None, dones=None):
        observations_are_padded = masks is not None and dones is None
        obs = self._split_observations(observations)
        history_latent = self._encode_history(obs["proprio_history"])
        estimator_output = self.estimator(history_latent)
        estimated_state = torch.cat(
            [
                estimator_output[..., :3],
                torch.sigmoid(estimator_output[..., 3:7]),
                estimator_output[..., 7:15],
            ],
            dim=-1,
        )
        depth_latent = self._encode_depth(obs["depth_image"])
        mixer_latent = self.mixer(torch.cat([history_latent, depth_latent], dim=-1))
        if dones is not None:
            # Padding after feature extraction avoids duplicating the full
            # observation tensor during recurrent PPO updates.
            padded_mixer_latent, masks = split_and_pad_trajectories(mixer_latent, dones)
            z = self.memory_a(padded_mixer_latent, masks, hidden_states)
        else:
            z = self.memory_a(mixer_latent, masks, hidden_states)
        self.reconstructed_ladder_obs = self.ladder_estimator(z)
        self.reconstructed_height_obs = self.height_decoder(z)

        noisy_proprio = obs["curr_proprio_noisy"]
        goal = obs["goal"]
        if observations_are_padded:
            noisy_proprio = unpad_trajectories(noisy_proprio, masks)
            goal = unpad_trajectories(goal, masks)
            estimated_state = unpad_trajectories(estimated_state, masks)
        return torch.cat(
            [
                noisy_proprio,
                goal,
                estimated_state[..., :7],
                self.reconstructed_ladder_obs[..., :8],
                estimated_state[..., 7:9],
                estimated_state[..., 9:15],
                self.reconstructed_ladder_obs[..., 8:13],
                z,
            ],
            dim=-1,
        )

    @staticmethod
    def _squared_error_loss(prediction, target, sum_features):
        squared_error = torch.square(prediction - target)
        if sum_features:
            return torch.mean(torch.sum(squared_error, dim=-1))
        return torch.mean(squared_error)

    def _estimator_loss(self, observations, masks=None, sum_features=True):
        obs = self._split_observations(observations)
        prediction = self.estimator(self._encode_history(obs["proprio_history"]))
        target = torch.cat(
            [
                obs["base_lin_vel"],
                obs["foot_contacts"],
                obs["friction"],
                obs["added_mass"],
                obs["applied_force"],
                obs["applied_torque"],
            ],
            dim=-1,
        )
        if masks is not None:
            prediction = unpad_trajectories(prediction, masks)
            target = unpad_trajectories(target, masks)
        per_output_loss = torch.cat(
            (
                torch.square(prediction[..., :3] - target[..., :3]),
                torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction[..., 3:7],
                    target[..., 3:7],
                    reduction="none",
                ),
                torch.square(prediction[..., 7:] - target[..., 7:]),
            ),
            dim=-1,
        )
        if sum_features:
            # Backpropagate the sum over all 15 supervised estimator outputs.
            return torch.mean(torch.sum(per_output_loss, dim=-1))
        # Keep the diagnostic independent of the number of supervised outputs.
        return torch.mean(per_output_loss)

    def estimator_loss(self, observations, masks=None):
        return self._estimator_loss(observations, masks, sum_features=True)

    def estimator_loss_mean(self, observations, masks=None):
        return self._estimator_loss(observations, masks, sum_features=False)

    def _height_reconstruction_loss(self, observations, masks=None, sum_features=True):
        obs = self._split_observations(observations)
        height_target = obs["height_scan"]
        if masks is not None:
            height_target = unpad_trajectories(height_target, masks)
        return self._squared_error_loss(self.reconstructed_height_obs, height_target, sum_features)

    def height_reconstruction_loss(self, observations, masks=None):
        return self._height_reconstruction_loss(observations, masks, sum_features=True)

    def height_reconstruction_loss_mean(self, observations, masks=None):
        return self._height_reconstruction_loss(observations, masks, sum_features=False)

    def _ladder_reconstruction_loss(self, observations, masks=None, sum_features=True):
        """Supervise the explicit distance and ladder-observation GRU head."""
        obs = self._split_observations(observations)
        ladder_target = torch.cat(
            (
                obs["effector_ladder_plane_distance"],
                obs["effector_nearest_bar_distance"],
                obs["ladder_info"],
            ),
            dim=-1,
        )
        if masks is not None:
            ladder_target = unpad_trajectories(ladder_target, masks)
        return self._squared_error_loss(self.reconstructed_ladder_obs, ladder_target, sum_features)

    def ladder_reconstruction_loss(self, observations, masks=None):
        return self._ladder_reconstruction_loss(observations, masks, sum_features=True)

    def ladder_reconstruction_loss_mean(self, observations, masks=None):
        return self._ladder_reconstruction_loss(observations, masks, sum_features=False)

    def initialize_from_teacher(self, checkpoint):
        """Copy the compatible privileged Teacher policy network."""
        state_dict = torch.load(checkpoint, map_location="cpu")["model_state_dict"]
        prefixes = ("actor.",)
        copied_state_dict = {
            key: value for key, value in state_dict.items()
            if key.startswith(prefixes)
        }
        expected_keys = {
            key for key in self.state_dict()
            if key.startswith(prefixes)
        }
        if set(copied_state_dict) != expected_keys:
            missing = sorted(expected_keys - set(copied_state_dict))
            unexpected = sorted(set(copied_state_dict) - expected_keys)
            raise RuntimeError(
                "Teacher checkpoint is incompatible with the Student actor initialization. "
                f"Missing={missing}, unexpected={unexpected}"
            )
        for key in expected_keys:
            if copied_state_dict[key].shape != self.state_dict()[key].shape:
                raise RuntimeError(
                    "Teacher checkpoint tensor shape does not match Student initialization for "
                    f"{key}: teacher={tuple(copied_state_dict[key].shape)}, "
                    f"student={tuple(self.state_dict()[key].shape)}"
                )
        self.load_state_dict(copied_state_dict, strict=False)
        print(f"Initialized Student actor from Teacher checkpoint: {checkpoint}")

    def update_distribution(self, observations, masks=None, hidden_states=None, dones=None):
        mean = self.actor(self._build_actor_input(observations, masks, hidden_states, dones))
        self.distribution = torch.distributions.Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, masks=None, hidden_states=None, dones=None):
        self.update_distribution(observations, masks, hidden_states, dones)
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

        self.height_encoder = nn.Sequential(
            nn.Linear(231, 128),
            copy.deepcopy(activation),
            nn.Linear(128, 64),
            copy.deepcopy(activation),
            nn.Linear(64, 32),
        )

        actor_layers = [nn.Linear(105, actor_hidden_dims[0]), copy.deepcopy(activation)]
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
            if key == "std" or key.startswith(("height_encoder.", "actor."))
        }
        self.load_state_dict(teacher_state_dict)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def act_inference(self, observations, masks=None):
        goal = observations[..., 0:3]
        privileged = torch.cat(
            [
                observations[..., 510:514],
                observations[..., 522:526],  # effector-center to ladder plane
                observations[..., 526:530],  # effector-center to nearest rung
                observations[..., 514:516],  # friction and added mass
                observations[..., 516:522],  # applied force and torque
                observations[..., 761:766],
            ],
            dim=-1,
        )
        height_latent = self.height_encoder(observations[..., 530:761])
        actions = self.actor(torch.cat(
            [observations[..., 3:45], goal, observations[..., 507:510], privileged, height_latent],
            dim=-1,
        ))
        if masks is not None:
            actions = unpad_trajectories(actions, masks)
        return actions

    def distribution_parameters(self, observations, masks=None):
        mean = self.act_inference(observations, masks)
        return mean, self.std.expand_as(mean)
