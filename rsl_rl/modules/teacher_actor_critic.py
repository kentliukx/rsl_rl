"""Trainable privileged Teacher policy for the ladder task."""

import copy

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class TeacherActorCritic(nn.Module):
    """Privileged Teacher PPO network for the current observation layout."""

    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation="elu",
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__()
        if kwargs:
            print("TeacherActorCritic.__init__ ignored arguments: " + str(list(kwargs.keys())))

        activation_module = get_activation(activation)
        self.obs_slices = {
            "goal": slice(0, 3),
            "curr_proprio_clean": slice(3, 45),
            "base_lin_vel": slice(507, 510),
            "foot_contacts": slice(510, 514),
            "friction": slice(514, 515),
            "added_mass": slice(515, 516),
            "applied_force": slice(516, 519),
            "applied_torque": slice(519, 522),
            "effector_ladder_plane_distance": slice(522, 526),
            "effector_nearest_bar_distance": slice(526, 530),
            "height_scan": slice(530, 761),
            "ladder_info": slice(761, 766),
        }

        self.height_encoder = self._build_encoder(231, 128, 64, 32, activation_module)
        self.critic_height_encoder = self._build_encoder(231, 128, 64, 32, activation_module)
        self.actor = self._build_mlp(105, actor_hidden_dims, num_actions, activation_module)
        self.critic = self._build_mlp(105, critic_hidden_dims, 1, activation_module)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"Actor height encoder: {self.height_encoder}")
        print(f"Critic height encoder: {self.critic_height_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    @staticmethod
    def _build_encoder(input_dim, hidden_dim_1, hidden_dim_2, output_dim, activation):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            copy.deepcopy(activation),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            copy.deepcopy(activation),
            nn.Linear(hidden_dim_2, output_dim),
        )

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim, activation):
        layers = [nn.Linear(input_dim, hidden_dims[0]), copy.deepcopy(activation)]
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
                layers.append(copy.deepcopy(activation))
        return nn.Sequential(*layers)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def _split_observations(self, observations):
        return {name: observations[..., obs_slice] for name, obs_slice in self.obs_slices.items()}

    @staticmethod
    def _build_teacher_privileged(obs):
        return torch.cat(
            (
                obs["foot_contacts"],
                obs["effector_ladder_plane_distance"],
                obs["effector_nearest_bar_distance"],
                obs["friction"],
                obs["added_mass"],
                obs["applied_force"],
                obs["applied_torque"],
                obs["ladder_info"],
            ),
            dim=-1,
        )

    def _build_actor_input(self, observations):
        obs = self._split_observations(observations)
        return torch.cat(
            (obs["curr_proprio_clean"], obs["goal"], obs["base_lin_vel"], self._build_teacher_privileged(obs),
             self.height_encoder(obs["height_scan"])),
            dim=-1,
        )

    def _build_critic_input(self, observations):
        obs = self._split_observations(observations)
        return torch.cat(
            (obs["curr_proprio_clean"], obs["goal"], obs["base_lin_vel"], self._build_teacher_privileged(obs),
             self.critic_height_encoder(obs["height_scan"])),
            dim=-1,
        )

    def update_distribution(self, observations):
        mean = self.actor(self._build_actor_input(observations))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor(self._build_actor_input(observations))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(self._build_critic_input(critic_observations))
