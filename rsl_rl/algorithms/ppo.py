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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=5,
                 num_mini_batches=4,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.01,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="adaptive",
                 desired_kl=0.01,
                 mini_batch_divide=1,
                 estimator_loss_coef=1,
                 height_reconstruction_loss_coef=1,
                 ladder_reconstruction_loss_coef=1,
                 imitation_loss_coef=0.0,
                 imitation_loss_min_coef=0.0,
                 imitation_terrain_level_lower=1.0,
                 imitation_terrain_level_upper=3.0,
                 imitation_terrain_level_lpf_k=0.2,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.mini_batch_divide = int(mini_batch_divide)
        if self.mini_batch_divide < 1:
            raise ValueError("mini_batch_divide must be at least 1.")
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.estimator_loss_coef = estimator_loss_coef
        self.height_reconstruction_loss_coef = height_reconstruction_loss_coef
        self.ladder_reconstruction_loss_coef = ladder_reconstruction_loss_coef
        self.imitation_loss_max_coef = imitation_loss_coef
        self.imitation_loss_min_coef = imitation_loss_min_coef
        self.imitation_loss_coef = imitation_loss_coef
        self.policy_loss_coef = 1.0
        self.imitation_terrain_level_lower = imitation_terrain_level_lower
        self.imitation_terrain_level_upper = imitation_terrain_level_upper
        self.imitation_terrain_level_lpf_k = imitation_terrain_level_lpf_k
        self.imitation_terrain_level_ema = None
        self.teacher = None

    def set_teacher(self, teacher):
        self.teacher = teacher
        self.policy_loss_coef = 0.0

    def update_imitation_coefficient(self, mean_terrain_level):
        if self.teacher is None:
            self.imitation_loss_coef = 0.0
            self.policy_loss_coef = 1.0
            return
        if mean_terrain_level is None:
            return
        if self.imitation_terrain_level_ema is None:
            self.imitation_terrain_level_ema = mean_terrain_level
        else:
            k = self.imitation_terrain_level_lpf_k
            self.imitation_terrain_level_ema = (1.0 - k) * self.imitation_terrain_level_ema + k * mean_terrain_level
        terrain_span = max(self.imitation_terrain_level_upper - self.imitation_terrain_level_lower, 1e-6)
        terrain_progress = (self.imitation_terrain_level_ema - self.imitation_terrain_level_lower) / terrain_span
        terrain_progress = min(max(terrain_progress, 0.0), 1.0)
        self.imitation_loss_coef = self.imitation_loss_max_coef + terrain_progress * (
            self.imitation_loss_min_coef - self.imitation_loss_max_coef
        )
        self.policy_loss_coef = 1 - self.imitation_loss_coef

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    @staticmethod
    def _mean_abs_gradient(loss, tensors, coefficient):
        gradients = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
        gradients = [gradient for gradient in gradients if gradient is not None]
        if not gradients:
            return 0.0
        absolute_gradient_sum = sum(gradient.detach().abs().sum().item() for gradient in gradients)
        num_gradient_elements = sum(gradient.numel() for gradient in gradients)
        num_samples = gradients[0].numel() // gradients[0].shape[-1]
        return absolute_gradient_sum / num_gradient_elements * num_samples * abs(coefficient)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_height_reconstruction_loss = 0
        mean_ladder_reconstruction_loss = 0
        mean_imitation_loss = 0
        mean_rl_policy_gradient = 0
        mean_imitation_gradient = 0
        micro_num_mini_batches = self.num_mini_batches * self.mini_batch_divide
        if self.actor_critic.is_recurrent:
            if self.storage.num_envs % micro_num_mini_batches != 0:
                raise ValueError(
                    "num_envs must be divisible by num_mini_batches * mini_batch_divide for recurrent PPO."
                )
            generator = self.storage.reccurent_mini_batch_generator(
                micro_num_mini_batches, self.num_learning_epochs
            )
        else:
            total_samples = self.storage.num_envs * self.storage.num_transitions_per_env
            if total_samples % micro_num_mini_batches != 0:
                raise ValueError(
                    "rollout sample count must be divisible by num_mini_batches * mini_batch_divide."
                )
            generator = self.storage.mini_batch_generator(
                micro_num_mini_batches, self.num_learning_epochs
            )

        accumulated_kl = 0.0
        for micro_batch_index, (obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, recurrent_dones_batch) in enumerate(generator):

                if micro_batch_index % self.mini_batch_divide == 0:
                    self.optimizer.zero_grad()
                    accumulated_kl = 0.0

                self.actor_critic.act(
                    obs_batch,
                    hidden_states=hid_states_batch[0],
                    dones=recurrent_dones_batch if self.actor_critic.is_recurrent else None,
                )
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        accumulated_kl += kl_mean.item() / self.mini_batch_divide


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                if hasattr(self.actor_critic, "estimator_loss"):
                    estimator_loss = self.actor_critic.estimator_loss(obs_batch)
                    with torch.no_grad():
                        estimator_loss_mean = (
                            self.actor_critic.estimator_loss_mean(obs_batch)
                            if hasattr(self.actor_critic, "estimator_loss_mean") else estimator_loss.detach()
                        )
                else:
                    estimator_loss = torch.zeros((), device=self.device)
                    estimator_loss_mean = estimator_loss
                if hasattr(self.actor_critic, "height_reconstruction_loss"):
                    height_reconstruction_loss = self.actor_critic.height_reconstruction_loss(obs_batch)
                    with torch.no_grad():
                        height_reconstruction_loss_mean = (
                            self.actor_critic.height_reconstruction_loss_mean(obs_batch)
                            if hasattr(self.actor_critic, "height_reconstruction_loss_mean")
                            else height_reconstruction_loss.detach()
                        )
                else:
                    height_reconstruction_loss = torch.zeros((), device=self.device)
                    height_reconstruction_loss_mean = height_reconstruction_loss
                if hasattr(self.actor_critic, "ladder_reconstruction_loss"):
                    ladder_reconstruction_loss = self.actor_critic.ladder_reconstruction_loss(obs_batch)
                    with torch.no_grad():
                        ladder_reconstruction_loss_mean = (
                            self.actor_critic.ladder_reconstruction_loss_mean(obs_batch)
                            if hasattr(self.actor_critic, "ladder_reconstruction_loss_mean")
                            else ladder_reconstruction_loss.detach()
                        )
                else:
                    ladder_reconstruction_loss = torch.zeros((), device=self.device)
                    ladder_reconstruction_loss_mean = ladder_reconstruction_loss
                if self.teacher is not None:
                    with torch.inference_mode():
                        teacher_mu, teacher_sigma = self.teacher.distribution_parameters(obs_batch)
                    student_sigma = sigma_batch.clamp_min(1e-6)
                    teacher_sigma = teacher_sigma.clamp_min(1e-6)
                    imitation_loss = torch.mean(torch.sum(
                        torch.log(student_sigma / teacher_sigma)
                        + (teacher_sigma.square() + (teacher_mu - mu_batch).square()) / (2.0 * student_sigma.square())
                        - 0.5,
                        dim=-1,
                    ))
                else:
                    imitation_loss = torch.zeros((), device=self.device)
                loss = (
                    self.policy_loss_coef * surrogate_loss
                    + self.value_loss_coef * value_loss
                    + self.estimator_loss_coef * estimator_loss
                    + self.height_reconstruction_loss_coef * height_reconstruction_loss
                    + self.ladder_reconstruction_loss_coef * ladder_reconstruction_loss
                    + self.imitation_loss_coef * imitation_loss
                    - self.entropy_coef * entropy_batch.mean()
                )

                mean_rl_policy_gradient += self._mean_abs_gradient(
                    surrogate_loss,
                    (mu_batch, sigma_batch),
                    self.policy_loss_coef,
                ) / self.mini_batch_divide
                if self.teacher is not None and self.imitation_loss_coef != 0.0:
                    mean_imitation_gradient += self._mean_abs_gradient(
                        imitation_loss,
                        (mu_batch, sigma_batch),
                        self.imitation_loss_coef,
                    ) / self.mini_batch_divide

                # Match the gradient of the original logical minibatch mean loss.
                (loss / self.mini_batch_divide).backward()

                is_last_microbatch = (micro_batch_index + 1) % self.mini_batch_divide == 0
                if is_last_microbatch:
                    if self.desired_kl != None and self.schedule == 'adaptive':
                        if accumulated_kl > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < accumulated_kl < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                mean_value_loss += value_loss.item() / self.mini_batch_divide
                mean_surrogate_loss += surrogate_loss.item() / self.mini_batch_divide
                mean_estimator_loss += estimator_loss_mean.item() / self.mini_batch_divide
                mean_height_reconstruction_loss += height_reconstruction_loss_mean.item() / self.mini_batch_divide
                mean_ladder_reconstruction_loss += ladder_reconstruction_loss_mean.item() / self.mini_batch_divide
                mean_imitation_loss += imitation_loss.item() / self.mini_batch_divide

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_height_reconstruction_loss /= num_updates
        mean_ladder_reconstruction_loss /= num_updates
        mean_imitation_loss /= num_updates
        mean_rl_policy_gradient /= num_updates
        mean_imitation_gradient /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimator_loss,
            mean_height_reconstruction_loss,
            mean_ladder_reconstruction_loss,
            mean_imitation_loss,
            mean_rl_policy_gradient,
            mean_imitation_gradient,
        )
