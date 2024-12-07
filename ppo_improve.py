import torch
import torch.nn as nn
import torch.optim as optim

PPO_CLIP = 0.2

class ppo_agent():
    def __init__(self,
                 actor_critic,
                 # clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.MSELoss = nn.MSELoss()

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts, epoch):
        # 引入GAE
        advantages = self.compute_gae(rollouts.rewards, rollouts.value_preds, rollouts.masks)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # 标准化优势
        
        # 动态调整熵权重
        entropy_weight = 0.01 * (1 - epoch / self.ppo_epoch * 0.3)  # 随着训练轮数减少，熵权重逐步减少


        for e in range(self.ppo_epoch):

            data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)
            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch,
                    masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_targ

                value_loss = (return_batch - values).pow(2)

                self.optimizer.zero_grad()
                # 改为自适应的熵正则化
                loss = -torch.min(surr1, surr2) + 0.5 * value_loss - entropy_weight * dist_entropy
                loss.mean().backward()
       
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def compute_gae(self, rewards, values, masks, gamma=0.99, tau=0.95):
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
            advantages[t] = last_gae_lambda = delta + gamma * tau * masks[t] * last_gae_lambda
        return advantages
