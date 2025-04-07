import torch 
import torch.nn as nn
import numpy as np

from copy import deepcopy
from typing import Self, List
from torch import Tensor

class Critic(nn.Module):
    
    def __init__(
        self: Self, 
        input_dim: int, 
        action_dim: int, 
        hidden_dims: List,
        tau: float, 
        *args, 
        **kwargs
    ) -> None:
        super(Critic, self).__init__(*args, **kwargs)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        for i in range(1, len(hidden_dims)):
            self.layers.extend(
                [nn.Linear(hidden_dims[i-1], hidden_dims[i]), 
                nn.ReLU()]
            )
        
        self.layers.append(
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self.tau = tau
        
    def forward(
        self: Self,
        obs: Tensor, 
        action: Tensor
    ) -> Tensor:
        inp = torch.cat([obs, action], dim=-1)
        return self.layers(inp)
    
    def soft_update(
        self: Self, 
        target_params: nn.Parameter
    ) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.parameters(), target_params):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
                
                
class Actor(nn.Module):
    
    def __init__(
        self: Self, 
        input_dim: int, 
        max_action: int | float,  
        hidden_dims: List, 
        tau: float,  
        *args, 
        **kwargs
    ) -> None:
        
        super(Actor, self).__init__(*args, **kwargs)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0])
        )
        
        for i in range(1, len(hidden_dims)):
            self.layers.extend(
                [nn.Linear(hidden_dims[i-1], hidden_dims[i]), 
                nn.ReLU()]
            )
        
        self.layers.extend(
            [nn.Linear(hidden_dims[-1], 1), 
             nn.Tanh()]
        )
        
        self.tau = tau 
        self.max_action = max_action
    
    def forward(
        self: Self, 
        obs: Tensor
    ) -> None:
        out = self.layers(obs) * self.max_action
        return out.clip(-self.max_action, self.max_action)
        
    def soft_update(
        self: Self, 
        target_params: nn.Parameter
    ) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.parameters(), target_params):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
                
class TD3:
    
    def __init__(
        self: Self, 
        input_dim: int, 
        action_dim: int, 
        max_action: int, 
        gamma: float = 0.99, 
        lr_critic: float = 0.001, 
        lr_actor: float = 0.001, 
        hidden_dims_critic: List = list([400, 300]), 
        hidden_dims_actor: List = list([400, 300]), 
        update_steps: int = 2, 
        tau: float = 0.005, 
        wd: float = 0.01, 
        noise: float = 0.1, 
        tps_noise: float = 0.2, 
        noise_clip: float = 0.5, 
        device: str = 'cpu'
    ) -> None:
        
        self.critic_1 = Critic(input_dim, action_dim, hidden_dims_critic, tau).to(device)
        self.critic_2 = Critic(input_dim, action_dim, hidden_dims_critic, tau).to(device)
        self.actor = Actor(input_dim, max_action, hidden_dims_actor, tau).to(device)
        
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.actor_target = deepcopy(self.actor)
        
        self.networks = [self.critic_1, self.critic_2, self.actor]
        self.targets = [self.critic_1_target, self.critic_2_target, self.actor_target]
        
        self.opt_critic_1 = torch.optim.AdamW(self.critic_1.parameters(), lr=lr_critic, weight_decay=wd)
        self.opt_critic_2 = torch.optim.AdamW(self.critic_2.parameters(), lr=lr_critic, weight_decay=wd) 
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor) 
        
        self.max_action = max_action
        self.noise = noise
        self.gamma = gamma
        self.tps_noise = tps_noise
        self.device = device
        self.policy_update = update_steps
        self.noise_clip = noise_clip
        self.normal = torch.distributions.Normal(loc=0.0, scale=self.tps_noise)
        
        self.loss_func = nn.MSELoss()
        
        self.timesteps = 0
        
    def explore_action(
        self: Self, 
        obs: np.ndarray,
    ) -> np.ndarray:
        
        with torch.no_grad():
            obs_torch = torch.as_tensor(obs).view(1, -1).to(device=self.device)
            action = self.actor(obs_torch).view(-1).cpu().numpy() 
            action += np.random.normal(scale=self.noise, size=action.shape)
            
        return np.clip(action, -self.max_action, self.max_action)
    
    def tps(
        self: Self, 
        batch_obs: Tensor
    ) -> Tensor:
        
        with torch.no_grad():
            actions = self.actor_target(batch_obs)
            noise = self.normal.sample(sample_shape=actions.shape)
            noise = noise.clip(-self.noise_clip, self.noise_clip)
            actions += noise
            
        return actions.clip(-self.max_action, self.max_action)
        
    def update(
        self: Self,
        batch_states: Tensor, 
        batch_actions: Tensor, 
        batch_rewards: Tensor, 
        batch_next_states: Tensor, 
        batch_dones: Tensor 
    ) -> None:
        
        actions_tilde = self.tps(batch_next_states)
        with torch.no_grad():
            q1_targets = self.critic_1_target(batch_next_states, actions_tilde)
            q2_targets = self.critic_2_target(batch_next_states, actions_tilde)
            
            cat_targets = torch.cat([q1_targets, q2_targets], dim=-1)
            min_values = torch.min(cat_targets, dim=1, keepdim=True)[0] 
            td_targets = torch.where(batch_dones, batch_rewards, batch_rewards + self.gamma * min_values)
        
        q1_values = self.critic_1(batch_states, batch_actions)
        loss_q1 = self.loss_func(q1_values, td_targets)
        self.opt_critic_1.zero_grad()
        loss_q1.backward()
        self.opt_critic_1.step()
        
        q2_values = self.critic_2(batch_states, batch_actions)
        loss_q2 = self.loss_func(q2_values, td_targets)
        self.opt_critic_2.zero_grad()
        loss_q2.backward()
        self.opt_critic_2.step()
        
        if self.timesteps % self.policy_update == 0:
            loss_actor = self.critic_1(batch_states, self.actor(batch_states)).mean()
            self.opt_actor.zero_grad()
            loss_actor.backward()
            self.opt_actor.step()
            
            self.update_targets()
            
        self.timesteps += 1
        
    def update_targets(
        self: Self, 
    ) -> None:
        for net, target in zip(self.networks, self.targets):
            net.soft_update(target.parameters())
    
    def __repr__(self):
        return f'TD3 Agent'