import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from tqdm import tqdm
from copy import deepcopy
from typing import Self, List
from td3.model.td3 import TD3
from td3.utils.metrics import RollingAverage
from td3.utils.replay import ReplayBuffer

def validation_step(
    env: gym.Env,
    agent: TD3,  
    runs: int = 10, 
    device: str = 'cpu',
    seed: int = 0
) -> List[float]:
    rewards = []
    for _ in range(runs):
        obs, _ = env.reset(seed=seed + 100)
        done = False
        ep_reward = 0 
        
        while not done:
            with torch.no_grad():
                obs_torch = torch.as_tensor(obs, dtype=torch.float).view(1, -1).to(device)
                action = agent.actor(obs_torch).view(-1).cpu().numpy()
                
                obs_prime, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                
                obs = obs_prime
                done = terminated or truncated

        rewards.append(ep_reward)
        
    return rewards


def train(
    env: gym.Env, 
    agent: TD3,  
    game_name:str, 
    timesteps: int = 1000000, 
    val_freq: int = 5000, 
    batch_size: int = 256, 
    buffer_size: int = int(1e6),
    preload: int = 1000, 
    window: int = 5, 
    num_val_runs: int = 10, 
    device: str = 'cpu',
    seed: int = 0
) -> RollingAverage:
    
    obs_space = np.prod(env.observation_space.shape)
    action_space = np.prod(env.action_space.shape)
    replay = ReplayBuffer(obs_space, action_space, buffer_size, device)
    
    metrics = RollingAverage(window)
    
    env_test = deepcopy(env)
    
    obs, _ = env.reset(seed=seed)
    done = False
    for _ in tqdm(range(preload)):
        action = env.action_space.sample()
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        replay.update(obs, action, reward, obs_prime, int(done))
        
        obs = obs_prime
        if done: 
            obs, _ = env.reset(seed=seed)
            done = False
    
    obs, _ = env.reset(seed=seed)
    done = False
    
    print('\n')
    for step in range(1, timesteps+1):
        action = agent.explore_action(obs)
        obs_prime, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        replay.update(obs, action, reward, obs_prime, int(done))
        
        obs = obs_prime
        if done:
            obs, _ = env.reset(seed=seed)
            done = False
        
        # update step 
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = replay.sample(batch_size)
        agent.update(
            batch_obs, 
            batch_actions, 
            batch_rewards, 
            batch_next_obs, 
            batch_dones
        )
        
        if step % val_freq == 0 or step == 1:
            val_rewards = validation_step(env_test, agent, num_val_runs, device)
            metrics.update(val_rewards)
        
        avg_reward = float(np.mean(val_rewards))
        print(f'Timestep: {step} | Average Val Reward: {avg_reward:.4f}', end='\r')

    env.close()
    env_test.close()
    # agent.save(game_name)
    
    return metrics

def plot_results(scores, timesteps, val_freq, game_name, save: bool = False):
    vars_low = []
    vars_high = []
    q=10

    for i in range(scores.shape[1]):
        vars_low.append(np.percentile(scores[:, i], q=q))
        vars_high.append(np.percentile(scores[:, i], q=100-q))

    mean_scores = np.mean(scores, axis=0) 
    plt.style.use('ggplot')   
    
    color = 'r'
    xs = np.arange(0, timesteps+val_freq, val_freq)
    plt.plot(xs, mean_scores, label='Average Val Score', color=color)
    plt.plot(xs, vars_low, alpha=0.1, color=color)
    plt.plot(xs, vars_high, alpha=0.1, color=color)
    plt.fill_between(xs, vars_low, vars_high, alpha=0.2, color=color)
    plt.legend()
    plt.grid(True)
    plt.title(f'{game_name} TD3 Results')
    plt.ylabel('Cumm. Reward')
    plt.xlabel('Timestep')
    if save:
        plt.savefig(f'lcs/TD3_{game_name}')
    # plt.show()
    
    
    