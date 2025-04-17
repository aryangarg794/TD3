import argparse
import gymnasium as gym
import numpy as np
import torch 

from td3.model.td3 import TD3

DIR = 'models/'

parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default='Ant-v4', help="Env to test on")
parser.add_argument("--numeps", type=int, default=2, help="How long to run the experiment")
parser.add_argument("--file", type=str, default='TD3_Ant-v4.pt', help="File to test")
parser.add_argument("--render", type=str, default='human', help="Render mode")

args = parser.parse_args()

if __name__ == "__main__":
    rewards = []
    env = gym.make(args.env, render_mode=args.render)
    obs_space = np.prod(env.observation_space.shape)
    action_space = np.prod(env.action_space.shape)
        
    agent = TD3(input_dim=obs_space, action_dim=action_space, max_action=env.action_space.high[0])
    agent.load(DIR + args.file)
    for ep in range(args.numeps):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done: 
            action = agent.actor(torch.as_tensor(obs, dtype=torch.float32)).cpu().detach().numpy()
            obs_prime, reward, terminated, truncated, _ = env.step(action)
            
            ep_reward += reward
            done = terminated or truncated
            obs = obs_prime
            
        rewards.append(ep_reward)
    
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    print(f'{avg_reward:.3f} ± {std_reward:.3f}')
    env.close()