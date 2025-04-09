import argparse
import torch 
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

from td3.model.td3 import TD3
from td3.utils.train import train, plot_results

parser = argparse.ArgumentParser(description="TD3 Experiment Runner")

parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="List of random seeds for running multiple experiments")
parser.add_argument("--env", type=str, default='HalfCheetah-v5', help="Env to test on")
parser.add_argument("--save", action="store_true", help="Save model or not")
parser.add_argument("--steps", type=int, default=int(1e6), help="How long to run the experiment")
parser.add_argument("--val", type=int, default=5000, help="When to evaluate")
parser.add_argument("--pre", type=int, default=int(25e3), help="How much to preload")
parser.add_argument("--rand_start", action="store_true", help="Randomized start or not (for the env)")

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    results = []
    
    print(f'Running env: {args.env} for {args.steps} steps')
    for seed in args.seeds:
        print(f'\nRunning Seed {seed}\n')
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
        env = gym.make(args.env)
        env.action_space.seed(seed)
        
        obs_space = np.prod(env.observation_space.shape)
        action_space = np.prod(env.action_space.shape)
        
        agent = TD3(
            obs_space, 
            action_space, 
            env.action_space.high[0],
            device=device
        )
    
        metrics = train(
            env, 
            agent, 
            game_name=args.env, 
            timesteps=args.steps,
            preload=args.pre, 
            val_freq=args.val, 
            device=device,
            seed=seed
        )
        
        results.append(metrics.averages)
    
    results = np.array(results)
    plot_results(results, args.steps, args.val, args.env, True)
    if args.save:
        agent.save(args.env)