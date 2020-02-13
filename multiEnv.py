import gym
import numpy as np
import glm
from gym import spaces
import torch

class MultiNetworkEnv(gym.Wrapper):
    def __init__(self, env,policies):
        gym.Wrapper.__init__(self, env)
        self.policies = policies
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(policies),), dtype=env.observation_space.dtype)
        

    def step(self, action):
        normActions = np.clip(action,-1.0,1.0)*0.5+0.5 
        sum = np.sum(normActions)
        if sum!=0.0:
            normActions = normActions/sum
        else:
            allZeros = 0
        result = torch.zeros(1,self.env.action_space.shape[0])
        for i,policy in enumerate(self.policies):
            pol_action = policy.getActions(self.prev_obs)
            pol_action = pol_action*normActions[i]
            result+=pol_action
        self.prev_obs, reward, done, info = self.env.step(result[0].detach().numpy())
        return self.prev_obs, reward, done, info

    def reset(self, **kwargs):
        self.prev_obs = self.env.reset(**kwargs)
        return self.prev_obs

