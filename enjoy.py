import os,sys,inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
robotEnv = os.path.realpath("../")
baseline = os.path.realpath("../baselines/")
roboschool = os.path.realpath("../roboschool/")
sys.path.insert(0, robotEnv)
sys.path.insert(0, baseline)
sys.path.insert(0, roboschool)

robotlib = os.path.realpath("../../Robot/pyRobotLib")
sys.path.insert(0, robotlib)
import pyRobotLib

from colorama import Fore, Back, Style 

import roboschool
import quadruppedEnv
import gym
import argparse
import os
# workaround to unpickle olf model files
import sys
from collections import deque
from a2c_ppo_acktr.model import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import torch
import glm

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


from quadruppedEnv import settings
quadruppedEnv.settings.robotNN = 1
quadruppedEnv.settings.jointVelocities = 0
quadruppedEnv.settings.history1Len = 0.0
quadruppedEnv.settings.history2Len = 0.0
quadruppedEnv.settings.history3Len = 0.0

sys.path.append('a2c_ppo_acktr')

from quadruppedEnv import makeEnv
#from makeEnv import make_env_with_best_settings

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=True,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--video-dir',
    default='./results/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
        '--num-steps',
        type=int,
        default=10000,
        help='number of forward steps in A2C (default: 5)')

args = parser.parse_args()

args.det = not args.non_det

args.env_name = "QuadruppedWalk-v1" #'RoboschoolAnt-v1' #"QuadruppedWalk-v1" #'RoboschoolAnt-v1' # "QuadruppedWalk-v1"
#args.load_dir = "./trained_models/"+args.env_name+"/ppo copy 4/"
args.load_dir = "./trained_models/"+args.env_name+"/ppo/"
#args.load_dir = "./trained_models/"+args.env_name+"/ppo copy 3/"
#args.use_proper_time_limits = True

env = makeEnv.make_env_with_best_settings(args.env_name)

hidden_size = 200

loadFilename = os.path.join(args.load_dir, "{}_{}.pt".format(args.env_name,hidden_size))
#loadFilename = "./trained_models/QuadruppedWalk-v1_best/ppo/QuadruppedWalk-v1QuadruppedWalk-v1_256_best_distance.pt"
# We need to use the same statistics for normalization as used in training
#actor_critic,_ = torch.load(loadFilename)
#actor_critic,ob_rms = torch.load(loadFilename)
#actor_critic.eval()

#recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
#masks = torch.zeros(1, 1)

obs = env.reset()

env.spec.max_episode_steps = args.num_steps

#env.cam.lookat[0] += 0.5         # x,y,z offset from the object (works if trackbodyid=-1)
#print(env0.viewer)
#env.viewer.cam.lookat[1] += 0.5
#env.viewer.cam.lookat[2] += 0.5
#env0.camera.move_and_look_at(0,-10,10,0,0,0)
#resetDebugVisualizerCamera(distance = 1)
#env0.camera.move_and_look_at

deque_maxLen = 1

episode_rewards = deque(maxlen=deque_maxLen)
episode_steps = deque(maxlen=deque_maxLen)
episode_rewards_alive = deque(maxlen=deque_maxLen)
episode_rewards_progress = deque(maxlen=deque_maxLen)
episode_rewards_servo = deque(maxlen=deque_maxLen)
episode_dist_to_target = deque(maxlen=deque_maxLen)

episode_rewards.append(0)
episode_steps.append(0)
episode_rewards_alive.append(0)
episode_rewards_progress.append(0)
episode_rewards_servo.append(0)
episode_dist_to_target.append(0)

episode_reward = 0
cur_episode_steps = 0

#loadFilename = "./trained_models/QuadruppedWalk-v1/ppo/QuadruppedWalk-v1_best_steps.pt"
#loadFilename = "./trained_models/QuadruppedWalk-v1/ppo/QuadruppedWalk-v1_best_distance.pt"
#actor_critic = Policy(
#    env.observation_space.shape,
#    env.action_space,
#    base_kwargs={'hidden_size' : 32, 'activation_layers_type' : "Tanh"})

#actor_critic,_ = torch.load(loadFilename)
#actor_critic,ob_rms = torch.load(loadFilename)
#actor_critic.eval()
'''
#print(actor_critic.base.actor._modules)

for k in actor_critic.base.actor._modules:
    print(actor_critic.base.actor._modules[k],end=' ')
    if hasattr(actor_critic.base.actor._modules[k],'bias'):
        biases = actor_critic.base.actor._modules[k].bias.data.cpu().numpy()
        print("bias", np.min(biases),np.max(biases), np.mean(biases), end=' ')
        weights = actor_critic.base.actor._modules[k].weight.data.cpu().numpy()
        print("weight", np.min(weights),np.max(weights), np.mean(weights), end=' ')
        #print("\nweight", weights)
    print("")
'''
device = torch.device("cpu")

while True:

    actor_critic,ob_rms = torch.load(loadFilename)
    #actor_critic,_ = torch.load(loadFilename)
    actor_critic.eval()
    #actor_critic.to(device)

    recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)


    obs = env.reset()

    while True:
        with torch.no_grad():
            inputs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            value, action, _, recurrent_hidden_states = actor_critic.act(
            inputs, recurrent_hidden_states, masks, deterministic=args.det)
            #action = actor_critic.getActionsOnly(inputs)

        #print(inputs)

        # Obser reward and next obs
        #print(action[0])
        obs, reward, done, infos = env.step(action[0].data.numpy())
        episode_reward+=reward
        cur_episode_steps+=1

        masks.fill_(0.0 if done else 1.0)

        if done:

            episode_rewards.append(episode_reward)
            episode_steps.append(cur_episode_steps)
            for info in infos:
                if 'alive' in info:
                    episode_rewards_alive.append(infos['alive'])
                if 'progress' in info:
                    episode_rewards_progress.append(infos['progress'])
                if 'servo' in info:
                    episode_rewards_servo.append(infos['servo'])
                if 'distToTarget' in info:
                    episode_dist_to_target.append(infos['distToTarget'])
            episode_reward = 0
            cur_episode_steps = 0

        env.render()

        if done:
            print(Fore.WHITE,infos)
            print(Style.RESET_ALL) 
            print(" reward mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_rewards),np.median(episode_rewards),
                        np.min(episode_rewards),np.max(episode_rewards)))

            print(" steps mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_steps),np.median(episode_steps),
                        np.min(episode_steps),np.max(episode_steps)))

            print(" alive mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_rewards_alive),np.median(episode_rewards_alive),
                        np.min(episode_rewards_alive),np.max(episode_rewards_alive)))

            print(" progress mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_rewards_progress),np.median(episode_rewards_progress),
                        np.min(episode_rewards_progress),np.max(episode_rewards_progress)))

            print(" servo mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_rewards_servo),np.median(episode_rewards_servo),
                        np.min(episode_rewards_servo),np.max(episode_rewards_servo)))

            print(" dist to target mean/median {:.3f}/{:.3f} min/max {:.3f}/{:.3f}".format(
                        np.mean(episode_dist_to_target),np.median(episode_dist_to_target),
                        np.min(episode_dist_to_target),np.max(episode_dist_to_target)))

            print(" Reward/Steps {:.3f} Energy/Steps: {:.3f}\n"
                .format(
                        np.mean(episode_rewards)/np.mean(episode_steps),
                        np.mean(episode_rewards_servo)/np.mean(episode_steps)))
            break
