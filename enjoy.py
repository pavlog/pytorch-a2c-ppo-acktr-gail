import os,sys,inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
robotEnv = os.path.realpath("../")
baseline = os.path.realpath("../baselines/")
roboschool = os.path.realpath("../roboschool/")
sys.path.insert(0, robotEnv)
sys.path.insert(0, baseline)
#sys.path.insert(0, roboschool)

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
import pickle

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

parser.add_argument(
        '--action-type',
        type=int,
        default=-1,
        help='action type to play (default: -1)')

args = parser.parse_args()

args.det = not args.non_det

policies = []

def make_env_multinetwork(envName):
    from multiEnv import MultiNetworkEnv
    env = makeEnv.make_env_with_best_settings_for_compound(envName)
    env = MultiNetworkEnv(env,policies)
    return env


# 0 is a walk
# 1 is a balance
# 2 analytical
# 3 analytical tasks with walk
# 4 walk tasks with analytica
# 5 walk tasks with analytica2

hidden_size = 64


trainType = 19
if args.action_type>=0:
    trainType = args.action_type
filesNamesSuffix = ""
makeEnvFunction = makeEnv.make_env_with_best_settings
if trainType==1:
    filesNamesSuffix = "balance_"
    makeEnvFunction = makeEnv.make_env_for_balance
if trainType==2:
    filesNamesSuffix = "analytical_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_analytical
if trainType==3:
    filesNamesSuffix = ""
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_analytical
if trainType==4:
    filesNamesSuffix = "analytical_"
    makeEnvFunction = makeEnv.make_env_with_best_settings
if trainType==5:
    filesNamesSuffix = "analytical2_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_analytical2
if trainType==6:
    filesNamesSuffix = "frontback_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_front_back
if trainType==7:
    filesNamesSuffix = "leftright_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_left_right
if trainType==8:
    filesNamesSuffix = "all_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_all
if trainType==9:
    filesNamesSuffix = "rotate_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_rotate
if trainType==10:
    from main import PPOPlayer
    filesNamesSuffix = "compound_"
    makeEnvFunction = make_env_multinetwork
    quadruppedEnv.settings.tasks_difficulty_from = 11
    quadruppedEnv.settings.tasks_difficulty_to = 11

if trainType==11:
    filesNamesSuffix = "test_"
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_test
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_test

if trainType==12:
    filesNamesSuffix = "zoo_"
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_test
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_test_zoo

if trainType==13:
    hidden_size = 128
    filesNamesSuffix = "zigote_"
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_test
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_zigote_front_back

if trainType==14:
    hidden_size = 64
    filesNamesSuffix = "zigote2_front_back_"
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_test
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_train
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_record
    #makeEnv.samplesEnvData = pickle.load( open( "./QuadruppedWalk-v1_MoveNoPhys.samples", "rb" ) )


if trainType==15:
    filesNamesSuffix = "all_bytasks_11_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_all
    quadruppedEnv.settings.tasks_difficulty_from = 11
    quadruppedEnv.settings.tasks_difficulty_to = 11

    #filesNamesSuffix = "all_bytasks_1_"
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_all
    #quadruppedEnv.settings.tasks_difficulty_from = 1
    #quadruppedEnv.settings.tasks_difficulty_to = 1

if trainType==16:
    filesNamesSuffix = "zigote_updown_"
    #filesNamesSuffix = "test_"
    #makeEnvFunction = makeEnv.make_env_with_best_settings_for_test
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_train_analytic
    quadruppedEnv.settings.tasks_difficulty_from = 11
    quadruppedEnv.settings.tasks_difficulty_to = 11

if trainType==17:
    from main import PPOPlayer
    filesNamesSuffix = "compound_tasks_"
    makeEnvFunction = make_env_multinetwork
    quadruppedEnv.settings.tasks_difficulty_from = 11
    quadruppedEnv.settings.tasks_difficulty_to = 11

if trainType==18:
    import pickle
    filesNamesSuffix = "zigote2_updown_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_record
    hidden_size = 128

if trainType==19:
    hidden_size = 64
    filesNamesSuffix = "all_bytasks_13_"
    makeEnvFunction = makeEnv.make_env_with_best_settings_for_all
    quadruppedEnv.settings.tasks_difficulty_from = 11
    quadruppedEnv.settings.tasks_difficulty_to = 11

args.env_name = "QuadruppedWalk-v1" #'RoboschoolAnt-v1' #"QuadruppedWalk-v1" #'RoboschoolAnt-v1' # "QuadruppedWalk-v1"
#args.load_dir = "./trained_models/"+args.env_name+"/ppo copy 4/"
args.load_dir = "./trained_models/"+args.env_name+"/ppo/"
#args.load_dir = "./trained_models/"+args.env_name+"/ppo copy 3/"
#args.use_proper_time_limits = True


device = torch.device("cpu")


if trainType==10:
    multiNetworkName = [
        "frontback_",
        "all_",
        "leftright_",
        "rotate_"
    ]
    policies = []
    for net in multiNetworkName:
        bestFilename = os.path.join(args.load_dir,"{}_{}{}_best.pt".format(args.env_name,net,hidden_size))
        ac,_ = torch.load(bestFilename)
        policies.append(PPOPlayer(ac,device))
        print("Policy multi loaded: ",bestFilename)

if trainType==17:
    multiNetworkName = [
        "all_bytasks_0_",
        "all_bytasks_1_",
        "all_bytasks_2_",
        "all_bytasks_3_",
        "all_bytasks_4_",
        "all_bytasks_5_",
        "all_bytasks_6_",
        "all_bytasks_7_",
        "all_bytasks_8_",
        "all_bytasks_9_",
        "all_bytasks_10_",
        "all_bytasks_11_",
        "all_bytasks_12_",
    ]
    policies = []
    for net in multiNetworkName:
        bestFilename = os.path.join(args.load_dir,"{}_{}{}_best.pt".format(args.env_name,net,hidden_size))
        ac,_ = torch.load(bestFilename)
        policies.append(PPOPlayer(ac,device))
        print("Policy multi loaded: ",bestFilename)

env = makeEnvFunction(args.env_name)

if trainType==18:
    #env.env.env.env.bodyFixedPos = True
    #env.env.env.env.gravity=0.000000
    TEMP = 0

loadFilename = os.path.join(args.load_dir, "{}_{}{}.pt".format(args.env_name,filesNamesSuffix,hidden_size))
loadFilename = os.path.join(args.load_dir, "{}_{}{}_best.pt".format(args.env_name,filesNamesSuffix,hidden_size))
#loadFilename = os.path.join(args.load_dir, "QuadruppedWalk-v1_64_best.pt")

#loadFilename = "./trained_models/QuadruppedWalk-v1_best/ppo/QuadruppedWalk-v1QuadruppedWalk-v1_256_best_distance.pt"
# We need to use the same statistics for normalization as used in training
#actor_critic,_ = torch.load(loadFilename)
#actor_critic,ob_rms = torch.load(loadFilename)
#actor_critic.eval()

#recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
#masks = torch.zeros(1, 1)

env.spec.max_episode_steps = min(args.num_steps,env.spec.max_episode_steps)

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


# Runs policy for X episodes and returns average reward
def evaluate_policy(env,policy, eval_episodes=10, render=False,device=None):
    print ("---------------------------------------")
    avg_reward = 0.
    policy.eval()
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        recurrent_hidden_states = torch.zeros(1,
                                      policy.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)
        while not done:
            inputs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            value, action, _, recurrent_hidden_states = policy.act(
            inputs, recurrent_hidden_states, masks)
            #obs = torch.FloatTensor((obs).reshape(1, -1)).to(device)

            #actions = samplesEnvData["action"]
            #action = actions[episode_steps]
            #obs, reward, done, info = env.step(action)
            if render:
                env.render()
            obs, reward, done, info = env.step(action[0].detach().numpy())
            episode_reward+=reward
            episode_steps+=1
            avg_reward += reward
            if done:
                if len(info):
                    print(info)
                else:
                    print("Reward:",episode_reward," Steps:",episode_steps)

    avg_reward /= eval_episodes

    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward



while True:

    actor_critic,_ = torch.load(loadFilename)
    actor_critic.eval()

    evaluate_policy(env,actor_critic,100,True,device)
