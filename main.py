import os,sys,inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
robotEnv = os.path.realpath("../")
#print(robotEnv)
baseline = os.path.realpath("../baselines/")
roboschool = os.path.realpath("../roboschool/")
sys.path.insert(0, robotEnv)
sys.path.insert(0, baseline)
#sys.path.insert(0, roboschool)
sys.path.insert(0, "../../Robot/RobotSimulator/Resources/")
import roboschool
import quadruppedEnv

from quadruppedEnv import settings
quadruppedEnv.settings.robotNN = 1
quadruppedEnv.settings.jointVelocities = 0
quadruppedEnv.settings.history1Len = 0.0
quadruppedEnv.settings.history2Len = 0.0
quadruppedEnv.settings.history3Len = 0.0


from colorama import Fore, Back, Style 

import argparse

import copy
import glob
import os
import time
from collections import deque
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from tensorboardX import SummaryWriter

from gym.envs.registration import register
import glm

from quadruppedEnv import makeEnv

class PPOPlayer:
        def __init__(self,actor,device):
            self.actor = actor
            self.actor.eval()
            self.device = device
            self.recurrent_hidden_states = torch.zeros(1,
                                        actor.recurrent_hidden_state_size)
            self.masks = torch.zeros(1, 1)

        def getActions(self,inputs):
            inputs = torch.FloatTensor(inputs.reshape(1, -1)).to(self.device)
            _, action, _, self.recurrent_hidden_states = self.actor.act(
            inputs, self.recurrent_hidden_states, self.masks)
            return action

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


import gym

class SamplesEnv(gym.Env):
    def __init__(self,data):
        self.numSteps = 4000
        self.data = data
        self.states = data["state"]
        self.numStates = len(self.states)
        self.actions = data["action"]
        self.next_states = data["next_state"]

        high = np.ones([len(self.states[0])])
        self.observation_space = gym.spaces.Box(-high, high)

        high = np.ones([len(self.actions[0])])
        self.action_space = gym.spaces.Box(-high, high)

        self.index = 0

    def seed(self,seedValue):
        return

    def reset(self):
        self.index+=1
        self.reward = 0.0
        self.steps = 0
        return self.states[(self.index-1)%self.numStates]
        #,env.action_space.shape[0],100,0.25

    def step(self,actions):
        newState = self.next_states[(self.index-1)%self.numStates]
        self.recordedActions = self.actions[(self.index-1)%self.numStates]
        reward = -np.sum((actions - self.recordedActions)**2) / self.action_space.shape[0]
        done = False
        if self.steps==self.numSteps:
            done=True
        self.index+=1
        self.steps+=1
        self.reward+=reward*10.0
        if done:
            return newState, reward*10.0, done, {"reward":self.reward, "steps":self.steps-1}
        else:
            return newState, reward*10.0, done, {}

global samplesEnvData
samplesEnvData={}
def makeSamplesEnv(evvName):
    return SamplesEnv(samplesEnvData)


class DefaultRewardsShaper:
    def __init__(self, clip_value = 0, scale_value = 1, shift_value = 0):
        self.clip_value = clip_value
        self.scale_value = scale_value
        self.shift_value = shift_value

    def __call__(self, reward):
        reward = reward + self.shift_value
        reward = reward * self.scale_value
        if self.clip_value > 0:
            reward = np.clip(reward, -self.clip_value, self.clip_value)
        return reward

def mutate(policy, power,powerLast):
    print("Mutation with:",power," last layer:",powerLast)
    with torch.no_grad():
        mutation_power = power

        countLast = 0
        count = 0
        for paramT in policy.base.actor.parameters():
            if(len(paramT.shape)==2): #weights of linear layer
                countLast = count
            count+=1

        count = 0
        for name,param in policy.base.actor.named_parameters():
            pow = power
            if count==countLast or count==countLast+1:
                pow = powerLast
            print(name,pow, end=' ')
            if(len(param.shape)==1): #bias of linear layer
                rands = np.random.randn(param.shape[0])
                for i0 in range(param.shape[0]):
                    param[i0]+= pow * rands[i0]
            if(len(param.shape)==2): #weights of linear layer
                rands = np.random.randn(param.shape[0],param.shape[1])
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1]+= pow * rands[i0][i1]
            count+=1

        policy.eval()
        print("")

def lock(policy, first,last):
    print("Lock with:",first," last layer:",last)
    with torch.no_grad():
        countLast = 0
        count = 0
        for paramT in policy.base.actor.parameters():
            if(len(paramT.shape)==2): #weights of linear layer
                countLast = count
            count+=1

        count = 0
        for param in policy.base.actor.parameters():
            needLock = first
            if count==countLast or count==countLast+1:
                needLock = last
            if needLock:
                param.requires_grad = False
            else:
                param.requires_grad = True
            count+=1

        policy.eval()

policies = []

def make_env_multinetwork(envName):
    from multiEnv import MultiNetworkEnv
    env = makeEnv.make_env_with_best_settings_for_compound(envName)
    env = MultiNetworkEnv(env,policies)
    return env

def printNetwork(net):
    print(net)
    for name, param in net.named_parameters():
        if hasattr(param,"requires_grad"):
            print(name, param.requires_grad,end=' ') #,param.data)
    print('')


def main():

    realEval = True #False

    gettrace = getattr(sys, 'gettrace', None)
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
            '--action-type',
            type=int,
            default=-1,
            help='action type to play (default: -1)')

    args = get_args(parser)

    args.algo = 'ppo'
    args.env_name = 'QuadruppedWalk-v1' #'RoboschoolAnt-v1' #'QuadruppedWalk-v1' #'RoboschoolAnt-v1' #'QuadruppedWalk-v1'
    args.use_gae = True
    args.num_steps = 2048
    #args.num_processes = 4
    args.num_processes = 4
    if gettrace():
        args.num_processes = 1
    args.lr = 0.0001
    args.entropy_coef = 0.0
    args.value_loss_coef  =0.5
    args.ppo_epoch  = 4
    args.num_mini_batch = 256
    args.gamma =0.99
    args.gae_lambda =0.95
    args.clip_param = 0.2
    args.use_linear_lr_decay = True #True #True #True
    args.use_proper_time_limits = True
    args.save_dir = "./trained_models/"+args.env_name+"/"
    args.load_dir  = "./trained_models/"+args.env_name+"/"
    args.log_dir = "./logs/robot"
    if gettrace():
        args.save_dir = "./trained_models/"+args.env_name+"debug/"
        args.load_dir  = "./trained_models/"+args.env_name+"debug/"
        args.log_dir = "./logs/robot_d"
    args.num_env_steps = 1000000
    args.log_interval = 30
    args.hidden_size =  64 
    args.last_hidden_size = args.hidden_size
    args.recurrent_policy = False #True
    args.save_interval = 20
    args.seed = 1
    reward_shaping = 0.01
    allowMutate = True


    # 0 is a walk
    # 1 is a balance
    # 2 multitasks
    # 3 multitask experiments
    trainType = 9
    filesNamesSuffix = ""
    if args.action_type>=0:
        trainType = args.action_type
 
    makeEnvFunction = makeEnv.make_env_with_best_settings
    if trainType==1:
        filesNamesSuffix = "balance_"
        makeEnvFunction = makeEnv.make_env_for_balance

    if trainType==2:
        filesNamesSuffix = "analytical_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_analytical

    if trainType==3:
        filesNamesSuffix = "analytical2_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_analytical2

    if trainType==4:
        filesNamesSuffix = "frontback_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_front_back

    if trainType==5:
        filesNamesSuffix = "leftright_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_left_right

    if trainType==6:
        filesNamesSuffix = "all_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_all

    if trainType==7:
        filesNamesSuffix = "rotate_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_rotate

    if trainType==8:
        filesNamesSuffix = "compound_"
        makeEnvFunction = make_env_multinetwork

    if trainType==9:
        import pickle
        realEval = False
        allowMutate = False
        args.use_linear_lr_decay = True #False
        args.num_env_steps = 5000000
        #reward_shaping = 0.1
        # from https://github.com/openai/baselines/issues/723
        '''
        args.num_steps = 4096
        args.lr = 0.0001
        args.num_mini_batch = 64
        args.gae_lambda =0.95
        args.gamma =0.99
        args.ppo_epoch  = 10
        args.clip_param = 0.3
        '''
        filesNamesSuffix = "test_"
        makeEnvFunction = makeEnv.make_env_with_best_settings_for_test
        '''
        args.num_processes = 1
        args.log_interval = 10
        print ("Samples preload")
        global samplesEnvData
        samplesEnvData = pickle.load( open( "../QuadruppedWalk-v1.samples", "rb" ) )
        makeEnvFunction = makeSamplesEnv
        '''

    reward_shaper = DefaultRewardsShaper(scale_value = reward_shaping)

    print("ActionType ",trainType," ",filesNamesSuffix)

    print("Num processes:", args.num_processes)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.log_dir = "/tmp/tensorboard/"
    #TesnorboardX
    writer = SummaryWriter(log_dir=args.log_dir+'runs/{}_PPO_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                            "ppo"))

    device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.set_num_threads(1)

    load_dir = os.path.join(args.load_dir, args.algo)

    multiNetworkName = [
        "frontback_",
        "all_",
        "leftright_",
        "rotate_"
    ]
    if trainType==8:
        for net in multiNetworkName:
            bestFilename = os.path.join(load_dir,"{}_{}{}_best.pt".format(args.env_name,net,args.hidden_size))
            ac,_ = torch.load(bestFilename)
            policies.append(PPOPlayer(ac,device))
            print("Policy multi loaded: ",bestFilename)



    envs = make_vec_envs(
                        args.env_name,
                        args.seed, args.num_processes,
                        args.gamma, None, device, False,
                        normalizeOb=False, normalizeReturns=False,
                        max_episode_steps=args.num_steps,
                        makeEnvFunc=makeEnvFunction,
                        num_frame_stack = 1,
                        info_keywords=('episode_steps','episode_reward','progress','servo','distToTarget',))
    #print(envs.observation_space.shape,envs.action_space)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy,'hidden_size' : args.hidden_size,'last_hidden_size' : args.last_hidden_size, 'activation_layers_type' : "Tanh"})

    '''
#    if args.load_dir not None:
    load_path = os.path.join(args.load_dir, args.algo)
    actor_critic, ob_rms = torch.load(os.path.join(load_path, args.env_name + ".pt"))
    '''
    load_path = os.path.join(load_dir, "{}_{}{}_best.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
    #load_path = os.path.join(load_path, "{}_{}{}.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
    preptrained_path = "../Train/trained_models/QuadruppedWalk-v1/Train_QuadruppedWalk-v1_256.pth"
    loadPretrained = False
    if loadPretrained and os.path.isfile(preptrained_path):
        print("Load preptrained")
        abj = torch.load(preptrained_path)
        print(abj)
        print(actor_critic.base)
        actor_critic.base.load_state_dict()
        actor_critic.base.eval()
    if os.path.isfile(load_path) and not loadPretrained:
        actor_critic, ob_rms = torch.load(load_path)
        actor_critic.eval()
        print("NN loaded: ",load_path)
    else:
        bestFilename = os.path.join(load_dir,"{}_{}{}_best_pretrain.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
        if os.path.isfile(bestFilename):
            actor_critic, ob_rms = torch.load(bestFilename)
            actor_critic.eval()
            print("NN loaded: ",bestFilename)


    maxReward = -10000.0
    maxSteps = 0
    minDistance = 50000.0


    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    deque_maxLen = 10

    episode_rewards = deque(maxlen=deque_maxLen)
    episode_steps = deque(maxlen=deque_maxLen)
    episode_rewards_alive = deque(maxlen=deque_maxLen)
    episode_rewards_progress = deque(maxlen=deque_maxLen)
    episode_rewards_servo = deque(maxlen=deque_maxLen)
    episode_dist_to_target = deque(maxlen=deque_maxLen)

    '''
    load_path = os.path.join(args.load_dir, args.algo)
    load_path = os.path.join(load_path, args.env_name + ".pt")
    actor_critic, ob_rms = torch.load(load_path)

    actor_critic.to(device)
    actor_critic.eval()
    #ob_rms.eval()
    '''
    '''
    args.use_gym_monitor = 1
    args.monitor_dir = "./results/"
    monitor_path = os.path.join(args.monitor_dir, args.algo)
    monitor_path = os.path.join(monitor_path, args.env_name)

    args.
    if args.use_gym_monitor:
        env = wrappers.Monitor(
            env, monitor_path, video_callable=False, force=True)
    '''
    i_episode=0

    save_path = os.path.join(args.save_dir, args.algo)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    trainOnSamplesAndExit = False
    if trainOnSamplesAndExit:
        import pickle
        print ("---------------------------------------")
        print ("Samples preload")
        data = pickle.load( open( "../QuadruppedWalk-v1.samples", "rb" ) )

        learning_rate = 0.0001
        max_episodes = 100
        max_timesteps = 4000
        betas = (0.9, 0.999)
        log_interval = 1

        envSamples = SamplesEnv(data)
        envSamples.numSteps = max_timesteps

        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.Adam(actor_critic.base.actor.parameters(),
                                                lr=learning_rate, betas=betas)
        #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # create a loss function
        criterion = nn.MSELoss(reduction="sum")

            # run the main training loop
        for epoch in range(max_episodes):
            state = envSamples.reset()
            time_step = 0
            testReward = 0
            testSteps = 0
            loss_sum = 0
            loss_max = 0

            for t in range(max_timesteps):
                time_step +=1

                nn_state = torch.FloatTensor((state).reshape(1, -1)).to(device)

                optimizer.zero_grad()
                net_out = actor_critic.base.forwardActor(nn_state)
                net_out = actor_critic.dist.fc_mean(net_out)

                state, reward, done, info = envSamples.step(net_out.detach().numpy())
                sim_action = envSamples.recordedActions

                sim_action_t = torch.FloatTensor([sim_action]).to(device)

                loss = criterion(net_out, sim_action_t)
                loss.backward()
                optimizer.step()
                loss_sum+=loss.mean()
                loss_max=max(loss_max,loss.max())

                testReward+=reward
                testSteps+=1

                if done:
                    if epoch % log_interval == 0:
                        #print(best_action_t*scaleActions-net_out*scaleActions)
                        print('Train Episode: {} t:{} Reward:{} Loss: mean:{:.6f} max: {:.6f}'.format(epoch, t,testReward,loss_sum/t,loss_max))
                        print(info)
                        reward = 0
                    break
        bestFilename = os.path.join(save_path,"{}_{}{}_best_pretrain.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
        torch.save([
            actor_critic,
            getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        ], bestFilename)
        exit(0)


    skipWriteBest = True

    printNetwork(actor_critic.base.actor)

    lock(actor_critic,first=False,last=False)
    #if trainType==9:
        #allowMutate = False
        #lock(actor_critic,first=True,last=False)
        #mutate(actor_critic,power=0.00,powerLast=0.3)

    printNetwork(actor_critic.base.actor)
    #from torchsummary import summary

    #summary(actor_critic.base.actor, (1, 48, 64))
    
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    episodeBucketIndex = 0

    maxReward = -10000000000
    numEval = 10
    if realEval:
        envEval = makeEnvFunction(args.env_name)
        if hasattr(envEval.env,"tasks") and len(envEval.env.tasks):
            numEval = max(numEval,len(envEval.env.tasks))
        maxReward = evaluate_policy(envEval,actor_critic,numEval*2,render=False,device=device)

    noMaxRewardCount = 0


    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        episode_r = 0.0
        stepsDone =0

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            #envs.venv.venv.venv.envs[0].render()

            index = 0
            for d in done:
                if d:
                    print(infos[index],flush=True)
                index+=1

            episodeDone = False

            '''
            index = 0
            for d in done:
                if d:
                    print("")
                    print(infos[index])
                index+=1
            '''

            for info in infos:
                if 'reward' in info.keys():
                    episodeDone = True
                    i_episode+=1
                    episode_rewards.append(info['reward'])
                    writer.add_scalar('reward/episode', info['reward'],i_episode)
                    #print("E:",i_episode," T:",info['episode_steps'], " R:", info['episode_reward'], " D:",info['distToTarget'])
                if 'steps' in info.keys():
                    episode_steps.append(info['steps'])
                    writer.add_scalar('reward/steps', info['steps'],i_episode)
                if 'alive' in info.keys():
                    episode_rewards_alive.append(info['alive'])
                    writer.add_scalar('reward/alive', info['alive'], i_episode)
                if 'prog' in info.keys():
                    episode_rewards_progress.append(info['prog'])
                    writer.add_scalar('reward/progress', info['prog'], i_episode)
                if 'servo' in info.keys():
                    episode_rewards_servo.append(info['servo'])
                    writer.add_scalar('reward/servo', info['servo'], i_episode)
                if 'd2T' in info.keys():
                    episode_dist_to_target.append(info['d2T'])
                    writer.add_scalar('reward/distToTarget', info['d2T'], i_episode)

                for val in info.keys():
                    if val not in ["reward","steps","alive","prog","servo","d2T",'epos','t']:
                        writer.add_scalar('reward/'+val, info[val], i_episode)

            #if episodeDone and i_episode%10==0:
            #    print(i_episode,"({:.1f}/{}/{:.2f}) ".format(episode_rewards[-1],episode_steps[-1],episode_dist_to_target[-1]),end='',flush=True)

            if episodeDone:
                episodeBucketIndex+=1
                print("Mean:",Fore.WHITE,np.mean(episode_rewards),Style.RESET_ALL," Median:",Fore.WHITE,np.median(episode_rewards),Style.RESET_ALL," max reward:", maxReward)

                #'''len(episode_rewards) and np.mean(episode_rewards)>maxReward and''' 
                if realEval:
                    if episodeBucketIndex % args.log_interval == 0 and  episodeBucketIndex>args.log_interval:
                        print("Step:",(j + 1) * args.num_processes * args.num_steps)
                        if skipWriteBest==False:
                            evalReward = evaluate_policy(envEval,actor_critic,numEval,device=device)
        
                            writer.add_scalar('reward/eval', evalReward, i_episode)
        
                            if evalReward>maxReward:
                                maxReward = evalReward
                                #maxReward = np.mean(episode_rewards)

                                bestFilename = os.path.join(save_path,"{}_{}{}_best.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
                                print("Writing best reward:",Fore.GREEN,"({:.1f}/{:.1f}/{}/{:.2f}) ".format(np.mean(episode_rewards),np.median(episode_rewards),np.mean(episode_steps),episode_dist_to_target[-1]),Style.RESET_ALL,bestFilename)
                                torch.save([
                                    actor_critic,
                                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                                ], bestFilename)
                                noMaxRewardCount = 0
                            else:
                                noMaxRewardCount+=1
                                if allowMutate:
                                    if noMaxRewardCount==5:
                                        print("Mutation low last layer")
                                        lock(actor_critic,first=False,last=False)
                                        mutate(actor_critic,power=0.00,powerLast=0.01)
                                    if noMaxRewardCount==8:
                                        print("Mutation low non last")
                                        lock(actor_critic,first=False,last=False)
                                        mutate(actor_critic,power=0.01,powerLast=0.0)
                                    if noMaxRewardCount==11:
                                        print("Mutation low all")
                                        lock(actor_critic,first=False,last=False)
                                        mutate(actor_critic,power=0.02,powerLast=0.2)
                                    if noMaxRewardCount==14:
                                        print("Mutation hi all")
                                        lock(actor_critic,first=False,last=False)
                                        mutate(actor_critic,power=0.03,powerLast=0.03)
                                        noMaxRewardCount = 0
                        else:
                            skipWriteBest = False
                else:
                    if len(episode_rewards) and np.mean(episode_rewards)>maxReward and j>args.log_interval:
                        if skipWriteBest==False:
                            maxReward = np.mean(episode_rewards)
                            writer.add_scalar('reward/maxReward', maxReward, i_episode)

                            bestFilename = os.path.join(save_path,"{}_{}{}_best.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
                            if len(episode_dist_to_target):
                                print("Writing best reward:",Fore.GREEN,"({:.1f}/{:.1f}/{}/{:.2f}) ".format(np.mean(episode_rewards),np.median(episode_rewards),np.mean(episode_steps),episode_dist_to_target[-1]),Style.RESET_ALL,bestFilename)
                            else:
                                print("Writing best reward:",Fore.GREEN,"({:.1f}/{:.1f}/{}) ".format(np.mean(episode_rewards),np.median(episode_rewards),np.mean(episode_steps)),Style.RESET_ALL,bestFilename)
                            
                            torch.save([
                                actor_critic,
                                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                            ], bestFilename)
                        else:
                            skipWriteBest = False
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            shaped_reward = reward_shaper(reward)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, shaped_reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":

            fileName = os.path.join(save_path, "{}_{}{}.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], fileName)
            print("Saved:",fileName, " cur avg rewards:",np.mean(episode_rewards))

            fileName = os.path.join(save_path, "{}_{}{}_actor.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
            torch.save(actor_critic.state_dict, fileName)
            print("Saved:",fileName)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("")
            print("Updates {}, num timesteps {}, FPS {}".format(j, total_num_steps,
                        int(total_num_steps / (end - start))))
            print(" Last {} training episodes:".format(
                        len(episode_rewards)))

            print(" reward mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_rewards),np.median(episode_rewards),
                        np.min(episode_rewards),np.max(episode_rewards)))

            print(" steps mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                        np.mean(episode_steps),np.median(episode_steps),
                        np.min(episode_steps),np.max(episode_steps)))

            if len(episode_rewards_alive):
                print(" alive mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                            np.mean(episode_rewards_alive),np.median(episode_rewards_alive),
                            np.min(episode_rewards_alive),np.max(episode_rewards_alive)))

            if len(episode_rewards_progress):
                print(" progress mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                            np.mean(episode_rewards_progress),np.median(episode_rewards_progress),
                            np.min(episode_rewards_progress),np.max(episode_rewards_progress)))

            if len(episode_rewards_servo):
                print(" servo mean/median {:.1f}/{:.1f} min/max {:.1f}/{:.1f}".format(
                            np.mean(episode_rewards_servo),np.median(episode_rewards_servo),
                            np.min(episode_rewards_servo),np.max(episode_rewards_servo)))

            if len(episode_dist_to_target):
                print(" dist to target mean/median {:.3f}/{:.3f} min/max {:.3f}/{:.3f}".format(
                                np.mean(episode_dist_to_target),np.median(episode_dist_to_target),
                                np.min(episode_dist_to_target),np.max(episode_dist_to_target)))

            print(" Reward/Steps {:.3f} Progress/Steps: {:.3f} entropy {:.1f} value_loss {:.5f} action_loss {:.5f}\n"
                .format(
                        np.mean(episode_rewards)/np.mean(episode_steps),
                        (0 if len(episode_rewards_progress)==0 else np.mean(episode_rewards_progress)/np.mean(episode_steps)),
                        dist_entropy, 
                        value_loss,
                        action_loss))

if __name__ == "__main__":
    main()
