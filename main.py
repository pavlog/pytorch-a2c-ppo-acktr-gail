import os,sys,inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
robotEnv = os.path.realpath("../")
baseline = os.path.realpath("../baselines/")
roboschool = os.path.realpath("../roboschool/")
sys.path.insert(0, robotEnv)
sys.path.insert(0, baseline)
sys.path.insert(0, roboschool)
import roboschool
import quadruppedEnv

from quadruppedEnv import settings
quadruppedEnv.settings.robotNN = 1
quadruppedEnv.settings.jointVelocities = 0
quadruppedEnv.settings.history1Len = 0.0
quadruppedEnv.settings.history2Len = 0.0
quadruppedEnv.settings.history3Len = 0.0

from wrappers import MaxAndSkipEnv
from wrappers import FrameStack


import copy
import glob
import os
import time
from collections import deque
import datetime

import gym
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
from evaluation import evaluate
from tensorboardX import SummaryWriter

from gym.envs.registration import register


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

frames_stack = 2
max_and_skip = 8

def make_env(envName):
    env = gym.make(envName)
    env.env.advancedLevel = True
    env.env.addObstacles = False
    env.env.ActionIsAngles = True
    env.env.ActionIsAnglesType = 2
    env.env.ActionsIsAdditive = False
    env.env.inputsSpace = 0
    env.env.actionsSpace = 0
    env.env.simRewardOnly = False


    env.env.targetDesired_episode_from = 0
    env.env.targetDesired_episode_to = 10000
    env.env.targetDesired_angleFrom = np.pi/8.0
    env.env.targetDesired_angleTo = np.pi/4.0

    env.env.spawnYawMultiplier = 0.4
    env.env.targetDesiredYawMultiplier = 0.4
    
    env.env.analyticReward = True
    env.env.analyticRewardType = 1


    env = MaxAndSkipEnv(env,max_and_skip,False)
    env = FrameStack(env,frames_stack,True)
    return env

#import multiprocessing
#multiprocessing.set_start_method('spawn', True)

def main():

    gettrace = getattr(sys, 'gettrace', None)
    
    args = get_args()

    args.algo = 'ppo'
    args.env_name = 'QuadruppedWalk-v1' #'RoboschoolAnt-v1' #'QuadruppedWalk-v1' #'RoboschoolAnt-v1' #'QuadruppedWalk-v1'
    args.use_gae = True
    args.num_steps = 512
    #args.num_processes = 4
    args.num_processes = 4
    if gettrace():
        args.num_processes = 1
    args.lr = 0.0001
    args.entropy_coef = 0.0
    args.value_loss_coef  =0.5
    args.ppo_epoch  = 4
    args.num_mini_batch = 4
    args.gamma =0.99
    args.gae_lambda =0.95
    args.use_linear_lr_decay = True
    args.use_proper_time_limits = True
    args.save_dir = "./trained_models/"+args.env_name+"/"
    args.load_dir  = "./trained_models/"+args.env_name+"/"
    args.log_dir = "./logs/robot"
    if gettrace():
        args.save_dir = "./trained_models/"+args.env_name+"debug/"
        args.load_dir  = "./trained_models/"+args.env_name+"debug/"
        args.log_dir = "./logs/robot_d"
    args.num_env_steps = 10000000
    args.log_interval = 10
    args.eval_interval = 2
    args.hidden_size = 256 
    args.last_hidden_size = 256
    args.recurrent_policy = True

    reward_shaper = DefaultRewardsShaper(scale_value = 0.01)

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

    envs = make_vec_envs(
                        args.env_name,
                        args.seed, args.num_processes,
                        args.gamma, None, device, False,
                        normalizeOb=False, normalizeReturns=False,
                        max_episode_steps=args.num_steps,
                        makeEnvFunc=make_env,
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
    load_path = os.path.join(args.load_dir, args.algo)
    load_path = os.path.join(load_path, "{}_{}.pt".format(args.env_name,args.hidden_size))
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


    maxReward = -10000.0
    maxSteps = 0
    minDistance = 50000.0


    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
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
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

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

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
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
                if 'episode_reward' in info.keys():
                    episodeDone = True
                    i_episode+=1
                    episode_rewards.append(info['episode_reward'])
                    writer.add_scalar('reward/episode', info['episode_reward'],i_episode)
                    #print("E:",i_episode," T:",info['episode_steps'], " R:", info['episode_reward'], " D:",info['distToTarget'])
                if 'episode_steps' in info.keys():
                    episode_steps.append(info['episode_steps'])
                    writer.add_scalar('reward/steps', info['episode_steps'],i_episode)
                if 'alive' in info.keys():
                    episode_rewards_alive.append(info['alive'])
                    writer.add_scalar('reward/alive', info['alive'], i_episode)
                if 'progress' in info.keys():
                    episode_rewards_progress.append(info['progress'])
                    writer.add_scalar('reward/progress', info['progress'], i_episode)
                if 'servo' in info.keys():
                    episode_rewards_servo.append(info['servo'])
                    writer.add_scalar('reward/servo', info['servo'], i_episode)
                if 'distToTarget' in info.keys():
                    episode_dist_to_target.append(info['distToTarget'])
                    writer.add_scalar('reward/distToTarget', info['distToTarget'], i_episode)

            #if episodeDone and i_episode%10==0:
            #    print(i_episode,"({:.1f}/{}/{:.2f}) ".format(episode_rewards[-1],episode_steps[-1],episode_dist_to_target[-1]),end='',flush=True)

            if episodeDone:
                if len(episode_rewards) and episode_rewards[-1]>maxReward and j>args.log_interval:
                    maxReward = episode_rewards[-1]

                    bestFilename = os.path.join(save_path,"{}_{}_best.pt".format(args.env_name,args.hidden_size))
                    print("Writing best reward:","({:.1f}/{}/{:.2f}) ".format(episode_rewards[-1],episode_steps[-1],episode_dist_to_target[-1]),bestFilename)
                    torch.save([
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                    ], bestFilename)

                if len(episode_dist_to_target) and episode_dist_to_target[-1]<minDistance and j>args.log_interval:
                    minDistance = episode_dist_to_target[-1]

                    bestFilename = os.path.join(save_path,"{}_{}_best_distance.pt".format(args.env_name,args.hidden_size))
                    print("Writing best distance:","({:.1f}/{}/{:.2f}) ".format(episode_rewards[-1],episode_steps[-1],episode_dist_to_target[-1]),bestFilename)
                    torch.save([
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                    ], bestFilename)

                if len(episode_steps) and episode_steps[-1]>maxSteps and j>args.log_interval:
                    maxSteps = episode_steps[-1]

                    bestFilename = os.path.join(save_path, "{}_{}_best_steps.pt".format(args.env_name,args.hidden_size))
                    print("Writing best steps:","({:.1f}/{}/{:.2f}) ".format(episode_rewards[-1],episode_steps[-1],episode_dist_to_target[-1]),bestFilename)
                    torch.save([
                        actor_critic,
                        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                    ], bestFilename)

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
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":

            fileName = os.path.join(save_path, "{}_{}.pt".format(args.env_name,args.hidden_size))
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], fileName)
            print("Saved:",fileName)

            fileName = os.path.join(save_path, "{}_{}_actor.pt".format(args.env_name,args.hidden_size))
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

            print(" Reward/Steps {:.3f} Energy/Steps: {:.3f} Progress/Steps: {:.3f} entropy {:.1f} value_loss {:.5f} action_loss {:.5f}\n"
                .format(
                        np.mean(episode_rewards)/np.mean(episode_steps),
                        np.mean(episode_rewards_servo)/np.mean(episode_steps),
                        np.mean(episode_rewards_progress)/np.mean(episode_steps),
                        dist_entropy, 
                        value_loss,
                        action_loss))
        '''
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
        '''

if __name__ == "__main__":
    main()
