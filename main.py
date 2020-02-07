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
from evaluation import evaluate
from tensorboardX import SummaryWriter

from gym.envs.registration import register
import glm

from quadruppedEnv import makeEnv


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

def main():


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
    args.ppo_epoch  = 10
    args.num_mini_batch = 64
    args.gamma =0.99
    args.gae_lambda =0.95
    args.clip_param = 0.2
    args.use_linear_lr_decay = True #True
    args.use_proper_time_limits = True
    args.save_dir = "./trained_models/"+args.env_name+"/"
    args.load_dir  = "./trained_models/"+args.env_name+"/"
    args.log_dir = "./logs/robot"
    if gettrace():
        args.save_dir = "./trained_models/"+args.env_name+"debug/"
        args.load_dir  = "./trained_models/"+args.env_name+"debug/"
        args.log_dir = "./logs/robot_d"
    args.num_env_steps = 2000000
    args.log_interval = 20
    args.eval_interval = 2
    args.hidden_size =  16 
    args.last_hidden_size = args.hidden_size
    args.recurrent_policy = False #True
    args.save_interval = 20
    args.seed = 2

    # 0 is a walk
    # 1 is a balance
    # 2 multitasks
    # 3 multitask experiments
    trainType = 3
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

    reward_shaper = DefaultRewardsShaper(scale_value = 0.001)

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
        base_kwargs={'recurrent': args.recurrent_policy,'hidden_size' : args.hidden_size,'last_hidden_size' : args.last_hidden_size, 'activation_layers_type' : "TanhM2"})

    '''
#    if args.load_dir not None:
    load_path = os.path.join(args.load_dir, args.algo)
    actor_critic, ob_rms = torch.load(os.path.join(load_path, args.env_name + ".pt"))
    '''
    load_path = os.path.join(args.load_dir, args.algo)
    load_path = os.path.join(load_path, "{}_{}{}_best.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
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

    skipWriteBest = True

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
                print("Mean:",Fore.WHITE,np.mean(episode_rewards),Style.RESET_ALL," Median:",Fore.WHITE,np.median(episode_rewards),Style.RESET_ALL," max reward:", maxReward)

                if len(episode_rewards) and np.mean(episode_rewards)>maxReward and j>args.log_interval:
                    if skipWriteBest==False:
                        maxReward = np.mean(episode_rewards)

                        bestFilename = os.path.join(save_path,"{}_{}{}_best.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
                        print("Writing best reward:",Fore.GREEN,"({:.1f}/{:.1f}/{}/{:.2f}) ".format(np.mean(episode_rewards),np.median(episode_rewards),np.mean(episode_steps),episode_dist_to_target[-1]),Style.RESET_ALL,bestFilename)
                        torch.save([
                            actor_critic,
                            getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                        ], bestFilename)
                    else:
                        skipWriteBest = False

                if len(episode_steps) and np.mean(episode_steps)>maxSteps and j>args.log_interval:
                    maxSteps = np.mean(episode_steps)

                    bestFilename = os.path.join(save_path, "{}_{}{}_best_steps.pt".format(args.env_name,filesNamesSuffix,args.hidden_size))
                    print("Writing best steps:","({:.1f}/{}/{:.2f}) ".format(np.mean(episode_rewards),np.mean(episode_steps),episode_dist_to_target[-1]),bestFilename)
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
