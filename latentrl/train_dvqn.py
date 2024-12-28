import argparse
import datetime
import math
import os
import random
import sys
import time
from collections import deque
from itertools import count
from pprint import pprint
import colored_traceback.auto
import numpy as np
import torch
from statistics import mean

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import wandb

from common.Logger import LoggerWandb
from policies import HDQN_TCURL_VQ, DVQN
from common.make_env import make_env_atari

from torch import nn

# from policies.hrl_dqn_agent import DuoLayerAgent, SingelLayerAgent
# from policies.vanilla_dqn_agent import VanillaDQNAgent


MAKE_ENV_FUNCS = {
    "Atari": make_env_atari,
}


def parse_args(args_str=None):
    cli = argparse.ArgumentParser()
    subparsers = cli.add_subparsers(dest="algo")
    subparsers.required = True
    parser_dvqn = subparsers.add_parser("dvqn")
    parser_avgdqn = subparsers.add_parser("avgdqn")
    parser_dqn = subparsers.add_parser("dqn")
    parser_ddqn = subparsers.add_parser("ddqn")
    parser_cddqn = subparsers.add_parser("cddqn")
    parser_sarsa = subparsers.add_parser("sarsa")
    parser_dueldqn = subparsers.add_parser("dueldqn")

    cli.add_argument("--domain_name", default="Breakout-v5", type=str)
    cli.add_argument("--domain_type", default="Atari", type=str)
    cli.add_argument("--total_timesteps", default=1e5, type=int)


    # cli.add_argument("--args_from_cli", default=False, action="store_true")
    # cli.add_argument(
    #     "--mode", default="dqn", choices=["dqn", "ddqn", "cddqn", "avgdqn", "sarsa", "dvqn", "dueldqn"], type=str
    # )

    cli.add_argument("--clip_reward", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--clip_grad_mode", default="norm", choices=["norm", "clamp"], type=str)
    cli.add_argument("--per", default=False, action=argparse.BooleanOptionalAction, help="prioritized experience replay")
    cli.add_argument("--input_format", default="full_img", type=str)
    cli.add_argument("--env_seed", default=940805, type=int)

    cli.add_argument("--init_steps", default=1e4, type=int)
    cli.add_argument("--batch_size", default=256, type=int)
    cli.add_argument("--size_replay_memory", default=1e5, type=int)
    cli.add_argument("--gamma", default=0.99, type=float)
    cli.add_argument("--exploration", nargs=3, default=[0.99, 0.1, 0.1], type=float)
    cli.add_argument("--epsilon_decay", default=0.99998, type=float)
    # cli.add_argument("--epsilon_min", default=0.01, type=float)
    parser_avgdqn.add_argument("--avgdqn_k", default=0, type=int)

    cli.add_argument("--Q_encoder_linear_dims", nargs="*", default=[-1], type=int)
    cli.add_argument("--Q_critic_dims", nargs="*", default=[256, 256], type=int)
    cli.add_argument("--Q_enc_detach", default=False, action=argparse.BooleanOptionalAction)
    parser_dvqn.add_argument("--share_encoder", default=False, action=argparse.BooleanOptionalAction)
    parser_dvqn.add_argument("--V_encoder_linear_dims", default=[-1], nargs="*", type=int)
    parser_dvqn.add_argument("--V_critic_dims", default=[256, 256], nargs="*", type=int)
    parser_dvqn.add_argument("--V_enc_detach", default=False, action=argparse.BooleanOptionalAction)
    parser_dvqn.add_argument("--use2Q", default=False, action=argparse.BooleanOptionalAction)

    cli.add_argument("--lr_Q", default="0.0001", type=str, help="sometimes start with lin")
    parser_dvqn.add_argument("--lr_V", default="0.0001", type=str, help="sometimes start with lin")
    cli.add_argument("--lr_curl", default="0.0001", type=str)
    cli.add_argument("--lr_vq", default="0.0001", type=str)
    cli.add_argument("--lr_decay", default=0.9999, type=float)
    cli.add_argument("--lr_min", default=0.01, type=float)

    cli.add_argument("--Q_gradient_steps", default=1, type=int)
    parser_dvqn.add_argument("--V_gradient_steps", default=1, type=int)

    cli.add_argument("--freq_Q_learn", default=1, type=int)
    cli.add_argument("--freq_Q_sync", default=1000, type=int)
    parser_dvqn.add_argument("--freq_V_learn", default=1, type=int)
    parser_dvqn.add_argument("--freq_V_sync", default=1000, type=int)
    cli.add_argument("--freq_curl_learn", default=1, type=int)
    cli.add_argument("--freq_curl_sync", default=1, type=int)

    cli.add_argument("--tau_Q_encoder", default=1, type=float)
    cli.add_argument("--tau_Q_critic", default=1, type=float)
    parser_dvqn.add_argument("--tau_V_encoder", default=1, type=float)
    parser_dvqn.add_argument("--tau_V_critic", default=1, type=float)
    cli.add_argument("--tau_curl", default=0.001, type=float)
    cli.add_argument("--tau_vq", default=0.001, type=float)

    cli.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "adam"],type=str)
    cli.add_argument("--criterion", default="l1", choices=["l1", "l2"], type=str)
    cli.add_argument("--freq_eval", default=1e4, type=int)
    cli.add_argument("--evaluation_episodes", default=10, type=int)
    cli.add_argument("--wandb_tags", default=None, nargs="*", type=str)
    cli.add_argument("--wandb_mode", default="online", type=str)
    cli.add_argument("--extra_note", default="", type=str)
    cli.add_argument("--repetitions", default=20, type=int)

    cli.add_argument("--use_curl", default=None, choices=["on_V", "on_Q", "off"], type=str)
    cli.add_argument("--curl_pair", default="raw", choices=["raw", "temp", "atc"], type=str)
    cli.add_argument("--use_vq", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument(
        "--curl_vq_cfg",
        nargs=3,
        default=[0.0, 0.0, 0.0],
        type=float,
        help="factors for cb_diversity, vq_entropy, vq_loss",
    )
    cli.add_argument("--curl_projection_dims", default=[-1], nargs="*", type=int)
    cli.add_argument("--curl_enc_detach", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--critic_upon_vq", default=False, action=argparse.BooleanOptionalAction)
    cli.add_argument("--num_vq_embeddings", default=16, type=int)
    # cli.add_argument("--dim_vq_embeddings", default=128, type=int)
    cli.add_argument("--vq_softmin_beta", default=0.5, type=float)

    if args_str is not None:
        args = cli.parse_args(args_str.split())
    else:
        args = cli.parse_args()


    if args.use_curl == "off":
        args.use_curl = None
    pprint(args)
    return args


def train(args):
    best_eval_reward = -np.inf
    best_train_reward = -np.inf
    for rep in range(args.repetitions):
        print(f"====Starting Repetition {rep}====")
        current_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        # 🐝 initialise a wandb run
        project_name = args.domain_name

        group_name = f"{args.algo}_step{int(args.total_timesteps/1000)}k_bs{args.batch_size}|{args.extra_note}"

        run = wandb.init(
            # project="HDQN_AbsTable_GrdNN_Atari",
            project=f"HDQN_Atari_{project_name}",
            # project="HDQN_MinAtar",
            # project="HDQN_Neo_Carracing",
            mode=args.wandb_mode,
            group=group_name,
            tags=args.wandb_tags,
            # notes=cfg["wandb_notes"],
            config=vars(args),
        )

        cwd_path=os.path.abspath(__file__)
        result_path = cwd_path.rsplit('/', 2)[0] + '/results'
        log_dir_root = os.path.join(
            result_path,
            args.domain_type,
            args.domain_name,
            group_name,
        )
        os.makedirs(os.path.join(log_dir_root, "best_models"), exist_ok=True)
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%b%d_%H-%M-%S")
        log_dir = os.path.join(log_dir_root, current_time)
        os.makedirs(log_dir, exist_ok=True)
        L = LoggerWandb()
        env = MAKE_ENV_FUNCS[args.domain_type]("ALE/" + args.domain_name, seed=args.env_seed)

        # agent = HDQN_Pixel(config, env)
        agent = DVQN(args, env, logger=L)
        if agent.use_vq:
            wandb.watch(agent.vq, log="all", log_freq=100, idx=0)
        # wandb.watch(agent.abs_V, log="all", log_freq=100, idx=1)
        # wandb.watch(agent.ground_Q, log="all", log_freq=100, idx=2)
        if agent.use_curl:
            wandb.watch(agent.curl, log="all", log_freq=100, idx=3)

        time_start_training = time.time()
        # gym.reset(seed=int(time.time()))
        total_steps = int(args.total_timesteps + args.init_steps)
        # agent.cache_goal_transition()
        episodic_reward_window = deque(maxlen=15)
        eval_rwd_window = deque(maxlen=6)
        ema_reward_list = []
        time_steps_list = []
        recent_dropping_episodes = 0
        while agent.timesteps_done < total_steps:
            time_start_episode = time.time()
            # Initialize the environment and state
            state, info = env.reset()
            episodic_reward = 0
            episodic_negative_reward = 0
            episodic_non_negative_reward = 0
            for t in count():
                # [Select and perform an action]
                action, action_prob = agent.act(state)
                # [Step]
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.timesteps_done += 1

                info['action_prob'] = action_prob
                agent.cache(state, action, next_state, reward, terminated, info)

                episodic_reward += reward
                if reward < 0:
                    episodic_negative_reward += reward
                else:
                    episodic_non_negative_reward += reward
                # [update]
                # action_prime = agent.act_table(info)
                if agent.timesteps_done >= args.init_steps:
                    agent.update()

                    if agent.timesteps_done % args.freq_eval == 0:
                        print("start eval ...")
                        avg_eval_rwd = test(agent, args, L)
                        eval_rwd_window.append(avg_eval_rwd)
                        print("eval end")

                if agent.timesteps_done >= total_steps:
                    truncated = True

                state = next_state

                if terminated or truncated:
                    agent.episodes_done += 1
                    episodic_reward_window.append(episodic_reward)

                    metrics = {
                        "Episodic/reward": episodic_reward,
                        "Episodic/negative_reward": episodic_negative_reward,
                        "Episodic/non_negative_reward": episodic_non_negative_reward,
                        "Episodic/timesteps_done": agent.timesteps_done,
                        "Episodic/length": t + 1,
                        "Episodic/total_time_elapsed": (time.time() - time_start_training) / 3600,
                        "Episodic/fps_per_episode": int((t + 1) / (time.time() - time_start_episode)),
                    }

                    L.log_and_dump(metrics, agent)
                    # L.dump2wandb(agent=agent, force=True)

                    print2console(
                        agent=agent,
                        episodic_reward=episodic_reward,
                        terminated=terminated,
                        truncated=truncated,
                        t=t,
                        time_start_episode=time_start_episode,
                        rep=rep,
                    )

                    break

        wandb.finish()


    print("Complete")
    env.close()


def test(agent, args, L: LoggerWandb):
    env = MAKE_ENV_FUNCS[args.domain_type]("ALE/" + args.domain_name)
    # env.eval()

    episodic_rews = []
    episodic_non_negative_rews = []
    episodic_negative_rews = []

    # Test performance over several episodes
    terminated = True
    for _ in range(args.evaluation_episodes):
        while True:
            if terminated:
                state, info = env.reset()
                reward_sum = 0
                non_negative_reward_sum = 0
                negative_reward_sum = 0
                terminated = False

            action = agent.act_e_greedy(state, epsilon=0.01)  # Choose an action ε-greedily
            state, reward, terminated, truncated, info = env.step(action)  # Step
            reward_sum += reward
            if reward >= 0:
                non_negative_reward_sum += reward
            else:
                negative_reward_sum += reward

            if terminated or truncated:
                episodic_rews.append(reward_sum)
                episodic_non_negative_rews.append(non_negative_reward_sum)
                episodic_negative_rews.append(negative_reward_sum)
                break
    env.close()
    if len(episodic_non_negative_rews) == 0:
        episodic_non_negative_rews.append(0)
    if len(episodic_negative_rews) == 0:
        episodic_negative_rews.append(0)
    avg_reward = sum(episodic_rews) / len(episodic_rews)
    avg_non_negative_reward = sum(episodic_non_negative_rews) / len(episodic_non_negative_rews)
    avg_negative_reward = sum(episodic_negative_rews) / len(episodic_negative_rews)
    # Return average reward and Q-value

    metrics = {
        "Evaluation/avg_episodic_reward": avg_reward,
        "Evaluation/avg_episodic_non_negative_reward": avg_non_negative_reward,
        "Evaluation/avg_episodic_negative_reward": avg_negative_reward,
        "Evaluation/timesteps_done": agent.timesteps_done,
        "Evaluation/episodes_done": agent.episodes_done,
    }
    L.log_and_dump(metrics, agent)

    return avg_reward


def print2console(agent, episodic_reward, terminated, truncated, t, time_start_episode, rep):
    print(f"===========Episode {agent.episodes_done} Done| Repetition {rep}=====")
    print("[Total_steps_done]:", agent.timesteps_done)
    print(f"[Episode {agent.episodes_done} Reward]: {episodic_reward}")
    print("[Exploration_rate]:", agent.exploration_rate)
    print("[Episodic_fps]:", int((t + 1) / (time.time() - time_start_episode)))
    print("[Episodic time cost]: {:.1f} s".format(time.time() - time_start_episode))
    print("[Episodic timesteps]: {} ".format(t + 1))
    print(f"[Terminal:{terminated} | Truncated: {truncated}]")
    print("[Current_progress_remaining]:", agent._current_progress_remaining)
    print(f"[wandb run name]: {wandb.run.project}/{wandb.run.group}/{wandb.run.name}")


if __name__ == "__main__":
    # check if cuda is available
    print("CUDA available: ", torch.cuda.is_available())
    # check number of gpus
    print("Number of GPUs: ", torch.cuda.device_count())
    os.environ["WANDB__SERVICE_WAIT"] = "1200"

    # [parse the args from the args file]
    # with open(
    #     f"/user/yuan.xue/u13186/DVQN_RL/latentrl/args.txt",
    #     "r",
    # ) as f:
    #     for args_str in f:
    #         # break if the line is empty
    #         if not args_str or args_str == "\n":
    #             break
    #         args = parse_args(args_str)
    #         train(args)
    
    # [parse the args from the command line]
    args = parse_args()
    train(args)



    # env = make_env_minigrid(env_id="MiniGrid-Empty-6x6-v0")
    # print(env.observation_space.shape)

    # import gymnasium

    # env = gymnasium.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")
    # env.reset()
    # for _ in range(100000):
    #     action = env.action_space.sample()
    #     # print("action: ", action)
    #     # print("[Before Step] env.agent_pos, env.agent_dir: ", env.agent_pos, env.agent_dir)
    #     next_state, reward, terminated, truncated, info = env.step(action)
    #     state = next_state
    #     if terminated or truncated:
    #         state, info = env.reset()
