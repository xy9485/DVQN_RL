import copy
import os

import gymnasium as gym
import numpy as np
from agents import Agent, get_agent
from buffer import ReplayBuffer
from utils import AlphaDecayScheduler, CountBasedAlphaScheduler, Logger, CountBasedEpsilonScheduler, EpsilonDecayScheduler, TemperatureDecayScheduler


def evaluate(eval_env, start_loc, goal_loc, agent, eval_total_steps, logger: Logger):
    eval_acc_reward = 0
    eval_step = 0
    eval_done = False
    eval_trunacted = False
    eval_obs, eval_info = eval_env.reset(options={"start_loc": start_loc, "goal_loc": goal_loc})
    init_obs = eval_obs
    list_max_Q = []
    while eval_step < eval_total_steps:
        if eval_done or eval_trunacted:
            eval_obs, eval_info = eval_env.reset(options={"start_loc": start_loc, "goal_loc": goal_loc})
        eval_action, action_info = agent.choose_action(eval_obs, greedy=True)
        eval_next_obs, eval_reward, eval_done, eval_trunacted, eval_info = eval_env.step(eval_action)
        list_max_Q.append(action_info["Q"])
        if init_obs == eval_obs:
            start_state_maxQ = action_info["Q"]
        eval_step += 1
        # eval_acc_reward += eval_reward
        eval_obs = eval_next_obs
    return {"avg_max_Q": np.mean(list_max_Q), "start_state_maxQ": start_state_maxQ}

def evaluate_one_episode(eval_env, start_loc, goal_loc, agent):
    eval_acc_reward = 0
    eval_step = 0
    eval_done = False
    eval_trunacted = False
    eval_obs, eval_info = eval_env.reset(options={"start_loc": start_loc, "goal_loc": goal_loc})
    init_obs = eval_obs
    list_max_Q = []
    rewards = []
    while not eval_done and not eval_trunacted:
        eval_action, action_info = agent.choose_action(eval_obs, greedy=False)
        eval_next_obs, eval_reward, eval_done, eval_trunacted, eval_info = eval_env.step(eval_action)
        list_max_Q.append(action_info["Q"])
        rewards.append(eval_reward)
        if init_obs == eval_obs:
            start_state_maxQ = action_info["Q"]
        eval_step += 1
        # eval_acc_reward += eval_reward
        eval_obs = eval_next_obs

    # Calculate the actual discounted returns
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + agent.gamma * G
        returns.append(G)

    # Reverse the returns list to correspond to the original trajectory order
    # returns.reverse()

    return {"avg_max_Q": np.mean(list_max_Q), "start_state_maxQ": start_state_maxQ, "avg_return": np.mean(returns)}

def run_policy(env, start_loc, goal_loc, agent, discount_factor):
    """
    Runs the policy derived from the Q-table and computes the actual discounted returns.

    Parameters:
    env - The environment (assumes OpenAI Gym-like environment)
    Q - The final Q-table
    discount_factor - The discount factor for future rewards
    max_steps - The maximum number of steps to run the policy

    Returns:
    returns - List of actual discounted returns for each visited state in the trajectory
    """
    state, info = env.reset(options={"start_loc": start_loc, "goal_loc": goal_loc})
    done = False
    truncated = False
    total_return = 0.0
    steps = 0
    trajectory = []

    # Run the policy until the episode is done or max_steps is reached
    while not done and not truncated:
        # Choose the best action from the Q-table
        action, action_info = agent.choose_action(state, greedy=True)

        # Take the action in the environment
        next_state, reward, done, truncated, _ = env.step(action)

        # Append the current state and reward to the trajectory
        trajectory.append((state, next_state, reward))

        # Update state
        state = next_state
        steps += 1

    # Calculate the actual discounted returns
    returns = []
    G = 0
    for _, _, reward in reversed(trajectory):
        G = reward + discount_factor * G
        returns.append(G)

    # Reverse the returns list to correspond to the original trajectory order
    returns.reverse()

    rewards = [r for _, _, r in trajectory]

    return returns, rewards


def run(
    env,
    start_loc,
    goal_loc,
    agent: Agent,
    config: dict,
    repeat_idx: int,
    total_steps: int,
    logger: Logger,
):
    replay_buffer = ReplayBuffer(obs_dim=1, size=config["buffer_size"], batch_size=config["batch_size"])

    obs, info = env.reset(options={"start_loc": start_loc, "goal_loc": goal_loc})
    # done = env.unwrapped.done
    done = False
    truncated = False
    episode_reward = 0
    episode_idx = 0
    learn_info_list = []
    trajectory = [obs]
    n_reset_buffer = 0
    n_learn = 0
    n_on_policy = 0
    running_avg = 1.0
    for step in range(total_steps):
        if done or truncated or step == total_steps - 1:
            # print(f'repeat_idx: {repeat_idx} | step: {step} | episode reward:{episode_reward} | maxQ: {np.max(agent.Q)} | eps: {agent.eps}')
            data = {
                "repeat_idx": repeat_idx,
                "episode_idx": episode_idx,
                "step": step,
                "reward": episode_reward,
                "maxQ": np.max(agent.Q),
                "len_episode": len(trajectory)-1,
            }
            if isinstance(agent.explore_scheduler, EpsilonDecayScheduler):
                data.update({"eps": agent.explore_scheduler.get_value()})
            elif isinstance(agent.explore_scheduler, TemperatureDecayScheduler):
                data.update({"temperature": agent.explore_scheduler.get_value()})
            elif isinstance(agent.explore_scheduler, CountBasedEpsilonScheduler):
                # eps is state dependent
                pass
            else:
                raise ValueError("Unknown explore scheduler")
            data.update(info)
            # update done and truncated
            data.update({"done": done, "truncated": truncated})

            if done or truncated:
                if env.unwrapped.agent_xy == env.unwrapped.goals_xy[0]:
                    data.update({"reach_desired_goal": 0})
                else:
                    data.update({"reach_desired_goal": 1})
            logger.dump_episodic_data(data)

            obs, info = env.reset(options={"start_loc": start_loc, "goal_loc": goal_loc})
            episode_reward = 0
            episode_idx += 1

            # print(f"repeat:{repeat_idx} trajectory: {trajectory} \n")
            # with open("results/temp.log", 'a') as f:
            #     f.write(f"repeat:{repeat_idx} step {step} done {done} truncated {truncated}trajectory: {trajectory}" + '\n')
            trajectory = [obs]

        if (step + 1) % 100 == 0:
            logger.dump({"repeat_idx": repeat_idx, "step": step})

        action, action_info = agent.choose_action(obs)

        new_obs, reward, done, truncated, info = env.step(action)
        replay_buffer.store(obs, action, reward, new_obs, float(done), action_info["action_prob"])
        # logger.log(key='max_Q(s,a)', value=action_info['max_Q'])

        # [update exploration parameters]
        if isinstance(agent.explore_scheduler, EpsilonDecayScheduler) or isinstance(
            agent.explore_scheduler, TemperatureDecayScheduler
        ):
            agent.explore_scheduler.step()
        elif isinstance(agent.explore_scheduler, CountBasedEpsilonScheduler):
            agent.explore_scheduler.step(obs)
        else:
            raise ValueError("Unknown explore scheduler")
        # [update alpha]
        if isinstance(agent.alpha_scheduler, AlphaDecayScheduler):
            agent.alpha_scheduler.step()
        elif isinstance(agent.alpha_scheduler, CountBasedAlphaScheduler):
            agent.alpha_scheduler.step(obs, action)
        else:
            raise ValueError("Unknown alpha scheduler")

        if len(replay_buffer) >= replay_buffer.batch_size:
            if config["algo_name"] == "VQ-learning" and config["importance_sampling"] > 0:
                trans = replay_buffer.sample_batch()
                if config["syncVQ"]:
                    learn_info = agent.learn_syncVQ(
                        trans["obs"],
                        trans["acts"],
                        trans["rews"],
                        trans["next_obs"],
                        trans["done"],
                        prob_action=trans["prob_acts"],
                    )
                else:
                    learn_info = agent.learn_importance_sampling(
                        trans["obs"],
                        trans["acts"],
                        trans["rews"],
                        trans["next_obs"],
                        trans["done"],
                        prob_action=trans["prob_acts"],
                    )
                n_learn += 1
                n_on_policy += 1
                # running_avg = (running_avg + np.mean(learn_info['mask_'])) / (n_on_policy + 1)
                # running_avg = running_avg * 0.5 + 0.5 * np.mean(learn_info['mask_'])
                # running_avg = np.mean(buf)
                running_avg = np.mean(learn_info["mask_"])
                if running_avg < config["importance_sampling"]:
                    replay_buffer.reset()
                    n_reset_buffer += 1
                    if config["syncVQ"]:
                        agent.sync_VQ()
                    # running_avg = 0
                    # n_on_policy = 0
                else:
                    pass
                    # print(f"np.mean(l_on_policy): {np.mean(l_on_policy)}")

                learn_info.update(
                    {
                        "n_learn": n_learn,
                        "n_reset_buffer": n_reset_buffer,
                        "n_on_policy": n_on_policy,
                        "running_avg": running_avg,
                    }
                )
                learn_info_list.append(learn_info)
                for k, v in learn_info.items():
                    logger.log(key=k, value=v)
            else:
                trans = replay_buffer.sample_batch()
                learn_info = agent.learn(
                    trans["obs"],
                    trans["acts"],
                    trans["rews"],
                    trans["next_obs"],
                    trans["done"],
                )
                learn_info_list.append(learn_info)
                for k, v in learn_info.items():
                    logger.log(key=k, value=v)
            if config["noisy_update"] > 0.0:
                agent.simulate_noisy_update(noise=config["noisy_update"], gaussian=False)
        trajectory.append(new_obs)

        # [Evaluation]
        if step % config["eval_freq"] == 0:
            # eval_dict = evaluate(
            #     eval_env=copy.deepcopy(env),
            #     start_loc=start_loc,
            #     goal_loc=goal_loc,
            #     agent=agent,
            #     eval_total_steps=config["eval_total_steps"],
            #     logger=logger,
            # )
            eval_dict = evaluate_one_episode(
                eval_env=copy.deepcopy(env),
                start_loc=start_loc,
                goal_loc=goal_loc,
                agent=agent,
            )
            eval_dict.update({"step": step, "repeat_idx": repeat_idx})
            logger.dump_data(eval_dict, path=logger.eval_log_path)

        episode_reward += reward
        obs = new_obs

    # print(f"n_reset_buffer: {n_reset_buffer}, n_learn: {n_learn}")
    env.close()


def run_experiment(MDP, config, n_repeat=8):
    total_steps = config["total_steps"]
    MOVES = MDP["MOVES"]
    obstacle_map = MDP["obstacle_map"]
    start_loc = MDP["start_loc"]
    goal_loc = MDP["goal_loc"]
    map_name = MDP["map_name"]
    env_name = MDP["env_name"]

    env = gym.make(
        env_name,
        obstacle_map=obstacle_map,
        # render_mode='human',
        MOVES=MOVES,  # if None, it will use the default 4 moves
        render_mode=None,
        max_episode_steps=config['max_episode_steps'],  # This is for TimeLimit wrapper
    )


    algo_name = config["algo_name"]
    if config.get("temperature", None) is not None and isinstance(config["temperature"], list):
        temp_init, temp_min, temp_decay = config["temperature"]
        explore_scheduler = TemperatureDecayScheduler(
            init_value=temp_init,
            min_value=temp_min,
            decay=temp_decay,
        )
        log_dir = (
        f"results/{map_name}/{algo_name}/randinit{int(config['random_value_init'])}temp[{temp_init},{temp_min},{temp_decay}]_ALPHA{config['alpha']}_buffer{config['buffer_size']}_{config['batch_size']}_IS{config['importance_sampling']}_noisyV{config['noisy_update']}_repeat{n_repeat}_steps{total_steps}"
        + "--")
    elif isinstance(config["eps"], list):
        eps_init, eps_min, eps_decay = config["eps"]
        explore_scheduler = EpsilonDecayScheduler(
            init_value=eps_init,
            min_value=eps_min,
            decay=eps_decay,
        )
    elif config["eps"] == "count_based":
        explore_scheduler = CountBasedEpsilonScheduler(init_value=1.0, min_value=0.0)
    else:
        raise ValueError("Unknown explore scheduler argument")

    if isinstance(config["alpha"], list):
        if len(config["alpha"]) == 3:
            alpha_init, alpha_min, alpha_decay = config["alpha"]
            alpha_scheduler = AlphaDecayScheduler(init_value=alpha_init, min_value=alpha_min, decay=alpha_decay)
        else:
            raise ValueError("config['alpha'] should be a list of 3 values")
    elif config["alpha"] == "count_based":
        alpha_scheduler = CountBasedAlphaScheduler(init_value=1, min_value=0.0, eta=0.8)
    else:
        raise ValueError("Unknown alpha scheduler argument")

    log_dir = (
    f"results/{map_name}/nA{config['n_actions']}_randinit{int(config['random_value_init'])}_Eps|{config['eps']}|_Alpha|{config['alpha']}|_Buff{config['buffer_size']}-{config['batch_size']}_noisyV{config['noisy_update']}_repeat{n_repeat}_steps{total_steps}/{algo_name}")

    log_paths = {
        "avg_meter": os.path.join(log_dir, "avg_meter.log"),
        "episodic": os.path.join(log_dir, "episodic.log"),
        "true_return": os.path.join(log_dir, "true_return.log"),
        "true_reward": os.path.join(log_dir, "true_reward.log"),
        "eval": os.path.join(log_dir, "eval.log"),
    }
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for log_path in log_paths.values():
        if os.path.exists(log_path):
            os.remove(log_path)

    logger = Logger(log_paths=log_paths)

    # algo = "VQ-learning"
    # if results/temp.log exists, remove it
    if os.path.exists("results/temp.log"):
        os.remove("results/temp.log")

    best_policy = None
    avg_true_return = -np.inf
    max_Q_matrices = []
    for i_repeat in range(n_repeat):
        agent = get_agent(config, env)

        agent.explore_scheduler = explore_scheduler
        agent.explore_scheduler.reset()

        agent.alpha_scheduler = alpha_scheduler
        agent.alpha_scheduler.reset()

        run(env, start_loc, goal_loc, agent, config=config, repeat_idx=i_repeat, total_steps=total_steps, logger=logger)
        # print(f"repeat {i_repeat} done")

        # [Evaluate the final policy, log and save the new best policy]
        avgs_true_returns = []
        list_true_returns = []
        list_true_rewards = []
        n_final_evals = config['n_final_evals']
        for _ in range(n_final_evals):  # considering the transition function of the mdp can be stochastic
            true_returns, true_rewards = run_policy(copy.deepcopy(env), start_loc, goal_loc, agent, discount_factor=config["gamma"])
            logger.dump_data(true_returns, log_paths["true_return"], overwrite=False)
            logger.dump_data(true_rewards, log_paths["true_reward"], overwrite=False)
            # list_true_returns.append(true_returns)
            # list_true_rewards.append(true_rewards)
            # avgs_true_returns.append(np.mean(true_returns))
            # logger.dump_data(true_returns, log_paths["true_return"], overwrite=False)
            # logger.dump_data(true_rewards, log_paths["true_reward"], overwrite=False)

        # if np.mean(avgs_true_returns) > avg_true_return:
        #     avg_true_return = np.mean(avgs_true_returns)
        #     best_policy = copy.deepcopy(agent.Q)
        #     # save the best policy
        #     np.save(os.path.join(log_dir, "best_policy.npy"), best_policy)

        #     with open(log_paths["true_return"], "w") as f:
        #         pass
        #     for true_returns in list_true_returns:
        #         logger.dump_data(true_returns, log_paths["true_return"], overwrite=False)
        #     for true_rewards in list_true_rewards:
        #         logger.dump_data(true_rewards, log_paths["true_reward"], overwrite=False)
        max_Q_matrices.append(copy.deepcopy(agent.Q))
    #print mean of max_Q_matrices
    # print("max_Q_matrices shape", np.stack(max_Q_matrices, axis=2).mean(axis=2).max(axis=1).shape)
    # print(np.stack(max_Q_matrices, axis=2).mean(axis=2).max(axis=1).reshape(-1, 3))
    return log_dir


if __name__ == "__main__":
    import gym_simplegrid

    MOVES = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),  # RIGHT
        # 4: (0, 0),  # STAY
        # 5: (0, 0),
        # 6: (0, 0),
        # 7: (0, 0),
        # 8: (0, 0),
        # 9: (0, 0),
    }
        # Create a obstacle map with 3x3 grid from paper Double Q-learning
    obstacle_map = [
        "000",
        "000",
        "000",
    ]
    start_loc = (2, 0)
    goal_loc = [(0,2),]
    map_name = "GridInDoubleQ"
    env_name = "GridEnvFromDoubleQLearning-v0"

    MDP = {
        "obstacle_map": obstacle_map,
        "start_loc": start_loc,
        "goal_loc": goal_loc,
        "map_name": map_name,
        "env_name": env_name,
        "MOVES": MOVES,
    }

#     config = {
#     "eps":"count_based", # [init, min, decay] like [0.5, 0.0, 0.99995] or "count_based"
#     # "temperature": [1.0, 0.01, 0.99995], # [init, min, decay]
#     "alpha": "count_based", # [init, min, decay] like [0.1, 0.01, 1.0] or "count_based"
#     "gamma": 0.99,
#     "random_value_init": True,
#     "alpha_v": 0.1,
#     "alpha_q": 0.1,
#     "buffer_size": 1,
#     "batch_size": 1,
#     "importance_sampling": 0.0,  # this will work as a threshold
#     "syncVQ": False,
#     "eval_freq": 500,
#     "eval_total_steps": 200,
#     "noisy_update": 0.0,
# }

    # Double Q-learning 3x3 GridWorld
    config = {
        # "eps": [0.1, 0.0, 0.997], # [init, min, decay] like [0.5, 0.0, 0.99995] or "count_based"
        "eps": "count_based",
        # "temperature": [1.0, 0.01, 0.99995], # [init, min, decay]
        # "alpha": [0.1, 0.00, 0.9995], # [init, min, decay] like [0.1, 0.01, 1.0] or "count_based"
        "alpha": "count_based",
        "gamma": 0.95,
        "random_value_init": False,
        "alpha_v": 0.1,
        "alpha_q": 0.1,
        "buffer_size": 1,
        "batch_size": 1,
        "importance_sampling": 0.0,  # this will work as a threshold
        "syncVQ": False,
        "noisy_update": 0.0,
        "eval_freq": 50,
        "eval_total_steps": 100,
        "max_episode_steps": 100,
        "total_steps": 10000,
        "n_final_evals": 10,
        "n_actions": len(MOVES),
    }

    algo_names = ["Q-learning", "VQ-learning", "DoubleQ-learning", "SARSA", "ExpectedSARSA"]
    # algo_names = ["SARSA",]
    log_dirs = []
    for algo_name in algo_names:
        config["algo_name"] = algo_name
        log_dir = run_experiment(MDP, config, n_repeat=10)
        # more hyperparameters to be set within run_experiment
        print(f"{algo_name} done!")
        print(log_dir)
        log_dirs.append(log_dir)

