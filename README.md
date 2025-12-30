**This is the repository for paper:**

**Towards Unbiased Action Value Estimation in Reinforcement Learning**


### Run DVQN and Baselines
The training requires WANDB to log data

To train a agent, go to ./shell_scripts and run:
```
./train.sh algo_name
```
`algo_name` is to be replaced by the following methods:

 `dvqn, dqn, sarsa, ddqn, cddqn, avgdqn, dueldqn`

Hyperparameters are to be modified within `train.sh`

### Run Tabular VQ-learning and Baselines
- trainig over Toy MDP, see ./toy_mdp/compare_TD_algos.ipynb
- training over 2-room gridworld, see ./gridworld/two_room.ipynb