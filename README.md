# This is the repository for the NIPS2023 submission: Regulating Action Value Estimation in Deep Reinforcement Learning

## Requirements
```
conda env create -f conda_env.yml
```
## Instructions
The training is logged by wandb by default.

- To train a DVQN agent:
```
python train_hdqn.py --domain_name Riverraid-v5 --no-use_dueling --grd_mode dqn --use_abs_V --share_encoder --no-per --no-dan --use_curl off --curl_pair temp --clip_reward --clip_grad --approach_abs_factor 0.5 --grd_encoder_linear_dims -1 --abs_encoder_linear_dims -1 --curl_projection_dims -1 --no-grd_enc_detach --freq_grd_sync 1000 --freq_abs_sync 1000 --tau_grd_encoder 1 --tau_grd_critic 1 --tau_abs_encoder 1 --tau_abs_critic 1 --exploration 0.99 0.1 0.1 --wandb_mode online --repetitions 20 --size_replay_memory 100000 --total_timesteps 100000 --extra_note dvqn
```
Running domain is set via `--domain_name`

- To train a DQN agent:
```
python train_hdqn.py --domain_name Riverraid-v5 --no-use_dueling --grd_mode dqn --no-use_abs_V --no-share_encoder --no-per --no-dan --use_curl off --curl_pair temp --clip_reward --clip_grad --approach_abs_factor 0.0 --grd_encoder_linear_dims -1 --abs_encoder_linear_dims -1 --curl_projection_dims -1 --no-grd_enc_detach --freq_grd_sync 1000 --freq_abs_sync 1000 --tau_grd_encoder 1 --tau_grd_critic 1 --tau_abs_encoder 1 --tau_abs_critic 1 --exploration 0.99 0.1 0.1 --wandb_mode online --repetitions 20 --size_replay_memory 100000 --total_timesteps 100000 --extra_note dqn
```
- To train a DDQN/CDDQN agent, replace --grd_mode and --extra_note above with `ddqn` or `cddqn`

- To train a Averaged DQN:

```
python train_hdqn.py --domain_name Riverraid-v5 --no-use_dueling --grd_mode avgdqn --avgdqn_k 10 --no-use_abs_V --no-share_encoder --no-per --no-dan --use_curl off --curl_pair temp --clip_reward --clip_grad --approach_abs_factor 0.0 --grd_encoder_linear_dims -1 --abs_encoder_linear_dims -1 --curl_projection_dims -1 --no-grd_enc_detach --freq_grd_sync 1000 --freq_abs_sync 1000 --tau_grd_encoder 1 --tau_grd_critic 1 --tau_abs_encoder 1 --tau_abs_critic 1 --exploration 0.99 0.1 0.1 --wandb_mode online --repetitions 20 --size_replay_memory 100000 --total_timesteps 1000000 --freq_eval 10000 --extra_note avgdqn
```

`avgdqn_k` denotes number of previous learned Q values to average. Averaged DQN's performance also varies depending on `freq_grd_sync`.

- To train a Dueling-DQN agent:
```
python train_hdqn.py --domain_name Riverraid-v5 --use_dueling --grd_mode dqn --no-use_abs_V --no-share_encoder --no-per --no-dan --use_curl off --curl_pair temp --clip_reward --clip_grad --approach_abs_factor 0.0 --grd_encoder_linear_dims -1 --abs_encoder_linear_dims -1 --curl_projection_dims -1 --no-grd_enc_detach --freq_grd_sync 1000 --freq_abs_sync 1000 --tau_grd_encoder 1 --tau_grd_critic 1 --tau_abs_encoder 1 --tau_abs_critic 1 --exploration 0.99 0.1 0.1 --wandb_mode online --repetitions 20 --size_replay_memory 100000 --total_timesteps 100000 --extra_note dueling-dqn
```

- To train DVQN combined with temporal difference objective (TC):
```
python train_hdqn.py --domain_name Riverraid-v5 --no-use_dueling --grd_mode dqn --use_abs_V --share_encoder --no-per --no-dan --use_curl on_grd --curl_pair temp --clip_reward --clip_grad --approach_abs_factor 0.5 --grd_encoder_linear_dims -1 --abs_encoder_linear_dims -1 --curl_projection_dims 256 128 --no-grd_enc_detach --freq_grd_sync 1000 --freq_abs_sync 1000 --tau_grd_encoder 1 --tau_grd_critic 1 --tau_abs_encoder 1 --tau_abs_critic 1 --exploration 0.99 0.1 0.1 --wandb_mode online --repetitions 20 --size_replay_memory 100000 --total_timesteps 100000 --extra_note dvqn+TC
```
- To train DVQN combined with CURL objective, replace --curl-pair above with `raw`

- To train DQN combined with temporal difference objective (TC) objective:
```
python train_hdqn.py --domain_name Riverraid-v5 --no-use_dueling --grd_mode dqn --no-use_abs_V --no-share_encoder --no-per --no-dan --use_curl on_grd --curl_pair temp --clip_reward --clip_grad --approach_abs_factor 0.0 --grd_encoder_linear_dims -1 --abs_encoder_linear_dims -1 --curl_projection_dims 256 128 --no-grd_enc_detach --freq_grd_sync 1000 --freq_abs_sync 1000 --tau_grd_encoder 1 --tau_grd_critic 1 --tau_abs_encoder 1 --tau_abs_critic 1 --exploration 0.99 0.1 0.1 --wandb_mode disabled --repetitions 20 --size_replay_memory 100000 --total_timesteps 100000 --extra_note dqn+TC
```
- To train DQN combined with CURL objective, replace --curl-pair above with `raw`
