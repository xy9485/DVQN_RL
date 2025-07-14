#!/bin/bash

# Define the output directory
OUTPUT_DIR="shell_output"
# create the output directory if it does not exist
mkdir -p ${OUTPUT_DIR}

#replace "dvqn" with your own conda environment
source activate dvqn

# Define the array of domain names
# domain_names=("Carnival-v5" "Alien-v5" "Riverraid-v5" "Asterix-v5" "Breakout-v5" "Boxing-v5" "Phoenix-v5" "Pong-v5" "DemonAttack-v5" "SpaceInvaders-v5")
domain_names=("Carnival-v5")
clip_grad_mode="norm"
freq_Q_sync=1
tau_Q_encoder=0.001
tau_Q_critic=0.001
repetitions=6
total_timesteps=1000000
size_replay_memory=1000000
redundant_actions=0
explore_final_fraction=0.1
wandb_mode="offline"
extra_note="your_note"

algo=$1
echo "algo: ${algo}"
# if algo is "dvqn", then set the following parameters
case $algo in
    dvqn)
        freq_V_sync=1
        tau_V_encoder=0.001
        tau_V_critic=0.001

        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo} --share_encoder --use2Q --freq_V_sync ${freq_V_sync} --tau_V_encoder ${tau_V_encoder} --tau_V_critic ${tau_V_critic}"
        ;;
    dqn)
        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo}"
        ;;
    ddqn)
        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo}"
        ;;
    sarsa)
        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo}"
        ;;
    avgdqn)
        avgdqn_k=10
        # avgdqn_k denotes number of previous learned Q values to average.
        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo} --avgdqn_k ${avgdqn_k}"
        ;;   
    cddqn)
        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo}"
        ;;
    dueldqn)
        common_args="--use_obs_augmentation --clip_reward --clip_grad_mode ${clip_grad_mode} --freq_Q_sync ${freq_Q_sync} --tau_Q_encoder ${tau_Q_encoder} --tau_Q_critic ${tau_Q_encoder} --wandb_mode ${wandb_mode} --repetitions ${repetitions} --size_replay_memory ${size_replay_memory} --total_timesteps ${total_timesteps} --redundant_actions ${redundant_actions} --explore_final_fraction ${explore_final_fraction} --extra_note ${extra_note} ${algo}"
        ;;  
esac

ARRAY_START=0
ARRAY_END=$((${#domain_names[@]} - 1))

echo "ARRAY_START: ${ARRAY_START}"
echo "ARRAY_END: ${ARRAY_END}"

# pid_file="pids_${algo}.txt"
# # if pid_files exists, delete it
# if [ -f $pid_file ]; then
#     rm $pid_file
# fi

# Loop through and start training for each domain name
for i in $(seq $ARRAY_START $ARRAY_END); do
    echo "start training in domain: ${domain_names[$i]}"
    domain_name=${domain_names[$i]}

    # Combine the domain name with the common arguments
    args="--domain_name ${domain_name} ${common_args}"

    # Run the job
    python -u ../latentrl/train_dvqn.py ${args} > ${OUTPUT_DIR}/output_${domain_name}_dvqn_${extra_note}.out 2> ${OUTPUT_DIR}/error_${domain_name}_dvqn_${extra_note}.err & run_pid=$!
    echo "run_pid: ${run_pid}"
    # echo $run_pid >> $pid_file
done

#make pid_file read-only
# chmod 444 $pid_file

echo "Processes are running in the background."

# Wait for all background jobs to finish
# wait