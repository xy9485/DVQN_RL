import math
from operator import itemgetter
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torchvision.transforms as T

from nn_models import EncoderImg
from typing import Any, Deque, Dict, List, Optional, Tuple, Type, TypeVar, Union
from torch import Tensor


class ReplayMemory(object):
    def __init__(self, capacity, device, gamma=0.99, batch_size=32):
        self.memory = deque([], maxlen=capacity)
        self.recent_goal_transitions = []
        self.device = device
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "terminated", "info")
        )
        self.gamma = gamma
        self.batch_size = batch_size

    def push(self, state, action, next_state, reward, terminated, info):
        """Save a transition"""
        self.memory.append(self.Transition(state, action, next_state, reward, terminated, info))
        self.latest_transition = self.Transition(
            state, action, next_state, reward, terminated, info
        )
        if terminated:
            self.recent_goal_transitions.append(
                self.Transition(state, action, next_state, reward, terminated, info)
            )

    # @property
    # def recent_goal_transitions(self):
    #     batch = self.Transition(*zip(*self._recent_goal_transitions))
    #     return (
    #         batch.state,
    #         batch.action,
    #         batch.next_state,
    #         batch.reward,
    #         batch.terminated,
    #         batch.info,
    #     )

    # @recent_goal_transitions.setter
    # def recent_goal_transitions(self, value):
    #     self._recent_goal_transitions = value

    def sample(self, batch_size=None, validation_size=None, mode=None):
        if validation_size:
            transitions = random.sample(self.memory, validation_size)
        else:
            transitions = random.sample(self.memory, batch_size)

        if mode == "pure":
            if len(self.recent_goal_transitions) > 0:
                transitions += self.recent_goal_transitions
                self.recent_goal_transitions = []
            batch = self.Transition(*zip(*transitions))
            return (
                batch.state,
                batch.action,
                batch.next_state,
                batch.reward,
                batch.terminated,
                batch.info,
            )
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
        # state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        # next_state_batch = np.stack(batch.next_state, axis=0).transpose(0, 3, 1, 2)
        # state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)
        # next_state_batch = torch.from_numpy(next_state_batch).contiguous().float().to(self.device)
        state_batch = np.stack(batch.state, axis=0)
        next_state_batch = np.stack(batch.next_state, axis=0)
        state_batch = torch.from_numpy(state_batch).to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).to(self.device)
        # batch = self.memory.lazy_sample(batch_size=self.batch_size)
        # state_batch = torch.cat(batch.state).to(self.device)
        # next_state_batch = torch.cat(batch.next_state).to(self.device)
        # action_batch = torch.cat(batch.action).to(self.device)
        # reward_batch = torch.cat(batch.reward).to(self.device)
        # terminated_batch = torch.cat(batch.terminated).to(self.device)

        action_batch = torch.as_tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.as_tensor(batch.reward).unsqueeze(1).to(self.device)
        terminated_batch = torch.as_tensor(batch.terminated).unsqueeze(1).to(self.device)
        info_batch = batch.info

        return (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            terminated_batch,
            info_batch,
        )

    def __len__(self):
        return len(self.memory)

    def lazy_to_tensor(self, batch):
        states = list(batch.state)
        next_states = list(batch.next_state)
        for index, (state, next_state) in enumerate(zip(states, next_states)):
            state = np.array(state)
            next_state = np.array(state)
            states[index] = T.ToTensor()(state).float().unsqueeze(0)
            next_states[index] = T.ToTensor()(next_state).float().unsqueeze(0)
        return self.Transition(states, batch.action, next_states, batch.reward, batch.done)

    def lazy_sample(
        self,
        batch_size=None,
        validation_size=None,
    ):
        """combination of def sample and def layzy_to_tensor"""
        if validation_size:
            transitions = random.sample(self.memory, validation_size)
        else:
            transitions = random.sample(self.memory, batch_size)
        batch = self.Transition(*zip(*transitions))

        states = list(batch.state)
        next_states = list(batch.next_state)
        for index, (state, next_state) in enumerate(zip(states, next_states)):
            state = np.array(state)
            next_state = np.array(state)
            states[index] = T.ToTensor()(state).float().unsqueeze(0)
            next_states[index] = T.ToTensor()(next_state).float().unsqueeze(0)
        return self.Transition(states, batch.action, next_states, batch.reward, batch.done)

    def __len__(self):
        return len(self.memory)


class ReplayBufferNStep(object):
    def __init__(self, capacity, device, gamma=0.99, batch_size=32):
        self.memory = deque([], maxlen=capacity)
        self.recent_goal_transitions = []
        self.device = device
        self.Transition = namedtuple("Transition", ("obs", "act", "n_obs", "rew", "gamma", "info"))
        self.gamma = gamma
        self.batch_size = batch_size

    def push(self, transition):
        """
        Save a transition:
        obs, act, n_obs, rew, terminated, info
        """
        obs, act, n_obs, rew, gamma, info = transition
        self.memory.append(self.Transition(obs, act, n_obs, rew, gamma, info))

    def sample(self, batch_size):

        transitions = random.sample(self.memory, batch_size)
        B = self.Transition(*zip(*transitions))

        # obs_B = np.stack(B.obs, axis=0)
        # n_obs_B = np.stack(B.n_obs, axis=0)
        # obs_B = torch.from_numpy(obs_B).to(self.device)
        # n_obs_B = torch.from_numpy(n_obs_B).to(self.device)
        obs_B = torch.as_tensor(np.array(B.obs)).to(self.device)
        n_obs_B = torch.as_tensor(np.array(B.n_obs)).to(self.device)
        act_B = torch.as_tensor(B.act).unsqueeze(1).to(self.device)
        rew_B = torch.as_tensor(B.rew).unsqueeze(1).to(self.device)
        gamma_B = torch.as_tensor(B.gamma).unsqueeze(1).to(self.device)
        info_B = B.info

        return obs_B, act_B, n_obs_B, rew_B, gamma_B, info_B

    def sample_n_step_transits2(
        self,
        n_step: int,
    ) -> Dict[int, namedtuple]:
        # sampel windowed transitions by anchor index
        anchor_idxs, transitions = random.sample(list(enumerate(self.memory)), self.batch_size)

        n_step_transits_batch = {}
        for anchor_idx in anchor_idxs:
            n_step_transits = {}
            # for i in n_step:
            #     idx = anchor_idx - (i + 1)
            #     if idx < 0 or self.memory[idx].terminated:
            #         break
            #     else:
            #         pos[-(i + 1)] = self.memory[idx].state

            # anchor_transit = self.memory[anchor_idx]
            # if anchor_transit.terminate:
            #     n_step_transits[0] = {
            #         "next_obs": anchor_transit.next_state,
            #         "rew": anchor_transit.reward,
            #         "gamma": gamma,
            #     }
            #     return n_step_transits

            for i in n_step:
                idx = anchor_idx + i
                rew = 0
                if idx < len(self.memory):
                    rew = rew + math.pow(gamma, i) * self.memory[idx].reward
                    n_step_transits[i] = {
                        "obs": self.memory[idx].state,
                        "act": self.memory[idx].action,
                        "next_obs": self.memory[idx].next_state,
                        "rew": rew,
                        "gamma": math.pow(gamma, i + 1) if not self.memory[idx].terminate else 0,
                        "info": self.memory[idx].info,
                    }

                if self.memory[idx].terminated:
                    break

            n_step_transits_batch[anchor_idx] = n_step_transits

        N_Step_Transitions = {}
        for i in n_step:
            obs = []
            act = []
            next_obs = []
            rew = []
            gamma = []
            info = []
            for anchor_idx, n_step_transits in n_step_transits_batch.items():
                if i in n_step_transits.keys():
                    obs.append(n_step_transits[i]["obs"])
                    act.append(n_step_transits[i]["act"])
                    next_obs.append(n_step_transits[i]["next_obs"])
                    rew.append(n_step_transits[i]["rew"])
                    gamma.append(n_step_transits[i]["gamma"])
                    info.append(n_step_transits[i]["info"])

            obs = torch.from_numpy(np.stack(obs, axis=0)).to(self.device)
            next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).to(self.device)
            # n_step_transition_dict[i + 1] = (obs, next_obs)
            act = torch.as_tensor(act).unsqueeze(1).to(self.device)
            rew = torch.as_tensor(rew).unsqueeze(1).to(self.device)
            gamma = torch.as_tensor(gamma).unsqueeze(1).to(self.device)

            if len(info) > 0:
                N_Step_Transitions[i] = self.Transition(obs, act, next_obs, rew, gamma, info)

    def sample_n_step_transits(
        self,
        n_step: int,
        batch_size: int,
    ) -> List[None | Tuple[np.ndarray, ...]]:
        # sampel windowed transitions by anchor index
        # anchor_idxs, transitions = random.sample(list(enumerate(self.memory)), self.batch_size)
        anchor_idxs = random.sample(range(len(self.memory)), batch_size)

        N_Step_T = []
        n_transit_rew = [0] * len(anchor_idxs)
        for i in range(n_step):
            if i > 0 and N_Step_T[-1] == None:
                break
            obs = {}
            act = {}
            n_obs = {}
            rew = {}
            gamma = {}
            info = {}
            for j, anchor_idx in enumerate(anchor_idxs):
                if i > 0 and N_Step_T[-1]["gamma"].get(j, 0.0) == 0.0:
                    # if previous transition is terminated or the last one in memory, skip
                    continue
                idx = anchor_idx + i
                if idx == len(self.memory):
                    continue
                n_transit_rew[j] = n_transit_rew[j] + math.pow(self.gamma, i) * self.memory[idx].rew
                n_step_gamma = math.pow(self.gamma, i + 1) if self.memory[idx].gamma > 0.0 else 0.0
                obs[j] = self.memory[anchor_idx].obs
                act[j] = self.memory[anchor_idx].act
                n_obs[j] = self.memory[idx].n_obs
                rew[j] = n_transit_rew[j]
                gamma[j] = n_step_gamma
                info[j] = self.memory[idx].info

            N_Step_T.append(
                {
                    "obs": obs,
                    "act": act,
                    "n_obs": n_obs,
                    "rew": rew,
                    "gamma": gamma,
                    "info": info,
                }
            )
            pass
        return N_Step_T

    def __len__(self):
        return len(self.memory)


class ReplayMemoryWithCluster(object):
    def __init__(self, capacity, device):
        self.memory = deque([], maxlen=capacity)
        self.recent_goal_transitions = []
        self.device = device
        self.Transition = namedtuple(
            "Transition",
            (
                "state",
                "abs_state",
                "action",
                "next_state",
                "next_abs_state",
                "reward",
                "terminated",
                "info",
            ),
        )

    def push(self, state, abs_state, action, next_state, next_abs_state, reward, terminated, info):
        """Save a transition"""
        self.memory.append(
            self.Transition(
                state, abs_state, action, next_state, next_abs_state, reward, terminated, info
            )
        )
        self.latest_transition = self.Transition(
            state, abs_state, action, next_state, next_abs_state, reward, terminated, info
        )
        if terminated:
            self.recent_goal_transitions.append(
                self.Transition(
                    state, abs_state, action, next_state, next_abs_state, reward, terminated, info
                )
            )

    # @property
    # def recent_goal_transitions(self):
    #     batch = self.Transition(*zip(*self._recent_goal_transitions))
    #     return (
    #         batch.state,
    #         batch.action,
    #         batch.next_state,
    #         batch.reward,
    #         batch.terminated,
    #         batch.info,
    #     )

    # @recent_goal_transitions.setter
    # def recent_goal_transitions(self, value):
    #     self._recent_goal_transitions = value

    def sample(self, batch_size=None, validation_size=None, mode=None):
        if validation_size:
            transitions = random.sample(self.memory, validation_size)
        else:
            transitions = random.sample(self.memory, batch_size)

        if mode == "pure":
            # if len(self.recent_goal_transitions) > 0:
            #     transitions += self.recent_goal_transitions
            #     self.recent_goal_transitions = []
            batch = self.Transition(*zip(*transitions))
            return (
                batch.state,
                batch.abs_state,
                batch.action,
                batch.next_state,
                batch.next_abs_state,
                batch.reward,
                batch.terminated,
                batch.info,
            )
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
        # state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        # next_state_batch = np.stack(batch.next_state, axis=0).transpose(0, 3, 1, 2)
        # state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)
        # next_state_batch = torch.from_numpy(next_state_batch).contiguous().float().to(self.device)
        state_batch = np.stack(batch.state, axis=0)
        next_state_batch = np.stack(batch.next_state, axis=0)
        state_batch = torch.from_numpy(state_batch).to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).to(self.device)
        abs_state_batch = batch.abs_state
        next_abs_state_batch = batch.next_abs_state
        # batch = self.memory.lazy_sample(batch_size=self.batch_size)
        # state_batch = torch.cat(batch.state).to(self.device)
        # next_state_batch = torch.cat(batch.next_state).to(self.device)
        # action_batch = torch.cat(batch.action).to(self.device)
        # reward_batch = torch.cat(batch.reward).to(self.device)
        # terminated_batch = torch.cat(batch.terminated).to(self.device)

        action_batch = torch.as_tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.as_tensor(batch.reward).unsqueeze(1).to(self.device)
        terminated_batch = torch.as_tensor(batch.terminated).unsqueeze(1).to(self.device)
        info_batch = batch.info

        return (
            state_batch,
            abs_state_batch,
            action_batch,
            next_state_batch,
            next_abs_state_batch,
            reward_batch,
            terminated_batch,
            info_batch,
        )

    def lazy_to_tensor(self, batch):
        states = list(batch.state)
        next_states = list(batch.next_state)
        for index, (state, next_state) in enumerate(zip(states, next_states)):
            state = np.array(state)
            next_state = np.array(state)
            states[index] = T.ToTensor()(state).float().unsqueeze(0)
            next_states[index] = T.ToTensor()(next_state).float().unsqueeze(0)
        return self.Transition(states, batch.action, next_states, batch.reward, batch.done)

    def lazy_sample(
        self,
        batch_size=None,
        validation_size=None,
    ):
        """combination of def sample and def layzy_to_tensor"""
        if validation_size:
            transitions = random.sample(self.memory, validation_size)
        else:
            transitions = random.sample(self.memory, batch_size)
        batch = self.Transition(*zip(*transitions))

        states = list(batch.state)
        next_states = list(batch.next_state)
        for index, (state, next_state) in enumerate(zip(states, next_states)):
            state = np.array(state)
            next_state = np.array(state)
            states[index] = T.ToTensor()(state).float().unsqueeze(0)
            next_states[index] = T.ToTensor()(next_state).float().unsqueeze(0)
        return self.Transition(states, batch.action, next_states, batch.reward, batch.done)

    def pop(self):
        return self.memory.pop()

    def __len__(self):
        return len(self.memory)


def transition_np2torch(obs, act, n_obs, rew, gamma, info, device):
    obs = torch.from_numpy(np.stack(obs, axis=0)).to(device)
    n_obs = torch.from_numpy(np.stack(n_obs, axis=0)).to(device)
    act = torch.as_tensor(act).unsqueeze(1).to(device)
    rew = torch.as_tensor(rew).unsqueeze(1).to(device)
    gamma = torch.as_tensor(gamma).unsqueeze(1).to(device)
    return obs, act, n_obs, rew, gamma, info


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
