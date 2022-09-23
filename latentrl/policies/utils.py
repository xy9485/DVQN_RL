from collections import deque, namedtuple
import random
import torch
import numpy as np
import torchvision.transforms as T


class ReplayMemory(object):
    def __init__(self, capacity, device):
        self.memory = deque([], maxlen=capacity)
        self.device = device
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "terminated", "info")
        )

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(
        self,
        batch_size=None,
        validation_size=None,
    ):
        if validation_size:
            transitions = random.sample(self.memory, validation_size)
        else:
            transitions = random.sample(self.memory, batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
        state_batch = np.stack(batch.state, axis=0).transpose(0, 3, 1, 2)
        next_state_batch = np.stack(batch.next_state, axis=0).transpose(0, 3, 1, 2)
        state_batch = torch.from_numpy(state_batch).contiguous().float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).contiguous().float().to(self.device)
        # batch = self.memory.lazy_sample(batch_size=self.batch_size)
        # state_batch = torch.cat(batch.state).to(self.device)
        # next_state_batch = torch.cat(batch.next_state).to(self.device)
        # action_batch = torch.cat(batch.action).to(self.device)
        # reward_batch = torch.cat(batch.reward).to(self.device)
        # terminated_batch = torch.cat(batch.terminated).to(self.device)

        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(self.device)
        terminated_batch = torch.tensor(batch.terminated).unsqueeze(1).to(self.device)
        info_batch = batch.info

        return (
            state_batch / 255.0,
            action_batch,
            next_state_batch / 255.0,
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
