from collections import deque, namedtuple
import random

import numpy as np
import torchvision.transforms as T


class ReplayMemory(object):
    def __init__(
        self,
        capacity,
    ):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "terminated")
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
        return self.Transition(*zip(*transitions))
        # return random.sample(self.memory, self.batch_size)

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
