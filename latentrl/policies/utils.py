from collections import deque, namedtuple
import random


class ReplayMemory(object):
    def __init__(
        self,
        capacity,
    ):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "done")
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
