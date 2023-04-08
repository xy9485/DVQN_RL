import numpy as np
import random
import torch


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.max = 1.0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)
        self.max = max(self.max, p)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01
    alpha = 0.6
    # beta = 0.4
    beta_start = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, max_steps):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_steps = max_steps

    def beta_by_step(self, steps):
        return min(1.0, self.beta_start + steps * (1.0 - self.beta_start) / self.max_steps)

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def push(self, data):
        # p = self._get_priority(error)
        p = self.tree.max
        self.tree.add(p, data)

    def sample(self, n, agent_steps):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        # self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])
        self.beta = self.beta_by_step(agent_steps)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weight /= weight.max()

        return batch, idxs, weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
