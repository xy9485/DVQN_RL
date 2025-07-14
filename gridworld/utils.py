from collections import defaultdict
import json
import os

import numpy as np

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)

        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def dump(self, info:dict):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data = {**info, **data}
        self._dump_to_file(data)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_paths:dict):
        self._log_paths = log_paths
        self.episodic_log_path = log_paths["episodic"]
        self.avg_meter_log_path = log_paths["avg_meter"]
        self.eval_log_path = log_paths["eval"]
        self.true_return_log_path = log_paths["true_return"]
        self._mg = MetersGroup(
           self.avg_meter_log_path
        )
        
    def dump_episodic_data(self, data):
        with open(self.episodic_log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def dump_data(self, data, path, overwrite=False):
        if overwrite:
            with open(path, 'w') as f:
                pass
        with open(path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log(self, key, value, n=1):
        self._mg.log(key, value, n)


    def dump(self, info:dict):
        self._mg.dump(info=info)


# define a abstract class for epsilon scheduler
class Scheduler:
    def get_value(self):
        """Return the current value"""
        raise NotImplementedError

    def step(self):
        """Update the value and return it"""
        raise NotImplementedError

    def reset(self):
        """Reset the value to its starting value"""
        raise NotImplementedError


class DecayScheduler(Scheduler):
    def __init__(self, init_value, min_value, decay):
        """
        Initialize an epsilon scheduler for exploration-exploitation trade-off.

        Args:
            start (float): Starting value
            end (float): Minimum value
            decay (float): Multiplicative decay factor
        """
        self.init_value = init_value
        self.reset()
        self.min_value = min_value
        self.decay = decay

    def get_value(self):
        """Return the current value"""
        return self.v

    def step(self):
        """
        Decay value by the decay factor and return the new value.
        value will never go below min_value.
        """
        self.v = max(self.min_value, self.v * self.decay)

    def reset(self):
        """Reset value to its starting value"""
        self.v = self.init_value

class EpsilonDecayScheduler(DecayScheduler):
    def __init__(self, init_value=0.5, min_value=0.0, decay=0.99995):
        super().__init__(init_value, min_value, decay)

class TemperatureDecayScheduler(DecayScheduler):
    def __init__(self, init_value=1.0, min_value=0.0, decay=0.99995):
        super().__init__(init_value, min_value, decay)

class AlphaDecayScheduler(DecayScheduler):
    def __init__(self, init_value=0.1, min_value=0.0, decay=1.0):
        super().__init__(init_value, min_value, decay)

class CountBasedEpsilonScheduler(Scheduler):
    """
    v(s) = 1/sqrt(n(s)), where n(s) is the
    number of times state s has been visited.
    """
    def __init__(self, init_value=1, min_value=0.0):
        self.init_value = init_value
        self.min_value = min_value
        self.reset()

    def get_value(self, state:int):
        # assert state is int, "State must be an integer"
        # assert isinstance(state, int), "State must be an integer"
        state_count = self.state_counts[state]
        assert state_count > 0, "State count must be positive"
        value = max(self.min_value, 1 / (state_count ** 0.5))
        return value

    def step(self, state):
        assert isinstance(state, int), "State must be an integer"
        self.state_counts[state] += 1

    def reset(self):
        """
        Reset all state visit counts.
        """
        # Dictionary to track visit counts for each state
        self.state_counts = defaultdict(lambda: self.init_value)

class CountBasedEpsilonScheduler2(Scheduler):
    """
    v(s) = 1/sqrt(n(s)), where n(s) is the
    number of times state s has been visited.
    """
    def __init__(self, n_state, min_value=0.0):
        self.n_state = n_state
        self.reset()
        self.min_value = min_value

    def reset(self):
        self.state_counts = np.ones(self.n_state)

    def step(self, state:int):
        assert isinstance(state, int), "State must be an integer"
        self.state_counts[state] += 1

    def get_value(self, state:int):
        state_count = self.state_counts[state]
        assert state_count > 0, "State count must be positive"
        value = max(self.min_value, 1 / (state_count ** 0.5))
        return value

class CountBasedAlphaScheduler(Scheduler):
    """
    alpha(s,a)=1/n(s,a)^eta,
    """
    def __init__(self, init_value=1, min_value=0.0, eta=0.8):
        self.init_value = init_value
        self.min_value = min_value
        self.eta = eta
        self.reset()

    def get_value(self, state:int, action:int):
        state_action_count = self.state_action_counts[state][action]
        assert state_action_count > 0, "State-action count must be positive"
        value = max(self.min_value, 1 / (state_action_count ** self.eta))
        return value

    def step(self, state:int, action:int):
        # assert isinstance(state, int), "State must be an integer"
        # assert isinstance(action, int), "Action must be an integer"
        self.state_action_counts[state][action] += 1

    def reset(self):
        """
        Reset all state-action visit counts.
        """
        # Dictionary to track visit counts for each state
        self.state_action_counts = defaultdict(lambda: defaultdict(lambda: self.init_value))

class CountBasedAlphaScheduler2(Scheduler):
    """
    alpha(s,a)=1/n(s,a)^eta,
    """
    def __init__(self, n_state, n_action, min_value=0.0, eta=0.8):
        self.n_state = n_state
        self.n_action = n_action
        self.min_value = min_value
        self.eta = eta
        self.reset()

    def reset(self):
        self.state_action_counts = np.ones((self.n_state, self.n_action))

    def step(self, state:int, action:int):
        assert isinstance(state, int), "State must be an integer"
        assert isinstance(action, int), "Action must be an integer"
        self.state_action_counts[state,action] += 1

    def get_value(self, state:int, action:int):
        state_action_count = self.state_action_counts[state,action]
        assert state_action_count > 0, "State-action count must be positive"
        value = max(self.min_value, 1 / (state_action_count ** self.eta))
        return value


def softmax(x: np.array, temperature):
    # check if shape of x is (B x N), if (N,), expand it to (1 x N)
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp((x - max_x) / temperature)  # subtract max(x) for numerical stability
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    # print(max_x.shape, e_x.shape, sum_e_x.shape)
    return e_x / sum_e_x
