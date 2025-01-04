from collections import deque
import copy
import random
import numpy as np


class Agent:
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, temperature_start=None, temperature_end=None, temperature_decay=None, **kwargs):
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        self.temperature = temperature_start
        self.temperature_min = temperature_end
        self.temperature_decay = temperature_decay
        
    # def softmax(x, temperature):
    #     e_x = np.exp((x - np.max(x)) / temperature)  # subtract max(x) for numerical stability
    #     return e_x / e_x.sum(axis=0)
    
    def softmax(self, x: np.array, temperature):
        # check if shape of x is (B x N), if (N,), expand it to (1 x N)
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        max_x = np.max(x, axis=-1, keepdims=True)        
        e_x = np.exp((x - max_x) / temperature)  # subtract max(x) for numerical stability
        sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
        # print(max_x.shape, e_x.shape, sum_e_x.shape)
        return e_x / sum_e_x


    def choose_action(self, state, Q=None, greedy=False):
        if Q is None:
            Q = self.Q
        if greedy:
            action = np.argmax(Q[state, :])
            #if there are multiple actions with the same value, choose randomly
            if np.sum(Q[state, :] == Q[state, action]) > 1:
                action = np.random.choice(np.where(Q[state, :] == Q[state, action])[0])
            action_prob = 1
            info_dict = {
                "action_prob": action_prob,
                "Q": Q[state, action],
            }
            return action, info_dict
        if self.temperature:
            # compute softmax probability given self.Q[state, :]
            prob = self.softmax(Q[state, :], temperature=self.temperature)
            prob = np.squeeze(prob)
            action = np.random.choice(self.n_actions, p=prob)
            action_prob = prob[action]
            info_dict = {
                "action_prob": action_prob,
                "Q": Q[state, action],
            }
            return action, info_dict
        
        eps = self.eps
        if np.random.random() > eps:
            action = np.argmax(Q[state, :])
            #if there are multiple actions with the same value, choose randomly
            if np.sum(Q[state, :] == Q[state, action]) > 1:
                action = np.random.choice(np.where(Q[state, :] == Q[state, action])[0])
                
            action_prob = 1-eps + eps/self.n_actions
        else:
            action = self.action_space.sample()
            action_prob = eps/self.n_actions
            
        info_dict = {
            "action_prob": action_prob,
            "Q": Q[state, action],
        }
        return action, info_dict

    def compute_action_prob(self, state, action, Q, temprature, eps):
        # assert temprature or eps won't be true at the same time
        # assert not (temprature and eps)
        if temprature:
            prob = self.softmax(Q[state, :], temperature=temprature)
            action_prob = prob[np.arange(prob.shape[0]), action]
            return action_prob
        action_mask = action == np.argmax(Q[state, :], axis=-1)
        action_mask = action_mask.astype(float)
        action_prob = (1-eps + eps/self.n_actions) * action_mask + (eps/self.n_actions) * (1-action_mask)
        return action_prob

    def decay_eps(self):
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
    
    def decay_temprature(self):
        self.temperature = self.temperature * self.temperature_decay if self.temperature > self.temperature_min else self.temperature_min
    
    def decay_explore(self):
        if self.temperature:
            self.decay_temprature()
        else:
            self.decay_eps()
            
    # define abstract method
    def learn(self, state, action, reward, state_, done, **kwargs):
        pass

    def simulate_noisy_update(self, noise=0.01):
        # self.Q = np.random.normal(self.Q, noise)
        # sample a noise from a uniform distribution between -noise and noise and add it to self.Q
        self.Q = self.Q + np.random.uniform(-noise, noise, self.Q.shape)




class Q_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay)
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.random_value_init = random_value_init
        self.reset_Q(self.random_value_init)

        
    def learn(self, state, action, reward, state_, done, **kwargs):
        # check if done is a numpy array
        
        Q_error = reward + self.gamma * np.max(self.Q[state_, :]) * (1-done) - self.Q[state, action]
        self.Q[state, action] = self.Q[state, action] + self.alpha * Q_error
        return {
            "Q_error": np.mean(Q_error), 
            "maxQ": np.max(self.Q),
            "Q(s,a)": np.mean(self.Q[state, action]),
            "eps": self.eps
            }

    def reset_Q(self, random):
        if random:
            # random from uniform distribution between a and b
            a, b = -1, 1
            self.Q = np.random.uniform(a, b, (self.n_states, self.n_actions))
        else:
            self.Q = np.zeros((self.n_states, self.n_actions))

class DoubleQ_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay)
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.random_value_init = random_value_init
        self.reset_Q(self.random_value_init)

        
    def learn(self, state, action, reward, state_, done, **kwargs):
        # check if done is a numpy array
        if np.random.random() > 0.5:
            A1 = random.choice(np.where(self.Q[state_, :] == np.max(self.Q[state_, :]))[0])
            Q_error = reward + self.gamma * self.Q_[state_, A1] * (1-done) - self.Q[state, action]
            self.Q[state, action] += self.alpha * Q_error
        else:
            A2 = random.choice(np.where(self.Q_[state_, :] == np.max(self.Q_[state_, :]))[0])
            Q_error = reward + self.gamma * self.Q[state_, A2] * (1-done) - self.Q_[state, action]
            self.Q_[state, action] += self.alpha * Q_error

        return {
            "Q_error": np.mean(Q_error), 
            "maxQ": np.max(self.Q),
            "Q(s,a)": np.mean(self.Q[state, action]),
            "eps": self.eps
            }
    def choose_action(self, state, greedy=False):
        # given state, sum Q and Q_ and choose action with highest value
        Q_sum = (self.Q + self.Q_)/2
        return super().choose_action(state, Q=Q_sum, greedy=greedy)

    def reset_Q(self, random):
        if random:
            # random from uniform distribution between a and b
            a, b = -1, 1
            self.Q = np.random.uniform(a, b, (self.n_states, self.n_actions))
            self.Q_ = np.random.uniform(a, b, (self.n_states, self.n_actions))
        else:
            self.Q = np.zeros((self.n_states, self.n_actions))
            self.Q_ = np.zeros((self.n_states, self.n_actions))

class VQ_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, **kwargs)
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.random_value_init = random_value_init
        self.alpha_v = kwargs["alpha_v"]
        self.alpha_q = kwargs["alpha_q"]
        self.reset_value_functions(random=self.random_value_init)
        self.importance_sampling = kwargs["importance_sampling"]

    def reset_value_functions(self, random):
        if random:
            # random from uniform distribution between a and b
            a, b = -1, 1
            self.V = np.random.uniform(a, b, self.n_states)
            self.Q = np.random.uniform(a, b, (self.n_states, self.n_actions))
            
        else:
            self.V = np.zeros(self.n_states)
            self.Q = np.zeros((self.n_states, self.n_actions))

    def learn(self, state, action, reward, state_, done, prob_action):
        if self.importance_sampling:
            # max_action = np.argmax(self.Q[state, :])
            # bool_mask = action == max_action
            # bool_mask = bool_mask.astype(float)
            # current_action_prob = (1-self.eps + self.eps/self.n_actions) * bool_mask + (self.eps/self.n_actions) * (1-bool_mask)
        
            action_prob_current_pi = self.compute_action_prob(state, action, self.Q, temprature=self.temperature, eps=self.eps)
            #importance sampling ratio
            importance_sampling_ratio = action_prob_current_pi / prob_action
            # get an array where elements are 1 when elements in importance_sampling_ratio at the same indice that are between [0.8,1.2]
            # and 0 otherwise
            
            mask_ = np.logical_and(importance_sampling_ratio > 0.9, importance_sampling_ratio < 1.1)
            mask_ = mask_.astype(float)

            V_target = importance_sampling_ratio * reward + self.gamma * self.V[state_] * (1-done)
            V_error = V_target - self.V[state]
            # V_target = reward + self.gamma * self.V[state_] * (1-done)
            # V_error = importance_sampling_ratio * V_target - self.V[state]
            # or 
            # V_error = importance_sampling_ratio * (V_target - self.V[state])
            V_error = V_error * mask_
            self.V[state] = self.V[state] + self.alpha_v * V_error

            Q_target = reward + self.gamma * self.V[state_] * (1-done)
            Q_error = Q_target - self.Q[state, action]
            # Q_error = Q_error * mask_
            self.Q[state, action] = self.Q[state, action] + self.alpha_q * Q_error
        else:
            V_error = reward + self.gamma * self.V[state_] * (1-done) - self.V[state]
            Q_error = reward + self.gamma * self.V[state_] * (1-done)- self.Q[state, action]
            # action_ = np.array([self.choose_action(state_)[0]], dtype=int)

            self.V[state] = self.V[state] + self.alpha_v * V_error
            # self.Q[state, action] = self.V[state_]
            self.Q[state, action] = self.Q[state, action] + self.alpha_q * Q_error

        return {
            "V_error": np.mean(V_error), 
            "Q_error": np.mean(Q_error), 
            "max_VQ_diff": np.max(np.abs(self.V - np.max(self.Q, axis=1))),
            "maxV": np.max(self.V),
            "maxQ": np.max(self.Q),
            "V(s)": np.mean(self.V[state]),
            # "V(s_)": np.mean(self.V[state_]) if not done else 0,
            "Q(s,a)": np.mean(self.Q[state, action]),
            # "Q(s_,a_)": np.mean(self.Q[state_, action_]) if not done else 0,
            "is_ratio": np.mean(importance_sampling_ratio) if self.importance_sampling else 0, 
            "mask_": np.mean(mask_) if self.importance_sampling else 0,
            "eps": self.eps,
            "temperature": self.temperature if self.temperature else 0
            }

    def learn_syncVQ(self, state, action, reward, state_, done, prob_action):
        # deep copy self.V and self.Q
        self.V_ = copy.deepcopy(self.V)
        self.Q_ = copy.deepcopy(self.Q)

        if self.importance_sampling:
            max_action = np.argmax(self.Q_[state, :])
            bool_mask = action == max_action
            bool_mask = bool_mask.astype(float)
            current_action_prob = (1-self.eps + self.eps/self.n_actions) * bool_mask + (self.eps/self.n_actions) * (1-bool_mask)
            #importance sampling ratio
            importance_sampling_ratio = current_action_prob / prob_action
            # get an array where elements are 1 when elements in importance_sampling_ratio at the same indice that are between [0.8,1.2]
            # and 0 otherwise
            
            # 
            target = reward + self.gamma * self.V_[state_] * (1-done)
            V_error = importance_sampling_ratio * target - self.V_[state]
            Q_error = importance_sampling_ratio * target - self.Q_[state, action]
            mask_ = np.logical_and(importance_sampling_ratio > 0.9, importance_sampling_ratio < 1.1)
            mask_ = mask_.astype(float)
            V_error = V_error * mask_
            Q_error = Q_error * mask_
        else:
            V_error = reward + self.gamma * self.V_[state_] * (1-done) - self.V_[state]
            Q_error = reward + self.gamma * self.V_[state_] * (1-done)- self.Q_[state, action]

        self.V_[state] = self.V_[state] + self.alpha_v * V_error
        self.Q_[state, action] = self.Q_[state, action] + self.alpha_q * Q_error
        return {
            "V_error": np.mean(V_error), 
            "Q_error": np.mean(Q_error), 
            "max_VQ_diff": np.max(np.abs(self.V_ - np.max(self.Q_, axis=1))),
            "maxV": np.max(self.V_),
            "maxQ": np.max(self.Q_),
            "V(s)": np.mean(self.V_[state]),
            "Q(s,a)": np.mean(self.Q_[state, action]),
            "is_ratio": np.mean(importance_sampling_ratio) if self.importance_sampling else 0, 
            "mask_": np.mean(mask_) if self.importance_sampling else 0,
            "bool_mask": np.mean(bool_mask) if self.importance_sampling else 0,
            "eps": self.eps
            }


    def sync_VQ(self):
        self.V = copy.deepcopy(self.V_)
        self.Q = copy.deepcopy(self.Q_)
    
    def simulate_noisy_update(self, noise=0.01):
        # self.Q = np.random.normal(self.Q, noise)
        # self.V = np.random.normal(self.V, noise)
        # sample a noise from a uniform distribution between -noise and noise and add it to self.Q
        self.Q = self.Q + np.random.uniform(-noise, noise, self.Q.shape)
        self.V = self.V + np.random.uniform(-noise, noise, self.V.shape)

class SARSA_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay)
        self.Q = np.zeros((observation_space.n, action_space.n))
        self.random_value_init = random_value_init
        self.reset_Q(self.random_value_init)

    def learn(self, state, action, reward, state_, done, **kwargs):
        action_, _ = self.choose_action(state_)
        Q_error = reward + self.gamma * self.Q[state_, action_] * (1-done) - self.Q[state, action]
        self.Q[state, action] = self.Q[state, action] + self.alpha * Q_error
        return {
            "Q_error": np.mean(Q_error), 
            "maxQ": np.max(self.Q),
            "Q(s,a)": np.mean(self.Q[state, action]),
            "eps": self.eps
            }

    def reset_Q(self, random):
        if random:
            # random from uniform distribution between a and b
            a, b = -1, 1
            self.Q = np.random.uniform(a, b, (self.n_states, self.n_actions))
        else:
            self.Q = np.zeros((self.n_states, self.n_actions))

class ExpectedSARSA_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay)
        self.Q = np.zeros((observation_space.n, action_space.n))
        self.random_value_init = random_value_init
        self.reset_Q(self.random_value_init)

    def learn(self, state, action, reward, state_, done, **kwargs):
        q_values = self.Q[state_, :]
        q_values_sum = np.sum(q_values)
        q_values_max = np.max(q_values)
        q_values_max_count = len(q_values[q_values == q_values_max])
        expected_value_for_max = q_values_max *  ((1-self.eps) / q_values_max_count + self.eps / self.n_actions) * q_values_max_count
        expected_value_for_non_max = (q_values_sum - q_values_max * q_values_max_count) * (self.eps / self.n_actions)
        excepted_value = expected_value_for_max + expected_value_for_non_max
        Q_error = reward + self.gamma * excepted_value * (1-done) - self.Q[state, action]
        self.Q[state, action] = self.Q[state, action] + self.alpha * Q_error
        return {
            "Q_error": np.mean(Q_error), 
            "maxQ": np.max(self.Q),
            "Q(s,a)": np.mean(self.Q[state, action]),
            "eps": self.eps
            }

    def reset_Q(self, random):
        if random:
            # random from uniform distribution between a and b
            a, b = -1, 1
            self.Q = np.random.uniform(a, b, (self.n_states, self.n_actions))
        else:
            self.Q = np.zeros((self.n_states, self.n_actions))


def get_agent(config, env):
    if config['algo_name'] == 'VQ-learning':
        agent = VQ_Agent(action_space=env.action_space, observation_space=env.observation_space, **config)
    elif config['algo_name'] == 'Q-learning':
        agent = Q_Agent(action_space=env.action_space, observation_space=env.observation_space, **config)
    elif config['algo_name'] == 'DoubleQ-learning':
        agent = DoubleQ_Agent(action_space=env.action_space, observation_space=env.observation_space, **config)
    elif config['algo_name'] == 'SARSA':
        agent = SARSA_Agent(action_space=env.action_space, observation_space=env.observation_space, **config)
    elif config['algo_name'] == 'ExpectedSARSA':
        agent = ExpectedSARSA_Agent(action_space=env.action_space, observation_space=env.observation_space, **config)
    else:
        raise ValueError(f"algo_name: {config['algo_name']} not supported")
    return agent

if __name__ == "__main__":
    def softmax(x: np.array, temperature):
        # shape of x (B x N)
        max_x = np.max(x, axis=-1, keepdims=True)        
        e_x = np.exp((x - max_x) / temperature)  # subtract max(x) for numerical stability
        sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
        print(max_x.shape, e_x.shape, sum_e_x.shape)
        return e_x / sum_e_x
    def softmax2(x, temperature):
        e_x = np.exp((x - np.max(x)) / temperature)  # subtract max(x) for numerical stability
        return e_x / e_x.sum(axis=0)
  
    s = np.random.randint(0, 5, 3)
    print(s)
    print(s.shape)
    print(len(s.shape))

    # create a matrix of size 5x3, with random values
    Q = np.random.rand(5,3)
    print(Q)
    print(Q[s, :])

    print(np.argmax(Q[s, :], axis=-1))

    l1 = [1,2,1]
    l2 = [1,2,3]
    print(np.array(l1) == np.array(l2))
    # compare [1,2,1] and [1,2,3] element-wise and return a boolean array