import copy
import random
import numpy as np


class Agent:
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_decay):
        self.action_space = action_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        
    def softmax(x, temperature):
        e_x = np.exp((x - np.max(x)) / temperature)  # subtract max(x) for numerical stability
        return e_x / e_x.sum(axis=0)

    def choose_action(self, state, softmax=False):
        if softmax:
            # compute softmax probability given self.Q[state, :]
            prob = self.softmax(self.Q[state, :], temperature=1.0)
            action = np.random.choice(self.n_actions, p=prob)
            return action
        if np.random.random() > self.eps:
            action = np.argmax(self.Q[state, :])
            #if there are multiple actions with the same value, choose randomly
            if np.sum(self.Q[state, :] == self.Q[state, action]) > 1:
                action = np.random.choice(np.where(self.Q[state, :] == self.Q[state, action])[0])
                
            action_prob = 1-self.eps + self.eps/self.n_actions
        else:
            action = self.action_space.sample()
            action_prob = self.eps/self.n_actions
            
        self.decay_eps()
        return action, action_prob

    def decay_eps(self):
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
            
    # define abstract method
    def learn(self, state, action, reward, state_):
        pass





class Q_Agent(Agent):
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec)
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.random_value_init = random_value_init
        self.reset_Q(self.random_value_init)

        
    def learn(self, state, action, reward, state_, done, prob_action):
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
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec)
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_states = observation_space.n
        self.n_actions = action_space.n
        self.random_value_init = random_value_init
        self.reset_Q(self.random_value_init)

        
    def learn(self, state, action, reward, state_, done, prob_action):
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
    def choose_action(self, state, softmax=False):
        # given state, sum Q and Q_ and choose action with highest value
        Q_sum = (self.Q[state, :] + self.Q_[state, :])/2
        if softmax:
            # compute softmax probability given self.Q[state, :]
            prob = self.softmax(Q_sum, temperature=1.0)
            action = np.random.choice(self.n_actions, p=prob)
            return action
        if np.random.random() > self.eps:
            action = np.argmax(Q_sum)
            #if there are multiple actions with the same value, choose randomly
            if np.sum(Q_sum == Q_sum[action]) > 1:
                action = np.random.choice(np.where(Q_sum == Q_sum[action])[0])
            
            action_prob = 1-self.eps + self.eps/self.n_actions
        else:
            action = self.action_space.sample()
            action_prob = self.eps/self.n_actions
        self.decay_eps()
        return action, action_prob

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
    def __init__(self, alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec, random_value_init, **kwargs):
        super().__init__(alpha, gamma, action_space, observation_space, eps_start, eps_end, eps_dec)
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
            max_action = np.argmax(self.Q[state, :])
            bool_mask = action == max_action
            bool_mask = bool_mask.astype(float)
            current_action_prob = (1-self.eps + self.eps/self.n_actions) * bool_mask + (self.eps/self.n_actions) * (1-bool_mask)
            #importance sampling ratio
            importance_sampling_ratio = current_action_prob / prob_action
            # get an array where elements are 1 when elements in importance_sampling_ratio at the same indice that are between [0.8,1.2]
            # and 0 otherwise
            
            # 
            target = reward + self.gamma * self.V[state_] * (1-done)
            V_error = importance_sampling_ratio * target - self.V[state]
            Q_error = importance_sampling_ratio * target - self.Q[state, action]
            mask_ = np.logical_and(importance_sampling_ratio > 0.9, importance_sampling_ratio < 1.1)
            mask_ = mask_.astype(float)
            V_error = V_error * mask_
            Q_error = Q_error * mask_
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
            "bool_mask": np.mean(bool_mask) if self.importance_sampling else 0,
            "eps": self.eps
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

