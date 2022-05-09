""" Some data loading utilities """
from bisect import bisect
import os
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np
import glob


class _RolloutDataset(torch.utils.data.Dataset):  # pylint: disable=too-few-public-methods
    def __init__(
        self, data_path, test_ratio=1 / 3, transform=None, buffer_size=200, train=True
    ):  # pylint: disable=too-many-arguments
        self._transform = transform

        self._files = sorted(glob.glob(join(data_path, "rollout_ep_[0-9][0-9][0-9].npz")))

        indices = np.arange(0, len(self._files))
        self.n_trainset = int(len(indices) * (1.0 - test_ratio))
        print("self.n_trainset:", self.n_trainset)

        if train:
            self._files = self._files[: self.n_trainset]
            # buffer_size = self.n_trainset
        else:
            self._files = self._files[self.n_trainset :]
            # buffer_size = len(indices) - self.n_trainset

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size
        # print(self._buffer_size)

    def load_next_buffer(self):
        """Loads next buffer"""
        self._buffer_fnames = self._files[
            self._buffer_index : self._buffer_index + self._buffer_size
        ]
        # print(len(self._buffer_fnames))
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        if self._buffer_index > 0:
            self._buffer_fnames += self._files[: self._buffer_index]

        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(
            total=len(self._buffer_fnames),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}",
        )
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [
                    self._cum_size[-1] + self._length_per_sequence(data["reward"].shape[0])
                ]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        # to have a full sequence, you need self.seq_len + 1 elements, as
        # you must produce both an seq_len obs and seq_len next_obs sequences
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _length_per_sequence(self, data_length):
        pass


class RolloutSequenceDataset(_RolloutDataset):  # pylint: disable=too-few-public-methods
    """Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)

    NOTE: seq_len < rollout_len in moste use cases

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """

    def __init__(
        self, root, seq_len, transform, buffer_size=200, train=True
    ):  # pylint: disable=too-many-arguments
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data["observations"][seq_index : seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data["actions"][seq_index + 1 : seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [
            data[key][seq_index + 1 : seq_index + self._seq_len + 1].astype(np.float32)
            for key in ("rewards", "terminals")
        ]
        # data is given in the form
        # (obs, action, reward, terminal, next_obs)
        return obs, action, reward, terminal, next_obs

    def _length_per_sequence(self, data_length):
        return data_length - self._seq_len


class RolloutObservationDataset(_RolloutDataset):  # pylint: disable=too-few-public-methods
    """Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of images

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """

    def _length_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data["obs"][seq_index])


class RolloutDatasetNaive(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        train=True,
        test_ratio=0.01,
    ):
        self.fpaths = np.array(
            sorted(glob.glob(os.path.join(data_path, "rollout_ep_[0-9][0-9][0-9]*.npz")))
        )
        # self.fpaths = np.array(
        #     sorted(glob.glob(os.path.join(data_path, "rollout_ep_0*.npz")))
        # )
        print("len(self.fpaths):", len(self.fpaths))
        print(os.path.join(data_path, "rollout_ep_[0-9][0-9][0-9]*.npz"))
        # np.random.seed(0)
        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices) * (1.0 - test_ratio))
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if train else self.test_indices
        # import pdb; pdb.set_trace()
        self._transform = transform
        # print(self.indices)

        self.memory = []
        for f in self.fpaths[self.indices]:
            with np.load(f) as data:
                # self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                # print(data['obs'].shape)
                self.memory += list(data["obs"])
        self.memory = np.array(self.memory)
        print(f"len(self.memory): {len(self.memory)}, Train: {train}")

    def __getitem__(self, idx):
        obs = self.memory[idx]
        obs = self._transform(obs)
        # obs = obs.permute(2, 0, 1) # (N, C, H, W)
        # print(obs.shape, obs.shape[1])
        return obs

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    roll = RolloutDatasetNaive("../datasets/CarRacing-v0")
