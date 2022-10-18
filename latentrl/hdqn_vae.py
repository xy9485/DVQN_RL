from cmath import e
import copy
import datetime
from json import decoder
import math
from pickle import FALSE
import random
import re
from pprint import pp
import PIL
from PIL import Image
import io

# from this import d
import time
from collections import Counter, deque, namedtuple
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# from unicodedata import decimal

import gym
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import Tensor, nn
from torchsummary import summary
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.colors as colors
from sympy.solvers import solve
from sympy import Symbol

import wandb
from latentrl.common.batch_kmeans import Batch_KMeans
from nn_models import Decoder_MiniGrid
from nn_models import Encoder_MiniGrid
from nn_models import RandomShiftsAug
from nn_models import Decoder
from nn_models import DQN, V_MLP, DQN_Repara
from policies.utils import ReplayMemory, ReplayMemoryWithCluster
from latentrl.common.learning_scheduler import EarlyStopping, ReduceLROnPlateau
from latentrl.common.utils import (
    Dataset_pretrain,
    get_linear_fn,
    linear_schedule,
    polyak_sync,
    soft_sync_params,
    update_learning_rate,
    wandb_log_image,
)


# def make_encoder(encoder_type, n_channel_input, n_channel_output, observation_space, hidden_dims):
#     assert encoder_type in _AVAILABLE_ENCODERS
#     return _AVAILABLE_ENCODERS[encoder_type](
#         n_channel_input, n_channel_output, observation_space, hidden_dims
#     )


# def weight_init(m):
#     """Custom weight init for Conv2D and Linear layers."""
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight.data)
#         m.bias.data.fill_(0.0)
#     elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
#         assert m.weight.size(2) == m.weight.size(3)
#         m.weight.data.fill_(0.0)
#         m.bias.data.fill_(0.0)
#         mid = m.weight.size(2) // 2
#         gain = nn.init.calculate_gain("relu")
#         nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
#     elif isinstance(m, nn.LayerNorm):
#         m.bias.data.zero_()
#         m.weight.data.fill_(1.0)


# _AVAILABLE_ENCODERS = {"pixel": EncoderRes}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = Encoder(
    #     inchannels=4,
    #     linear_out_dim=64,
    #     observation_space=gym.spaces.box.Box(low=0, high=1, shape=(84, 84, 4)),
    #     n_redisual_layers=0,
    # ).to(device)
    # summary(encoder, (4, 84, 84))

    # # repara = ReparameterizeModule(out_dim=32, shape_encoder_fm=encoder.shape_encoder_fm).to(device)
    # # summary(repara, (64, 4, 4))

    # decoder = Decoder(
    #     32,
    #     4,
    #     shape_encoder_fm=encoder.shape_encoder_fm,
    # ).to(device)
    # summary(decoder, (32,))

    encoder = Encoder_MiniGrid(
        inchannels=3,
        linear_out_dim=64,
        observation_space=gym.spaces.box.Box(low=0, high=1, shape=(8, 8, 3)),
        n_redisual_layers=0,
    ).to(device)
    summary(encoder, (3, 8, 8))

    decoder = Decoder_MiniGrid(
        in_dim=64,
        out_channels=3,
        shape_conv_output=encoder.shape_encoder_fm,
    ).to(device)
    summary(decoder, (64,))
