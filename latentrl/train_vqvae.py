""" Training VAE """
import argparse
import sys
import os
import GPUtil

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getAvailable(
    order="random",
    limit=4,
    maxLoad=0.5,
    maxMemory=0.5,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
)
assert len(DEVICE_ID_LIST) > 0, "no availible cuda currently"
print("availible CUDAs:", DEVICE_ID_LIST)
DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
# os.environ["DISPLAY"] = ":199"
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

from os.path import join, exists
from os import mkdir, makedirs, getpid
import time
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

# from models.vae import VAE
# from models.vae2 import VAE
from nn_models.vqvae import VQVAE
from nn_models.vae3 import VAE
from latentrl.common.transforms import transform_dict

from latentrl.common.utils import save_checkpoint

## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from latentrl.common.learning_scheduler import EarlyStopping

# from utils.learning import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from data.loaders import RolloutObservationDataset
from data2 import RolloutObservationDataset, RolloutDatasetNaive


from hparams import VAEHyperParams as hp

parser = argparse.ArgumentParser(description="VQVAE Trainer")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    metavar="N",
    help="number of epochs to train (default: 1000)",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default="/workspace/repos_dev/VQVAE_RL/vae_models",
    help="Directory where results are logged",
)
parser.add_argument(
    "--noreload", action="store_true", help="Best model is not reloaded if specified"
)
parser.add_argument(
    "--nosamples",
    action="store_true",
    help="Does not save samples during training if specified",
)
parser.add_argument(
    "-of", "--output2file", action="store_true", help="print to a file when using nohup"
)


args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

# check vae dir exists, if not, create it
in_channels = 3
embedding_dim = 16
num_embeddings = 64
game_name = "skiing"
vae_version = "vqvae_c3_embedding16x64_3_temp"

autoencoder_dir = join(args.save_dir, game_name, vae_version)
# summary(model.vq_layer, (16,dd 52, 40))
makedirs(autoencoder_dir, exist_ok=True)
makedirs(join(autoencoder_dir, "samples"), exist_ok=True)

if args.output2file:
    sys.stdout = open(f"{autoencoder_dir}/output.txt", "w")
    sys.stderr = sys.stdout
    print("Current PID: ", getpid())
# if not exists(autoencoder_dir):
#     mkdir(autoencoder_dir)
#     mkdir(join(autoencoder_dir, 'samples'))

print("args:", args)
cuda = torch.cuda.is_available()
print("cuda availible:", cuda)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)

transform = transform_dict[game_name]
path_datesets = "/workspace/repos_dev/VQVAE_RL/datasets/ALE/Skiing-v5_1000x1000"
# path_datesets = "/workspace/repos_dev/VQVAE_RL/datasets/Boxing-v0_1000x300"
# path_datesets = "/workspace/repos_dev/VQVAE_RL/datasets/CarRacing-v0"
# dataset_train = RolloutObservationDataset(path_datesets,
#                                           transform, train=True)
# dataset_test = RolloutObservationDataset(path_datesets,
#                                          transform, train=False)
dataset_train = RolloutDatasetNaive(path_datesets, transform, train=True, test_ratio=0.1)
dataset_test = RolloutDatasetNaive(path_datesets, transform, train=False, test_ratio=0.1)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2
)

# reconstruction_path = os.makedirs(f"../reconstruction/{vae_version}/", exist_ok=True)
reconstruction_path = os.path.join(
    "/workspace/repos_dev/VQVAE_RL/reconstruction", game_name, vae_version
)

os.makedirs(
    reconstruction_path,
    exist_ok=True,
)

model = VQVAE(
    in_channels=in_channels,
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    reconstruction_path=reconstruction_path,
).to(device)

print(model)
# summary(model.encoder, (3, 210, 160))
# summary(model.vq_layer, (16, 52, 40))
# summary(model.decoder, (16, 52, 40))

summary(model.encoder, (3, 84, 84))
summary(model.decoder, (16, 21, 21))

optimizer = optim.Adam(model.parameters(), lr=0.0005)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
scheduler = ReduceLROnPlateau(optimizer, "min")  # default setting
earlystopping = EarlyStopping("min", patience=5)

# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x, mu, logsigma):
#     """ VAE loss function """
#     BCE = F.mse_loss(recon_x, x, size_average=False)
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
#     return BCE + KLD

# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.mse_loss(recon_x, x, size_average=False)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD


def train(epoch):
    """One training epoch"""
    model.train()
    # dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device, dtype=torch.float)
        # print(data.shape)
        optimizer.zero_grad()
        recon_batch, input, vqloss = model(data)
        # print(mu.cpu().detach().numpy().shape, logvar.shape)
        loss = model.loss_function(recon_batch, input, vqloss)["loss"]
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )  # to compute average loss for each sample in the same batch

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(train_loader.dataset))
    )


def validate():
    """One test epoch"""
    model.eval()
    print("Epoch Test:")
    # dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device, dtype=torch.float)
            recon_batch, input, vqloss = model(data)
            test_loss += model.loss_function(recon_batch, input, vqloss)["loss"].item()

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


reload_file = join(autoencoder_dir, "best.tar")
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print(
        "Reloading model at epoch {}"
        ", with test error {}".format(state["epoch"], state["precision"])
    )
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    earlystopping.load_state_dict(state["earlystopping"])


cur_best = None

print("starting training")
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = validate()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(autoencoder_dir, "best.tar")
    filename = join(autoencoder_dir, "checkpoint.tar")
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "precision": test_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "earlystopping": earlystopping.state_dict(),
        },
        is_best,
        filename,
        best_filename,
    )

    # if not args.nosamples:
    #     with torch.no_grad():
    #         sample = torch.randn(64, 128).to(device)
    #         sample = model.decoder(sample).cpu()
    #         save_image(sample.view(64, 3, 64, 64),
    #                    join(autoencoder_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break

if args.output2file == 1:
    sys.stdout.close()
