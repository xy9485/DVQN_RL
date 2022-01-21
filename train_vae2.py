""" Training VAE """
import argparse
import sys
from os.path import join, exists
from os import mkdir, makedirs,getpid

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary
from models.vae import VAE

from utils.misc import save_checkpoint
# from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
# from data.loaders import RolloutObservationDataset
from data2 import RolloutObservationDataset

from hparams import VAEHyperParams as hp

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, default='./logdir', help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')
parser.add_argument('-of', '--output2file', action='store_true',
                    help='print to a file when using nohup')


args = parser.parse_args()

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae_buffersize200')
makedirs(vae_dir, exist_ok=True)
makedirs(join(vae_dir, 'samples'), exist_ok=True)

if args.output2file:
    sys.stdout = open(f"{vae_dir}/output.txt", 'w')
    sys.stderr = sys.stdout
    print("Current PID: ",getpid())
# if not exists(vae_dir):
#     mkdir(vae_dir)
#     mkdir(join(vae_dir, 'samples'))

print("args:",args)
cuda = torch.cuda.is_available()
print("cuda availible:", cuda)

torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")


# transform_train = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((RED_SIZE, RED_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((RED_SIZE, RED_SIZE)),
#     transforms.ToTensor(),
# ])

transform_easy = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset_train = RolloutObservationDataset('datasets',
                                          transform_easy, train=True)
dataset_test = RolloutObservationDataset('datasets',
                                         transform_easy, train=False)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)


model = VAE(3, hp.vsize).to(device)
# print(model)
summary(model,(3,64,64))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

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

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device,dtype=torch.float)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))   # to compute average loss for each sample in the same batch

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    print("Epoch Test:")
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device, dtype=torch.float)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss



reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

print("starting training")
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(64, hp.vsize).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, 64, 64),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break

if args.output2file == 1:
    sys.stdout.close()