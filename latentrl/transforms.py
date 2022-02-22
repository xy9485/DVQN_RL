# import torch
# import torch.nn as nn
# import torch.utils.data
# from torch import optim
# from torch.nn import functional as F
from torchvision import transforms as T
# from torchvision.utils import save_image
# from torchsummary import summary
# from typing import Any, Dict, List, Optional, Tuple, Type, Union

class Crop:
    def __init__(self, vertical_cut=None, horizontal_cut=None, channel_first=False):
        self._vertical_cut = vertical_cut
        self._horizontal_cut = horizontal_cut
        self.channel_first = channel_first

    def __call__(self, sample):
        if self.channel_first:
            return sample[:, :self._vertical_cut, :self._horizontal_cut]
        else:
            return sample[:self._vertical_cut, :self._horizontal_cut, :]

class RandomBlock(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, p=0.5):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.p = p

    def __call__(self, sample):
        image = sample
        if torch.rand(1) < self.p:
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image[top: top + new_h, left: left + new_w] = 0

        return image

class AddGaussianNoise:
    def __init__(self, mean=0, stddev=1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)

    def __repr__(self):
        repr = f"{self.__class__.__name__}(mean={self.mean}, stddev = {self.stddev})"
        return repr


# transform_carracing = T.Compose([
#     T.ToTensor(),   # W,H,C -> C,W,H & divide 255
#     Crop(vertical_cut=84),
#     T.Resize((64,64)),
#     T.RandomHorizontalFlip(p=0.5),
# ])

transform_carracing = T.Compose([
    Crop(vertical_cut=84, channel_first=False),
    T.ToPILImage(),
    T.Resize((64,64)),  # interpolation=InterpolationMode.BILINEAR
    T.RandomHorizontalFlip(p=0.25),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),    # W,H,C -> C,W,H & divide 255
    # AddGaussianNoise(0., 1.),
    # RandoBlock(output_size=16, p=0.5)
    # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_lunarlander = T.Compose([
    T.ToTensor(),
])

transform_dict={
    'car_racing': transform_carracing,
    'lunar_lander': transform_lunarlander
}

if __name__ == '__main__':
    from PIL import Image
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import gym

    import torch
    import torchvision.transforms as T

    plt.rcParams["savefig.bbox"] = 'tight'
    # orig_img = Image.open(Path('assets') / 'astronaut.jpg')
    env = gym.make("CarRacing-v0")
    env.reset()
    for _ in range(70):
        action = env.action_space.sample()
        orig_img, reward, done, _  = env.step(action)
    # orig_img = T.ToPILImage()(orig_img)
    # if you change the seed, make sure that the randomly-applied transforms
    # properly show that the image can be both transformed and *not* transformed!
    torch.manual_seed(0)


    def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [orig_img] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title='Original image')
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()

    # print(transform_carracing(orig_img).shape)

    padded_imgs = [transform_carracing(orig_img).permute(1,2,0) for _ in range(3)]      #try to plot something to test the transforms
    plot(padded_imgs)
    plt.show()
