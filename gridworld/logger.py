# from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
# import shutil
# import torch
# import torchvision
# import numpy as np
# from termcolor import colored



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
            # if key.startswith('train'):
            #     key = key[len('train') + 1:]
            # else:
            #     key = key[len('eval') + 1:]
            # key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def dump(self, step):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_paths:dict, use_tb=False):
        self._log_paths = log_paths
        self.episodic_log_path = log_paths["episodic"]
        self.avg_meter_log_path = log_paths["avg_meter"]
        # if use_tb:
        #     tb_dir = os.path.join(log_dir, 'tb')
        #     if os.path.exists(tb_dir):
        #         shutil.rmtree(tb_dir)
        #     self._sw = SummaryWriter(tb_dir)
        # else:
        #     self._sw = None
        self._mg = MetersGroup(
           self.avg_meter_log_path
        )
        # self._train_mg = MetersGroup(
        #     os.path.join(log_dir, 'train.log')
        # )
        # self._eval_mg = MetersGroup(
        #     os.path.join(log_dir, 'eval.log')
        # )

    # def _try_sw_log(self, key, value, step):
    #     if self._sw is not None:
    #         self._sw.add_scalar(key, value, step)

    # def _try_sw_log_image(self, key, image, step):
    #     if self._sw is not None:
    #         assert image.dim() == 3
    #         grid = torchvision.utils.make_grid(image.unsqueeze(1))
    #         self._sw.add_image(key, grid, step)

    # def _try_sw_log_video(self, key, frames, step):
    #     if self._sw is not None:
    #         frames = torch.from_numpy(np.array(frames))
    #         frames = frames.unsqueeze(0)
    #         self._sw.add_video(key, frames, step, fps=30)

    # def _try_sw_log_histogram(self, key, histogram, step):
    #     if self._sw is not None:
    #         self._sw.add_histogram(key, histogram, step)
        
    def dump_episodic_data(self, data:dict):
        with open(self.episodic_log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log(self, key, value, n=1):
        # assert key.startswith('train') or key.startswith('eval')
        # if type(value) == torch.Tensor:
        #     value = value.item()
        # self._try_sw_log(key, value / n, step)
        # mg = self._train_mg if key.startswith('train') else self._eval_mg
        # mg.log(key, value, n)
        self._mg.log(key, value, n)

    # def log_param(self, key, param, step):
    #     self.log_histogram(key + '_w', param.weight.data, step)
    #     if hasattr(param.weight, 'grad') and param.weight.grad is not None:
    #         self.log_histogram(key + '_w_g', param.weight.grad.data, step)
    #     if hasattr(param, 'bias'):
    #         self.log_histogram(key + '_b', param.bias.data, step)
    #         if hasattr(param.bias, 'grad') and param.bias.grad is not None:
    #             self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    # def log_image(self, key, image, step):
    #     assert key.startswith('train') or key.startswith('eval')
    #     self._try_sw_log_image(key, image, step)

    # def log_video(self, key, frames, step):
    #     assert key.startswith('train') or key.startswith('eval')
    #     self._try_sw_log_video(key, frames, step)

    # def log_histogram(self, key, histogram, step):
    #     assert key.startswith('train') or key.startswith('eval')
    #     self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        # self._train_mg.dump(step, 'train')
        # self._eval_mg.dump(step, 'eval')
        self._mg.dump(step)