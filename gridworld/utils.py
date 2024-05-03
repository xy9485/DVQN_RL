from collections import defaultdict
import json
import os


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
        self._mg = MetersGroup(
           self.avg_meter_log_path
        )
        
    def dump_episodic_data(self, data):
        with open(self.episodic_log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log(self, key, value, n=1):
        self._mg.log(key, value, n)


    def dump(self, info:dict):
        self._mg.dump(info=info)