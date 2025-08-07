# # detection/utils/utils.py

import torch
import collections
import datetime
import time
import os
import yaml

class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg)


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __str__(self):
        return self.delimiter.join(
            f'{name}: {meter}' for name, meter in self.meters.items())

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        for obj in iterable:
            yield obj
            if i % print_freq == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(iterable) - i)
                eta_str = str(datetime.timedelta(seconds=int(eta)))
                print(f'{header} [{i}/{len(iterable)}] eta: {eta_str}  {self}')
            i += 1


def collate_fn(batch):
    """Custom collate function for handling lists of images and targets."""
    return tuple(zip(*batch))


def save_model(model, path, name="model"):
    """Save model state_dict with timestamp."""
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pth"
    full_path = os.path.join(path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"✅ Model saved to {full_path}")


def get_num_classes(data_yaml_path):
    """Read number of classes from data.yaml."""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return len(data.get('names', []))