import cv2
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import List


def setup_seed(seed):
    cv2.setNumThreads(0)  # Fix dataloader deadlock
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_allocated_gpu_mem():
    return torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0


def setup_tqdm_loader(data_loader, mode, epoch):
    if mode == 'train':
        desc = f'Train'.ljust(10) + f'{epoch}'
    elif mode == 'val':
        split = f'Val'.ljust(10)
        desc = split + f'{epoch}'
    else:
        raise NotImplementedError

    bar_format = '{l_bar}{bar}{r_bar}'
    pbar = tqdm(data_loader, bar_format=bar_format, total=len(data_loader), desc=desc)
    return pbar


class DSMetrics:
    def __init__(self, losses=None, bb_metrics=None, kp_metrics=None):
        self.losses = {} if losses is None else losses
        self.bb_metrics = {} if bb_metrics is None else bb_metrics
        self.kp_metrics = {} if kp_metrics is None else kp_metrics

    def to_dict(self, depth_one=True):
        metrics_list = [self.losses, self.bb_metrics, self.kp_metrics]
        metrics_names = ['loss', 'bb', 'kp']

        if depth_one:
            out_dict = {}
        else:
            out_dict = dict.fromkeys(metrics_names, {})

        for m_dict, m_name in zip(metrics_list, metrics_names):
            for k, v in m_dict.items():
                if depth_one:
                    if m_name == 'loss':
                        key = k
                    else:
                        key = f'{m_name}_{k}'
                    out_dict[key] = v
                else:
                    out_dict[m_name] = v
        return out_dict

    def inline(self):
        m_dict = self.to_dict(depth_one=True)
        out = []
        for k, v in m_dict.items():
            out.append(f'{k}: {v:.5f}')
        return ', '.join(out)

    def extend_with(self, loss=None, bb=None, kp=None):
        if loss is not None:
            self.losses = loss
        if bb is not None:
            self.bb_metrics = bb
        if kp is not None:
            self.kp_metrics = kp

    def get_fitness_score(self, labels: List[str], weights: List[float], reduction='mean'):
        m_dict = self.to_dict(depth_one=True)

        scores = []
        for lbl, w in zip(labels, weights):
            value = m_dict.get(lbl)
            if value is None:
                logger.warning(f'Fitness score: No such metric name: {lbl}')
            else:
                scores.append(value * w)

        if reduction == 'mean':
            score = sum(scores) / (len(scores) + 1e-8)
        elif reduction == 'sum':
            score = sum(scores)
        else:
            raise NotImplementedError(f'Fitness score: Unknown reduction mode: {reduction}. Available '
                                      f'options: [`mean`, `sum`]')
        return score

    def __str__(self):
        lines = []

        if self.losses:
            lines.append('Losses:')
            out = [f'{k}: {v:.5f}' for k, v in self.losses.items()]
            lines.append(', '.join(out))

        if self.bb_metrics:
            lines.append('Detection Metrics:')
            out = [f'{k}: {v:.4f}' for k, v in self.bb_metrics.items()]
            lines.append(', '.join(out))

        if self.kp_metrics:
            lines.append('Keypoint Metrics:')
            out = [f'{k}: {v:.4f}' for k, v in self.kp_metrics.items()]
            lines.append(', '.join(out))

        str_ = '\n'.join(lines)
        return str_


class Metrics:
    def __init__(self, metrics: List[DSMetrics] = None, split_names: List[str] = None):
        self.metrics = metrics if metrics is not None else []
        self.split_names = split_names if split_names is not None else []

    def to_dict(self, depth_one=True):
        out_dict = {}
        for s_name, metrics in zip(self.split_names, self.metrics):
            m_dict = metrics.to_dict(depth_one)
            if depth_one:
                for k, v in m_dict.items():
                    key = f'{s_name}_{k}'
                    out_dict[key] = v
            else:
                out_dict[s_name] = m_dict
        return out_dict

    def inline(self):
        m_dict = self.to_dict(depth_one=True)
        out = []
        for k, v in m_dict.items():
            out.append(f'{k}: {v:.5f}')
        return ', '.join(out)

    def add_split(self, metrics, split_name):
        self.metrics.append(metrics)
        self.split_names.append(split_name)

    def get_fitness_score(self, labels: List[str], weights: List[float], ds_reduction: str = 'mean', metrics_reduction: str = 'mean'):
        scores = {}
        for s_name, metrics in zip(self.split_names, self.metrics):
            scores[s_name] = metrics.get_fitness_score(labels, weights, metrics_reduction)

        score = -1
        scores_ = list(scores.values())
        if ds_reduction == 'mean':
            score = sum(scores_) / (len(scores_) + 1e-8)
        elif ds_reduction == 'sum':
            score = sum(scores_)
        elif ds_reduction.startswith('select_'):
            s_name = ds_reduction[6:]
            score = scores.get(s_name)
            if score is None:
                raise NotImplementedError(f'Fitness Score: Unknown split name in the reduction option: {ds_reduction}')

        name = ' + '.join([f'{float(w):.1f}*{l}' for l, w in zip(labels, weights)])
        name = f'{ds_reduction}({name})'
        return score, name

    def __str__(self):
        lines = []
        for s_name, metrics in zip(self.split_names, self.metrics):
            lines.append(s_name.center(100, '-'))
            lines.append(f'{str(metrics)}')

        str_ = '\n'.join(lines)
        return str_

