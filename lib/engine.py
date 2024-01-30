import torch
from loguru import logger

from torchmetrics import MultioutputWrapper, MeanMetric

from lib.evaluator import DSMetrics, Metrics

from typing import Iterable, Callable
from lib.utils import setup_tqdm_loader, get_allocated_gpu_mem


def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, lr_scheduler: Callable, scaler: object,
        epoch: int, device: str
):
    amp_enabled = False
    model.train()
    n_l = criterion.num_loss_items
    loss_avg = MultioutputWrapper(base_metric=MeanMetric(), num_outputs=n_l).to(device)

    pbar = setup_tqdm_loader(data_loader, mode='train', epoch=epoch)
    for i, batch in enumerate(pbar):
        torch.save({'img': batch['image'], 'ann': batch['annotations'], 'meta': batch['meta']}, '/app/train_batch_1.jpg')
        batch.to(device)
        img, targets, meta = batch['image'], batch['annotations'], batch['meta']

        with torch.autocast(device_type=device, enabled=amp_enabled):
            raw_output = model(img)

        with torch.autocast(device_type=device, enabled=amp_enabled):
            loss_dict = criterion(raw_output, targets, meta)

        loss = sum(loss_dict.values())
        loss_items = torch.stack(list(loss_dict.values()))

        if torch.any(torch.isnan(loss_items)):
            logger.warning('Nan Loss encountered')
        else:
            loss_avg.update(loss_items.unsqueeze(0))
            # backward
            scaler.scale(loss).backward()

        # optimize
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if lr_scheduler:
            lr_scheduler.step()

        # tqdm pbar postfix
        avg_losses = {name: f'{value:.4f}' for name, value in zip(loss_dict.keys(), loss_avg.compute())}
        mem = get_allocated_gpu_mem()  # (GB)
        pbar.set_postfix({**avg_losses, 'gpu_mem': f'{mem:.2f}Gb', 'img_size': meta['img1_shape']})

    loss_items = dict(zip(loss_dict.keys(), map(float, loss_avg.compute())))
    train_metrics = DSMetrics(losses=loss_items)
    train_metrics = Metrics(metrics=[train_metrics], split_names=['Train'])
    return train_metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, evaluator: Callable, epoch: int, device: str):
    model.eval()
    criterion.eval()
    n_l = criterion.num_loss_items
    loss_avg = MultioutputWrapper(base_metric=MeanMetric(), num_outputs=n_l).to(device)

    evaluator.reset()
    pbar = setup_tqdm_loader(data_loader, mode='val', epoch=epoch)

    for i, batch in enumerate(pbar):
        batch.to(device)
        img, targets, meta = batch['image'], batch['annotations'], batch['meta']

        raw_output = model(img)

        loss_dict = criterion(raw_output, targets, meta)
        loss_items = torch.stack(list(loss_dict.values()))

        output = model.postprocess(raw_output, iou_thresh=0.45, conf_thresh=0.02)

        evaluator.add_batch(output, batch, meta)

        if torch.any(torch.isnan(loss_items)):
            logger.warning('Nan Loss encountered')
        else:
            loss_avg.update(loss_items.unsqueeze(0))

        # tqdm pbar postfix
        avg_losses = {name: f'{value:.4f}' for name, value in zip(loss_dict.keys(), loss_avg.compute())}
        mem = get_allocated_gpu_mem()  # (GB)
        pbar.set_postfix({**avg_losses, 'gpu_mem': f'{mem:.2f}Gb', 'img_size': meta['img1_shape']})

    metrics = evaluator.compute()
    loss_items = dict(zip(loss_dict.keys(), map(float, loss_avg.compute())))
    metrics.extend_with(loss=loss_items)
    return metrics
