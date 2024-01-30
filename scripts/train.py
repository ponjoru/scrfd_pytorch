import torch
import albumentations as A
import albumentations_experimental as AE

from pathlib import Path
from loguru import logger
from torch.cuda.amp.grad_scaler import GradScaler
from albumentations.pytorch import ToTensorV2

from lib.engine import train_one_epoch, evaluate
from lib.evaluator import WiderFaceEvaluator
from lib.utils import setup_seed
from lib.model import init_scrfd_2_5g_model, init_scrfd_500m_model, init_scrfd_10g_model

from datasets.custom_transform import BBSafeScaledRandomCrop
from datasets.widerface import WiderFaceDataset

from scrfd.loss import SCRFDLoss
from scrfd.losses import QualityFocalLoss, DIoULoss, SmoothL1Loss
from scrfd.anchor import AnchorGenerator
from scrfd.atss import ATSSModule


def get_transforms(mode):
    img_size = 640
    mean = [0.5, 0.5, 0.5]
    std = [0.50196, 0.50196, 0.50196]

    bbox_params = A.BboxParams(
        format='pascal_voc',
        min_visibility=0.3,
        label_fields=['bb_classes', 'bb_weights', 'bbox_id']
    )
    keypoint_params = A.KeypointParams(
        format='xy',
        label_fields=['kp_classes', 'kp_weights', 'kp2bb_ids'],
        remove_invisible=False
    )

    if mode == 'train':
        t = A.Compose([
                BBSafeScaledRandomCrop(p=0.4),
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, value=[0, 0, 0], border_mode=0, position='top_left'),
                A.ColorJitter(hue=0.0705, saturation=[0.5, 1.5], contrast=[0.5, 1.5], brightness=0.1254, p=0.3),
                AE.HorizontalFlipSymmetricKeypoints(symmetric_keypoints=[[0, 1], [2, 2], [3, 4]], p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
            keypoint_params=keypoint_params
        )
    else:
        t = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, value=[0, 0, 0], border_mode=0, position='top_left'),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
            keypoint_params=keypoint_params
        )
    return t


def init_criterion(use_kps=True):
    cls_loss = QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)
    bb_loss = DIoULoss(loss_weight=2.0)
    kp_loss = SmoothL1Loss(beta=0.1111111111111111, loss_weight=0.1)
    anchor = AnchorGenerator(
        scales=[1., 2.],
        aspect_ratios=[1.0],
        sizes=[16, 64, 256],
        anchor_strides=[8, 16, 32],
    )
    matcher = ATSSModule(top_k=9)

    criterion = SCRFDLoss(
        cls_loss,
        bb_loss,
        anchor,
        matcher,
        kp_loss,
        num_classes=1,
        strides=[8, 16, 32],
        use_kps=use_kps,
        use_qscore=True
    )
    return criterion


def save_checkpoint(ckpt, save_dir, save_name):
    save_path = f'{save_dir}/{save_name}'
    torch.save(ckpt, save_path)


def train():
    max_epochs = 600
    device = 'cuda:0'
    # device = 'cpu'
    use_kps = True
    nw = 8
    eval_only = True
    setup_seed(42)
    # ds_path = '/Users/igorpopov/Documents/masters/nn_optimization/data/widerface'
    ds_path = '/datasets/face_detection/widerface/widerface'
    lr = 1e-6
    wd = 1e-4
    save_dir = f'/app/runs/exp0'

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    load_from = f'/app/weights/scrfd_500m_kps.pth'
    model = init_scrfd_500m_model(load_from=load_from, device=device)

    # load_from = '/app/weights/scrfd_2.5g_kps.pth'
    # model = init_scrfd_2_5g_model(load_from=load_from, device=device, use_kps=use_kps)

    # model = init_scrfd_10g_model(load_from=load_from, device=device, use_kps=use_kps)
    # load_from = '/app/weights/scrfd_10g_kps.pth'

    criterion = init_criterion(use_kps=use_kps)

    t_transforms = get_transforms('train')
    t_dataset = WiderFaceDataset(ds_path, 'train', min_size=None, transforms=t_transforms, color_layout='RGB')
    t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=8, num_workers=nw, pin_memory=True, shuffle=False, collate_fn=t_dataset.collate_fn)

    v_transforms = get_transforms('val')
    v_dataset = WiderFaceDataset(ds_path, 'val', min_size=None, transforms=v_transforms, color_layout='RGB')
    v_dataloader = torch.utils.data.DataLoader(v_dataset, batch_size=32, num_workers=nw, pin_memory=True, collate_fn=v_dataset.collate_fn)

    p_groups = model.get_param_groups(wd=wd, no_decay_bn_filter_bias=True)
    optimizer = torch.optim.Adam(p_groups, lr=lr)

    lr_scheduler = None

    scaler = GradScaler(enabled=True)

    evaluator = WiderFaceEvaluator(gt_dir=f'{ds_path}/val/gt', iou_thresh=0.5)

    if eval_only:
        v_metrics = evaluate(model, criterion, v_dataloader, evaluator, epoch=0, device=device)
        print(v_metrics)
        exit(0)

    best_score = -1.0
    for epoch in range(max_epochs):
        t_metrics = train_one_epoch(model, criterion, t_dataloader, optimizer, lr_scheduler, scaler, epoch, device)
        v_metrics = evaluate(model, criterion, v_dataloader, evaluator, epoch, device)

        t_metrics = t_metrics.to_dict()
        v_metrics = v_metrics.to_dict()
        score = (v_metrics['bb_easy_AP'] + v_metrics['bb_medium_AP'] + v_metrics['bb_hard_AP']) / 3.0

        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
            'train_metrics': t_metrics,
            'valid_metrics': v_metrics,
            'epoch': epoch,
            'score': score,
            'best_score': best_score,
        }

        save_checkpoint(state_dict, save_dir=save_dir, save_name='last.pt')
        if score >= best_score:
            best_score = score
            save_checkpoint(state_dict, save_dir=save_dir, save_name='best.pt')
        logger.info(v_metrics)


if __name__ == '__main__':
    train()
