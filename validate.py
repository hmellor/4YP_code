import os
import sys
import yaml
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.superpixels import setup_superpixels
from ptsemseg.superpixels import convert_to_superpixels
from ptsemseg.superpixels import convert_to_pixels


def iou(gt, pred):
    iou_sum = 0
    for cls in np.unique(pred):
        cls_pred = (pred==cls)
        cls_gt = (gt==cls)
        intersection = (cls_pred & cls_gt).sum()
        union = (cls_pred | cls_gt).sum()
        iou_sum  += intersection / union
    iou = iou_sum / len(np.unique(pred))
    return iou


def validate(cfg, args):

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    if isinstance(cfg['training']['loss']['superpixels'], int):
        use_superpixels = True
        cfg['data']['train_split'] = 'train_super'
        cfg['data']['val_split'] = 'val_super'
        setup_superpixels(100)
    elif cfg['training']['loss']['superpixels'] is not None:
        raise Exception(
            "cfg['training']['loss']['superpixels'] is of the wrong type"
        )
    else:
        use_superpixels = False

    loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        superpixels=cfg['training']['loss']['superpixels'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    sum_iou = 0

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))

    # Load State
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.to(device)

    for i, (images, labels, labels_s, masks) in tqdm(enumerate(valloader)):
        images = images.to(device)
        labels = labels.to(device)
        labels_s = labels_s.to(device)
        masks = masks.to(device)

        outputs = model(images)
        if use_superpixels:
            outputs_s, labels_s, sizes = convert_to_superpixels(
                outputs, labels_s, masks)
            outputs = convert_to_pixels(
                outputs_s, outputs, masks)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()

        image_iou = iou(gt, pred)
        sum_iou += image_iou
    mean_iou = sum_iou / i
    return mean_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )


    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    cfg['training']['loss']['superpixels'] = 10000

    validate(cfg, args)
