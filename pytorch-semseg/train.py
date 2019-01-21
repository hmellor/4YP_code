import os
import sys
import yaml
import time
import shutil
import torch
import visdom
import random
import argparse
import datetime
import logger
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm
from math import ceil, floor
from collections import MutableMapping

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def train(cfg, writer, logger_old, name):

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)
    running_metrics_train = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)

    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger_old.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger_old.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger_old.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
                )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger_old.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger_old.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    time_meter = averageMeter()

    train_len = t_loader.train_len
    best_iou = -100.0
    i = start_iter
    j = 0
    flag = True

    # Prepare logging
    xp_name = cfg['model']['arch']+'_'+cfg['training']['loss']['name']+'_'+name
    xp=logger.Experiment(xp_name,
                         use_visdom=True, visdom_opts={'server': 'http://localhost',
                         'port': 8098}, time_indexing=False, xlabel='Epoch')
    # log the hyperparameters of the experiment
    xp.log_config(flatten(cfg))
    # create parent metric for training metrics (easier interface)
    xp.ParentWrapper(tag='train', name='parent',
                    children=(xp.AvgMetric(name="loss"),
                              xp.AvgMetric(name='acc'),
                              xp.AvgMetric(name='acccls'),
                              xp.AvgMetric(name='fwavacc'),
                              xp.AvgMetric(name='meaniu')))
    xp.ParentWrapper(tag='val', name='parent',
                    children=(xp.AvgMetric(name="loss"),
                              xp.AvgMetric(name='acc'),
                              xp.AvgMetric(name='acccls'),
                              xp.AvgMetric(name='fwavacc'),
                              xp.AvgMetric(name='meaniu')))
    best_loss = xp.BestMetric(tag='val-best', name='loss', mode='min')
    best_acc = xp.BestMetric(tag='val-best', name='acc')
    best_acccls = xp.BestMetric(tag='val-best', name='acccls')
    best_fwavacc = xp.BestMetric(tag='val-best', name='fwavacc')
    best_meaniu = xp.BestMetric(tag='val-best', name='meaniu')

    xp.plotter.set_win_opts(name="loss", opts={'title': 'Loss'})
    xp.plotter.set_win_opts(name="acc", opts={'title': 'Micro-Average'})
    xp.plotter.set_win_opts(name="acccls", opts={'title': 'Macro-Average'})
    xp.plotter.set_win_opts(name="fwavacc", opts={'title': 'FreqW Accuracy'})
    xp.plotter.set_win_opts(name="meaniu", opts={'title': 'Mean IoU'})

    it_per_step = cfg['training']['acc_batch_size']
    eff_batch_size = cfg['training']['batch_size'] * it_per_step
    while i <= train_len*(cfg['training']['epochs']) and flag:
        for (images, labels) in trainloader:
            i += 1
            j += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(input=outputs, target=labels)

            # accumulate train metrics during train
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()
            running_metrics_train.update(gt, pred)
            train_loss_meter.update(loss.item())

            # accumulate gradients based on the accumulation batch size
            if i % it_per_step == 1 or it_per_step == 1:
                optimizer.zero_grad()

            grad_rescaling = torch.tensor(1. / it_per_step).type_as(loss)
            loss.backward(grad_rescaling)
            if (i + 1) % it_per_step == 1 or it_per_step == 1:
                optimizer.step()
                optimizer.zero_grad()

            time_meter.update(time.time() - start_ts)
            # training logs
            if (j + 1) % (cfg['training']['print_interval']*it_per_step) == 0:
                fmt_str = "Epoch [{}/{}] Iter [{}/{:d}] Loss: {:.4f}  Time/Image: {:.4f}"
                total_iter = int(train_len / eff_batch_size)
                total_epoch = int(cfg['training']['epochs'])
                current_epoch = ceil((i + 1) / train_len)
                current_iter  = int((j + 1 )/ it_per_step)
                print_str = fmt_str.format(current_epoch,
                                           total_epoch,
                                           current_iter,
                                           total_iter,
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                logger_old.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                time_meter.reset()
            # end of epoch evaluation
            if (i + 1) % train_len == 0 or \
               (i + 1) == train_len*(cfg['training']['epochs']):
                optimizer.step()
                optimizer.zero_grad()
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)
                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                writer.add_scalar('loss/train_loss', train_loss_meter.avg, i+1)
                logger_old.info("Epoch %d Val Loss: %.4f" % (int((i + 1)/train_len), val_loss_meter.avg))
                logger_old.info("Epoch %d Train Loss: %.4f" % (int((i + 1)/train_len), train_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                print("Validation metrics:")
                for k, v in score.items():
                    print(k, v)
                    logger_old.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/{}'.format(k), v, i+1)

                for k, v in class_iou.items():
                    logger_old.info('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                xp.Parent_Val.update(loss = val_loss_meter.avg,
                                    acc     = score['Overall Acc: \t'],
                                    acccls = score['Mean Acc : \t'],
                                    fwavacc = score['FreqW Acc : \t'],
                                    meaniu = score['Mean IoU : \t'])

                score, class_iou = running_metrics_train.get_scores()
                print("Training metrics:")
                for k, v in score.items():
                    print(k, v)
                    logger_old.info('{}: {}'.format(k, v))
                    writer.add_scalar('train_metrics/{}'.format(k), v, i+1)

                for k, v in class_iou.items():
                    logger_old.info('{}: {}'.format(k, v))
                    writer.add_scalar('train_metrics/cls_{}'.format(k), v, i+1)

                xp.Parent_Train.update(loss = train_loss_meter.avg,
                                    acc     = score['Overall Acc: \t'],
                                    acccls = score['Mean Acc : \t'],
                                    fwavacc = score['FreqW Acc : \t'],
                                    meaniu = score['Mean IoU : \t'])

                xp.Parent_Val.log_and_reset()
                xp.Parent_Train.log_and_reset()
                best_loss.update(xp.loss_val).log()
                best_acc.update(xp.acc_val).log()
                best_acccls.update(xp.acccls_val).log()
                best_fwavacc.update(xp.fwavacc_val).log()
                best_meaniu.update(xp.meaniu_val).log()

                visdir = os.path.join(
                    'runs',
                    "{}_{}".format(cfg['model']['arch'],cfg['training']['loss']['name']),
                    name,
                    'plots.json'
                )
                xp.to_json(visdir)

                val_loss_meter.reset()
                train_loss_meter.reset()
                running_metrics_val.reset()
                running_metrics_train.reset()
                j = 0

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)

            if (i + 1) == train_len*(cfg['training']['epochs']):
                flag = False
                break

if __name__ == "__main__":
    run_id = random.randint(1, 100000)
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use"
    )
    parser.add_argument(
        "--name",
        nargs="?",
        type=str,
        default=str(run_id),
        help="Name of the experiment"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    logdir = os.path.join(
        'runs',
        cfg['model']['arch'] + '_' + cfg['training']['loss']['name'],
        args.name
    )
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger_old = get_logger(logdir)
    logger_old.info('Let the games begin')

    train(cfg, writer, logger_old, args.name)
