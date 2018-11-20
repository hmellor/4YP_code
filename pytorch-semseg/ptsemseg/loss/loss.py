import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        print('resizing, prediction too large')
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        print('resizing, prediction too small')
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
#    print('input: ', input.size() ,', target: ', target.size())
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    print(loss.type())
    return loss

def macro_average(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        print('resizing, prediction too large')
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        print('resizing, prediction too small')
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    prediction = input.data.max(1)[1]
    prediction = prediction.view(-1)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    gt_class = torch.max(target)
    total = prediction.numel()
    print(torch.min(target), torch.max(target))
    print(torch.min(prediction), torch.max(prediction))
    print('input: ', input.size(), ', target: ', target.size(), ', prediction: ', prediction.size())

    loss = 0
    for i in range(c):
        tar_class = torch.eq(target.float(), i)
        pred_class = torch.eq(prediction.float(), i)
        incorrect = torch.ne(pred_class, tar_class)
        incorrect_num = torch.sum(incorrect)
        incorrect_frac = incorrect_num.float() / total
        loss += incorrect_frac
    return loss

def micro_average(input, target):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        print('resizing, prediction too large')
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        print('resizing, prediction too small')
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    # Reshape to nxc and nx1 respectively
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)

    # Initialize new variables
    pixel_count = nt*ht*wt
    y_star = torch.zeros_like(target)
    t = time.time()

    input += 1
    for pixel, (column, gt) in enumerate(zip(input, target)):

        if pixel % 1000 == 0:
            print(time.time()-t, 's')
            print(pixel)
            t = time.time()

        column[gt] -= 1
        y_star[pixel] = torch.argmax(column)

    score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
    score_y_star = torch.sum(input.gather(1, y_star.unsqueeze(1)))

    loss = torch.sum(torch.ne(y_star, target), dtype=torch.float)/pixel_count + score_y_star - score_y
    print(loss.type())
    return loss

def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
