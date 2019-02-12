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
    return loss

def zehan_iou(input, target, size):
    n_pixels, c = input.size()
    all_classes = torch.arange(0,c, device = input.device)
    gt_classes = target.unique(sorted=True)[1:]
    loss = torch.zeros(1, gt_classes.numel(), device=input.device)
    for i, gt_class in enumerate(gt_classes):
        theta=input[:,gt_class]-input[:,all_classes[all_classes.ne(gt_class)]].logsumexp(1)
        mask_gt = target.eq(gt_class)
        mask_not_gt = target.ne(gt_class)

        n_gt = mask_gt.long().sum()
        # Zehan's algorithm
        # Sort all scores that are supposed to be background and sum them cumulatively
        theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
        theta_hat = theta_tilde.cumsum(0)
        # Iterate through all possible values of U from the min U to all the super-pixels
        for U in torch.arange(n_gt, n_pixels + 1, device=input.device):
            # Reset I and sigma for the current U
            I = 0
            sigma = 0
            if U > 0:
                # Indices of theta values that would increase max{S+delta}
                indices = theta[mask_gt] >= 1. / float(U)
                # Add these scores to sigma
                sigma = (indices.float() * theta[mask_gt]).sum()
                # Update I with the number of suitable theta values
                I = indices.sum()
                # Add the iou term
                sigma -= float(I) / float(U)

            if U > n_gt:
                sigma += theta_hat[U - n_gt - 1]
            if sigma >= loss[0, i]:
                loss[0, i] = sigma
        loss[0, i] += 1 - theta[mask_gt].sum()
    return loss.mean()

def macro_average(input, target, size=None):

    #raise RuntimeError("this code assumes that the superpixels are computed as an average of scores rather than a sum")

    if size is not None:
        pass
    else:
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
    # Initiualise new variables
    p, _ = input.size()
    delta = torch.ones_like(input)
    arange = torch.arange(p, device=input.device)
    # Calculate delta
    delta[arange, target] -= 1
    unique = torch.unique(target)

    for k in unique:
        delta[target==k,:] /= torch.sum(target==k).float() * unique.size(0)

    input = input/p + delta
    # Evaluate optimal prediction
    pred = torch.argmax(input, 1)
    if size is not None:
        # Evaluate scores for ground truth and prediction
        size_summed = size.sum()
        y_index = (target.float() * size).unsqueeze(1).long()
        pred_index = (pred.float() * size).unsqueeze(1).long()
        score_y = torch.sum(input.gather(1, y_index)) / size_summed
        score_pred_delta = torch.sum(input.gather(1, pred_index)) / size_summed
    else:
        score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
        score_pred_delta = torch.sum(input.gather(1, pred.unsqueeze(1)))
    # Evaluate total loss
    loss = score_pred_delta - score_y
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

    # Initialise new variables
    pixel_count = nt*ht*wt
    pred = torch.zeros_like(target)
    arange = torch.arange(pixel_count, device=input.device)
    # Add delta to input
    input += 1
    input[arange, target] -=1
    # Evaluate optimal prediction
    pred = torch.argmax(input, 1)
    # Evaluate scores for ground truth and prediction
    score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
    score_pred_delta = torch.sum(input.gather(1, pred.unsqueeze(1)))
    # Evaluate total loss
    loss = score_pred_delta - score_y
    return loss/pixel_count

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
