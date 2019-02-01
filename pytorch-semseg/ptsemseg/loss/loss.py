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

def zehan_iou(input, target):
#    t=time.time()
    n_pixels, c = input.size()
    all_classes = torch.arange(0,c)
    gt_class = target.max().long()
    theta=input[:,gt_class]-input[:,all_classes[all_classes.ne(gt_class)]].exp().sum(1).log()
    mask_gt = target.ne(0)
    mask_not_gt = target.eq(0)

    n_gt = mask_gt.long().sum()
    # Zehan's algorithm
    loss = torch.zeros(1, device=input.device)
    # Sort all scores that are supposed to be background and sum them cumulatively
    theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
    theta_hat = theta_tilde.cumsum(0)
#    print("Time to initialise all variables:", time.time()-t)
#    t1=time.time()
    # Iterate through all possible values of U from the min U to all the super-pixels
    for U in torch.arange(n_gt, n_pixels + 1, device=input.device):
        # Reset I and sigma for the current U
        I = 0
        sigma = 0
        # For all the superpixels that are the class in the ground truth
#        t=time.time()
        for theta_j in theta[mask_gt]:
            # If including the jth super=pixel will increase the max{S+delta}, include it
            if theta_j >= 1. / float(U):
                # Add the score and increase the intersection
                sigma += theta_j
                I += 1
#        print("Time for inner loop:", time.time()-t)
        if U > n_gt:
            sigma += theta_hat[U - n_gt - 1]
        if U > 0:
            sigma -= float(I) / float(U)
        if sigma >= loss:
            loss = sigma
#    print("Time for outer loop:", time.time()-t1)
    loss += 1 - theta[mask_gt].sum()
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
    # Add delta to input
    input = input/p + delta
    # Evaluate optimal prediction
    pred = torch.argmax(input, 1)
    # Evaluate scores for ground truth and prediction
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
