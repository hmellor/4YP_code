import torch
from random import uniform
import numpy as np


def zehan_iou(input, target, size):
    n_pixels, c = input.size()
    all_classes = torch.arange(0, c, device=input.device)
    gt_classes = target.unique(sorted=True)
    loss = torch.full((gt_classes.numel(),),
                      - float('inf'), device=input.device)
    for i, gt_class in enumerate(gt_classes):
        theta = input[:, gt_class].clone()
        theta -= input[:, all_classes[all_classes.ne(gt_class)]].logsumexp(1)
        mask_gt = target.eq(gt_class)
        mask_not_gt = target.ne(gt_class)

        n_gt = mask_gt.long().sum()
        # Zehan's algorithm
        # Sort all scores that are supposed to be background and sum them
        # cumulatively.
        theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
        theta_hat = theta_tilde.cumsum(0)
        # Evaluate loss for all possible values of U from the min U to all the
        # super-pixels.
        U = torch.arange(n_gt, n_pixels + 1, device=input.device)
        indices = theta[mask_gt].repeat(U.numel(), 1).t() >= 1. / U.float()
        sigma = (indices.float().t() * theta[mask_gt]).sum(1)
        I = indices.sum(0)
        sigma -= I.float() / U.float()
        sigma[1:] += theta_hat[U[1:] - n_gt - 1]
        # Check that sigma is unimodal
        grads = np.diff(sigma.cpu().detach().numpy())
        signs = (np.diff(np.sign(grads[grads != 0])) != 0) * 1
        assert signs.sum() <= 1, 'sparse sigma is not unimodal'
        loss[i] = sigma.max()
        loss[i] += 1 - theta[mask_gt].sum()
    return loss.mean()


n_pixels = 2000
c_classes = 20
size = None
torch.manual_seed(0)

for _ in range(10000):
    target = torch.randint(0, 2, (n_pixels,)) * uniform(0, c_classes)
    if target.sum() == 0:
        continue
    input = torch.zeros(n_pixels, c_classes).uniform_(-10, 10)
    _ = zehan_iou(input, target, size)
