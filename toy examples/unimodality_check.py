import torch
import matplotlib.pyplot as plt
from random import uniform
import numpy as np


def zehan_iou(input, target, size):
    n_pixels, c = input.size()
    all_classes = torch.arange(0, c, device=input.device)
    gt_classes = target.unique(sorted=True)
    loss = torch.full((gt_classes.numel(),),
                      - float('inf'), device=input.device)
    for i, gt_class in enumerate(gt_classes):
        theta = input[:, gt_class] - input[:,
                                           all_classes[all_classes.ne(gt_class)]].logsumexp(1)
        mask_gt = target.eq(gt_class)
        mask_not_gt = target.ne(gt_class)

        n_gt = mask_gt.long().sum()
        # Zehan's algorithm
        # Sort all scores that are supposed to be background and sum them cumulatively
        theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
        theta_hat = theta_tilde.cumsum(0)
        # Evaluate loss for all possible values of U from the min U to all the super-pixels
        U = torch.arange(n_gt, n_pixels + 1, device=input.device)
        indices = theta[mask_gt].repeat(U.numel(), 1).t() >= 1. / U.float()
        sigma = (indices.float().t() * theta[mask_gt]).sum(1)
        I = indices.sum(0)
        sigma -= I.float() / U.float()
        sigma[1:] += theta_hat[U[1:] - n_gt - 1]
        if (theta > 0).sum() > 1 and (indices.float().t() * theta[mask_gt]).sum(1).unique().numel() > 1:
            print(theta_hat[U[1:] - n_gt - 1] - (I.float() / U.float())[1:])
  #          print(np.gradient((theta_hat[U[1:] - n_gt - 1] - (I.float() / U.float())[1:]).cpu().detach().numpy()))
#            print(min(theta_tilde.abs()), 1./n_gt.float())
            #print(((indices.float().t() * theta[mask_gt]).sum(1)[0:-1] - (indices.float().t() * theta[mask_gt]).sum(1)[1:]))
            print((indices.float().t() * theta[mask_gt]).sum(1), I, U.numel())
            diff_thet = ((indices.float().t() * theta[mask_gt]).sum(1)[1:] - (indices.float().t() * theta[mask_gt]).sum(1)[0:-1])
            print(diff_thet, 1/U.float())
            diff_iou  = ((I.float() / U.float())[1:] - (I.float() / U.float())[0:-1])
            print("thet diff:", diff_thet,"iou diff", diff_iou)
            print((diff_thet-diff_iou).abs(), "\n")
            #print(((I.float() / U.float())[0:-1] - (I.float() / U.float())[1:]))
            print("\nNumber of sign changes:{}".format(((np.diff(np.sign(np.gradient(((indices.float().t() * theta[mask_gt]).sum(1)).cpu().detach().numpy()))) != 0) * 1).sum()))
           # print(np.gradient((I.float() / U.float()).cpu().detach().numpy()))
     #       print(theta_tilde.abs() < 1./U[1:].float().sum())
          #  assert (theta_tilde.abs() < 1./U[1:].float()).sum() == 0, "\nNumber of sign changes:{}".format(((np.diff(np.sign(np.gradient((I.float() / U.float()).cpu().detach().numpy()))) != 0) * 1).sum())
     #       print("New sample\n")
     #       print("theta", theta)
     #       print("theta[gt] > 1/U:", (indices.float().t() * theta[mask_gt]).sum(1))
     #       print(I.float())
     #       print(np.gradient((I.float() / U.float()).cpu().detach().numpy()))
     #       print((np.diff(np.sign(np.gradient((I.float() / U.float()).cpu().detach().numpy()))) != 0) * 1)
     #       print("unique in theta[gt] > 1/U:", (indices.float().t() * theta[mask_gt]).sum(1).unique())
     #       print("IoU:", I.float() / U.float())
     #       print("theta_hat:", theta_hat[U[1:] - n_gt - 1], "\n")
        loss[i] = sigma.max()
        loss[i] += 1 - theta[mask_gt].sum()
    return loss.mean(), sigma


n_pixels = 30
c_classes = 20
size = None
torch.manual_seed(0)

for _ in range(1000):
    target = torch.randint(0, 2, (n_pixels,)) * uniform(0, c_classes)
    if target.sum() == 0:
        continue
    input = torch.zeros(n_pixels, c_classes).uniform_(-10, 10)
    _, sigma = zehan_iou(input, target, size)
    grads = np.gradient(sigma.cpu().detach().numpy())
    signs = (np.diff(np.sign(grads)) != 0) * 1
    assert signs.sum() <= 1, "Number of sign changes: {}".format(signs.sum())
