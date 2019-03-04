import torch
import torch.nn.functional as F


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
    loss = F.cross_entropy(
        input,
        target,
        weight=weight,
        size_average=size_average,
        ignore_index=250
    )
    return loss


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
        # print(U.size(), indices.size(), sigma.size(), I.size())
        sigma -= I.float() / U.float()
        sigma[1:] += theta_hat[U[1:] - n_gt - 1]
        loss[i] = sigma.max()
        loss[i] += 1 - theta[mask_gt].sum()
    return loss.mean()


def macro_average(input, target, size=None):
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
        delta[target == k,
              :] /= torch.sum(target == k).float() * unique.size(0)

    input = input / p + delta
    # Evaluate optimal prediction
    pred = torch.argmax(input, 1)
    if size is not None:
        # Evaluate scores for ground truth and prediction
        size_summed = size.sum()
        score_y = torch.sum(input.gather(
            1, target.unsqueeze(1)) * size) / size_summed
        score_pred_delta = torch.sum(input.gather(
            1, pred.unsqueeze(1)) * size) / size_summed
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
    pixel_count = nt * ht * wt
    pred = torch.zeros_like(target)
    arange = torch.arange(pixel_count, device=input.device)
    # Add delta to input
    input += 1
    input[arange, target] -= 1
    # Evaluate optimal prediction
    pred = torch.argmax(input, 1)
    # Evaluate scores for ground truth and prediction
    score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
    score_pred_delta = torch.sum(input.gather(1, pred.unsqueeze(1)))
    # Evaluate total loss
    loss = score_pred_delta - score_y
    return loss / pixel_count


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(
            scale * torch.ones(n_inp), torch.arange(n_inp))

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
