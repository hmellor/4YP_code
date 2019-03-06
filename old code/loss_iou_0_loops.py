import torch


def zehan_iou1(input, target, size):
    n_pixels, c = input.size()
    all_classes = torch.arange(0, c, device=input.device)
    gt_classes = target.unique(sorted=True)[1:]
    loss = torch.zeros(1, gt_classes.numel(), device=input.device)
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


def zehan_iou0(input, target, size):
    n_pixels, c = input.size()
    all_classes = torch.arange(0, c, device=input.device)
    gt_classes = target.unique(sorted=True)[1:]
    loss = torch.full((gt_classes.numel(),),
                      - float('inf'), device=input.device)
    for i, gt_class in enumerate(gt_classes):
        theta = input[:, gt_class]
        theta -= input[:, all_classes[all_classes.ne(gt_class)]].logsumexp(1)
        mask_gt = target.eq(gt_class)
        mask_not_gt = target.ne(gt_class)

        n_gt = mask_gt.long().sum()
        # Zehan's algorithm
        # Sort all scores that are supposed to be background and sum them cumulatively
        theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
        theta_hat = theta_tilde.cumsum(0)
        # Iterate through all possible values of U from the min U to all the super-pixels

        U = torch.arange(n_gt, n_pixels + 1, device=input.device)
        indices = theta[mask_gt].repeat(U.numel(), 1).t() >= 1. / U.float()
        sigma = (indices.float().t() * theta[mask_gt]).sum(1)
        I = indices.sum(0)
        #print(U.size(), indices.size(), sigma.size(), I.size())
        sigma -= I.float() / U.float()
        sigma[1:] += theta_hat[U[1:] - n_gt - 1]
        loss[i] = sigma.max()
        loss[i] += 1 - theta[mask_gt].sum()
    return loss.mean()


n_pixels = 300
c_classes = 20

for _ in range(1):
    torch.manual_seed(0)

    theta = torch.randn(n_pixels, c_classes)
    y = torch.randint(0, c_classes, (n_pixels,))
    size = None

    value_1_loop = zehan_iou1(theta, y, size)
    value_0_loop = zehan_iou0(theta, y, size)

    assert (value_0_loop
            - value_1_loop).abs() < 1e-3, "{} {}".format(value_0_loop, value_1_loop)
