import torch

n_pixels = 10
c_classes = 3


def loss_zehan(theta_, y):
    gt_classes = y.unique(sorted=True)[1:]
    print(theta_)
    print(gt_classes)
    print(y)
    
    for i, k in enumerate(gt_classes):
        theta = theta_[:,i]
        print(theta)
        
        mask_gt = y.eq(k)
        print(mask_gt)
        mask_not_gt = y.ne(k)
        print(mask_not_gt)

        n_gt = mask_gt.sum()
    
        # Zehan's algorithm
        loss = torch.zeros(1)
        # Sort all scores that are supposed to be background and sum them cumulatively
        theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
        theta_hat = theta_tilde.cumsum(0)
        print(theta_hat)
        # Iterate through all possible values of U from the min U to all the super-pixels
        for U in torch.arange(n_gt, n_pixels + 1):
            # Reset I and sigma for the currenc U
            I = 0
            sigma = 0
            # Indices of theta values that would increase max{S+delta}
            indices = theta[mask_gt] >= 1. / float(U)
            # Add these scores to sigma
            sigma = (indices.float() * theta[mask_gt]).sum()
            # Update I with the number of suitable theta values
            I = indices.sum()
    
            if U > n_gt:
                sigma += theta_hat[U - n_gt - 1]
            sigma -= float(I) / float(U)
            if sigma >= loss:
                loss = sigma
        loss += 1 - theta[mask_gt].sum()
        print(loss)
    return 1

for _ in range(1):
    theta = torch.randn(n_pixels,c_classes-1)
    y = torch.randint(0, c_classes, (n_pixels,))
    if y.sum() == 0 or y.sum() == n_pixels:
        continue

    value_zehan = loss_zehan(theta, y)
