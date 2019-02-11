import itertools
import torch

n_pixels = 10


def loss_zehan(theta, y):

    mask_gt = y.ne(0)
    mask_not_gt = y.eq(0)
    
    n_gt = mask_gt.sum()

    # Zehan's algorithm
    loss = torch.zeros(1)
    # Sort all scores that are supposed to be background and sum them cumulatively
    theta_tilde = theta[mask_not_gt].sort(descending=True)[0]
    theta_hat = theta_tilde.cumsum(0)

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
    return loss


def loss_brute_force_search(theta, y):
    mask_gt = y.ne(0)
    loss = -float('inf')

    for ybar in itertools.product([0, 1], repeat=n_pixels):
        ybar = torch.tensor(ybar)
        mask = ybar.eq(1)
        I = torch.min(ybar, mask_gt.long()).sum()
        U = torch.max(ybar, mask_gt.long()).sum()
        element_loss = theta[mask].sum() + 1 - float(I) / float(U)
        if element_loss > loss:
            loss = element_loss
            ymax = ybar

    loss -= theta[mask_gt].sum()

    return loss

for _ in range(1):
    theta = torch.randn(n_pixels)
    y = torch.randint(0, 2, (n_pixels,))
    if y.sum() == 0 or y.sum() == n_pixels:
        continue

    value_zehan = loss_zehan(theta, y)
    value_brute_force = loss_brute_force_search(theta, y)
    assert (value_zehan - value_brute_force).abs() < 1e-3, "{} {}".format(value_brute_force, value_zehan)
