# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:22:47 2019

@author: hejme
"""

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import torch
import time

## Testing a preprocessed segmentation
image = img_as_float(io.imread('2007_002099.jpg'))
segments = torch.load('2007_002099.pt').numpy()
# show the output of SLIC
fig = plt.figure("Superpixels --  segments")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.show()


## Pre processing step
t = time.time()
# load the image and convert it to a floating point data type
image = img_as_float(io.imread('2007_000039.jpg'))
# Perform SLIC segmentation
numSegments = 10
segments = slic(image, n_segments = numSegments, sigma = 5)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.show()

# Convery to torch and save for later
segments = torch.from_numpy(segments)
torch.save(segments, '2007_000039.pt')
print("\nSingle image pre-processing time: {}\n".format(time.time()-t))


## Loss function
# Delete tensors to ensure they are being created correctly in loss
del image
del segments

###############################################################################
## Generating scores and target which would be input to loss function
input = torch.rand(1, 21, 375, 500, dtype=torch.float)
target = torch.from_numpy(
        io.imread('2007_000039_encoded.png')
        ).unsqueeze_(0).long()
#target = torch.cat((target,target))
names = ('2007_000039',)
###############################################################################
t = time.time()

# Extract size data from input and target
n, c, h, w = input.size()
nt, ht, wt = target.size()
# Load the pre-processed segmentation
segments = torch.zeros(n,h,w)
segments_u = 0
for idx in range(n):
    segments[idx,:,:] = torch.load('{}.pt'.format(names[idx]))
    segments_u += segments[idx,:,:].unique().numel()
    print("Total segments:           {}".format(segments_u))
# Initialise superpixel tensors
input_s  = torch.zeros(segments_u,c)
target_s = torch.zeros(segments_u)
# Some prints for sanity checks
print("Input shape:              {}\nTarget shape:             {}".format(input.shape, target.shape)) 
print("Input super-pixel shape:  {}\nTarget super-pixel shape: {}".format(input_s.shape, target_s.shape))
print("Segments shape:           {}".format(segments.shape))
# Iterate through all the images
for img in range(n):
    # Define variable for number of unique segments for current image
    img_seg_u = segments[img,:,:].unique().numel()
    # Iterate through all the clusters
    for idx in range(img_seg_u):
        # Define mask for cluster idx
        mask = segments[img,:,:]==idx
        # First take slices to select image, then apply mask, then 2D mode for majority class
        target_s[(img*img_seg_u)+idx] = target[img,:,:][mask].mode()[0].mode()[0]
        # Iterate through all the classes
        for k in range(c):
            # Same process as before but also iterating through classes and taking mean because these are scores
            input_s[(img*img_seg_u)+idx,k] = input[img,k,:,:][mask].mean()
print("\nInput target super-pixeling time: {}\n".format(time.time()-t))
t = time.time()
# Calculate the score for the superpixels both being and not being in the class
score_not_class = input_s[:,torch.arange(0,c)[torch.arange(0,c)!=max(target_s).long()]].exp().sum(1).log()
score_class = input_s[:,max(target_s).long()]
theta = score_class - score_not_class
print("Theta:                    {}".format(score_class))

# Zehan's algorithm
h=torch.zeros(1)
# Sort all scores that are supposed to be background and sum them cumulatively
theta_tilde = theta[target_s==0].sort(descending=True)[0]
theta_hat = theta_tilde.cumsum(0)
# Iterate through all possible values of U from the min U to all the super-pixels
for U in torch.arange((target_s!=0).sum(),target_s.numel()).float():
    print("U: {}".format(U))
    # Reset I and sigma for the currenc U
    I=0
    sigma=0
    # For all the superpixels that are the class in the ground truth
    for j in target_s.nonzero():
        # If including the jth super=pixel will increase the max{S+delta}, include it
        if theta[j] >= 1/U:
            # Add the score and increase the intersection
            sigma += theta[j]
            I += 1
    print("I: {}".format(I))
    sigma += theta_hat[U.int()-(target_s!=0).sum().int()]
    print("Theta_hat[U-|y|]: {}".format(theta_hat[U.int()-(target_s!=0).sum().int()]))
    sigma -= I/U
    print("Sigma: {}, h: {}".format(sigma, h))
    if sigma >= h:
        h = sigma
print("Ground truth score: {}".format(theta[target_s!=0]))
h += 1 - theta[target_s!=0].sum()
print("Loss:{}".format(h))
#print("Input super-pixels:\n{}\nTarget super-pixels:\n{}".format(input_s, target_s))
print("\nLoss eval time for super-pixels: {}\n".format(time.time()-t))