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
print("Single image pre-processing time:", time.time()-t)


## Loss function
# Delete tensors to ensure they are being created correctly in loss
del image
del segments

###############################################################################
## Generating scores and target which would be input to loss function
input = torch.rand(2, 21, 375, 500, dtype=torch.float)
target = torch.from_numpy(
        io.imread('2007_000039_encoded.png')
        ).unsqueeze_(0).long()
target = torch.cat((target,target))
names = ('2007_000039','2007_000039')
###############################################################################
t = time.time()

# Extract size data from input and target
n, c, h, w = input.size()
nt, ht, wt = target.size()
# Load the pre-processed segmentation
segments = torch.zeros(n,h,w)
for idx in range(n):
    segments[idx,:,:] = torch.load('{}.pt'.format(names[idx]))
# Initialise superpixel tensors
input_s  = torch.zeros((n*segments.unique().numel(),c))
target_s = torch.zeros(nt*segments.unique().numel())
# Some prints for sanity checks
print("Input shape:\n{}\nTarget shape:\n{}".format(input.shape, target.shape)) 
print("Input super-pixel shape:\n{}\nTarget super-pixel shape:\n{}".format(input_s.shape, target_s.shape))
print("Segments shape:\n{}".format(segments.shape))
# Iterate through all the images
for img in range(n):
    # Iterate through all the clusters
    for idx in range(segments[img,:,:].unique().numel()):
        # First take slices to select image, then use segments slice as mask, then 2D mode for majority class
        target_s[(img*segments.unique().numel())+idx] = target[img,:,:][segments[img,:,:]==idx].mode()[0].mode()[0]
        # Iterate through all the classes
        for k in range(c):
            # Same process as before but also iterating through classes and taking mean because these are scores
            input_s[(img*segments.unique().numel())+idx,k] = input[img,k,:,:][segments[img,:,:]==idx].mean()


print("Input super-pixels:\n{}\nTarget super-pixels:\n{}".format(input_s, target_s))
print("Loss eval time for preprocessed image:", time.time()-t)