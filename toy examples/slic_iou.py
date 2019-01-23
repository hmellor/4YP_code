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

## Pre processing steop
t = time.time()
# load the image and convert it to a floating point data type
image = img_as_float(io.imread('2007_000039.png'))
target = io.imread('2007_000039_encoded.png')
# Perform SLIC and convert to torch
numSegments = 10
segments = slic(image, n_segments = numSegments, sigma = 5)
segments = torch.from_numpy(segments)
target = torch.from_numpy(target) / 5 # Scaled for more readable prints
# Calculate mean colour for each cluster
mask = torch.zeros_like(segments, dtype=torch.uint8)
super_pixels = torch.zeros(segments.unique().numel())
for idx in range(super_pixels.numel()):
    mask[:,:] = 0
    mask[segments == idx] = 1
    super_pixels[idx] = target[mask==1].float().mean()
# Store ground truth superpixel data for later
torch.save(super_pixels, 'supx')
torch.save(segments, 'segments')
print("Single image pre-processing time:", time.time()-t)

## Loss function
# Delete tensors to ensure they are being created correctly in loss
del mask
del super_pixels
del segments
t = time.time()
super_pixels = torch.load('supx')
segments = torch.load('segments')
mask = torch.zeros_like(segments, dtype=torch.uint8)
super_z = torch.zeros((super_pixels.numel(),5), dtype=torch.float)


z = torch.randint(0,4,(mask.shape[0],mask.shape[1],5))

print("mask shape:\n{}\nz shape:\n{}".format(mask.shape, z.shape)) 
print("target shape:\n{}\nunique in target:\n{}".format(target.shape, target.unique()))
print("super_pixels:\n{}\nsuper_z:\n{}".format(super_pixels.shape, super_z))

for k in range(5):
    z_slice = z.select(2,k)
    print("k:\n{}\nz_slice shape:\n{}".format(k, z_slice.shape))
    for idx in range(super_pixels.numel()):
        mask[:,:] = 0
        mask[segments == idx] = 1
        print("[{},{}]".format(idx, k))
        print(z_slice[mask==1].mean())
        super_z[idx,k] = z_slice[mask==1].mean()


print("super_pixels:\n{}\n super_z:\n{}".format(super_pixels, super_z))
print("Loss eval time for preprocessed image:", time.time()-t)

#for k in image.unique():
#    print(k)

   # print("Segment: {0}".format(idx))
    #print(torch.mean(image[mask]))
#    fig = plt.figure("Segment {idx}")
#    ax = fig.add_subplot(1, 1, 1)
#    ax.imshow((image[:,:,1]*mask.double()).numpy())
#    plt.axis("off")
#    plt.show()
        
#        print(torch.mean(image_wrong[mask]))
#        fig = plt.figure("Segment {idx}")
#        ax = fig.add_subplot(1, 1, 1)
#        ax.imshow((image_wrong*mask.double()).numpy())
#        plt.axis("off")
#        plt.show()