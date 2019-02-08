# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:57:59 2019

@author: hejme
"""

import torch
torch.manual_seed(3)
batch_size = 1
source_s = torch.randint(0,10,(batch_size,3,4,5), dtype=torch.float)
source_i = torch.tensor([[0,0,1,1,3],[0,1,1,2,3],[1,1,2,2,3],[1,1,2,2,3]]).unsqueeze(0).repeat(batch_size,1,1)
s = source_s.clone().squeeze()
I = source_i.clone().squeeze()
Q = I.unique().numel()

counter = torch.ones_like(I)
size = torch.zeros(Q)
size.put_(I,counter.float(),True)
print(size)

c = s.size()[0]
print("scores:\n", s)
print("I:\n", I)
print("num superpixels: ", Q)
print("num classes: ", c)
s = s.view(c, -1)
I = I.view(1,-1).repeat(c,1)
arange = torch.arange(start=1,end=c)
I[arange,:] += Q*arange.view(-1,1)
print("I:\n", I)
print("s size: ", s.size())
print("I size: ", I.size())
t = torch.zeros(Q,c)
t = t.put_(I,s,True).view(c,Q).t()
print("superpixel version of s:\n", t, "\n\n")

"""
s = source_s.clone()
I = torch.tensor([[0,0,1,1],[0,1,1,2],[1,1,2,2]])
print("scores:\n", s)
print("I:\n", I)
t = torch.zeros(Q,c)
classes = torch.arange(c)
for idx in range(Q):
            # Define mask for cluster idx

            for k in range(c):
                segment_mask = I==idx
            # First take slices to select image, then apply mask, then sum scores
                t[idx,k] = s[k,segment_mask].sum()
print("superpixel version of s:\n", t)
"""
