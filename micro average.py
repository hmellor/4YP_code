# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:23:02 2018

@author: hejme
"""

import torch
from numpy.random import randint

target = torch.tensor([0,1,2,2], dtype=torch.long)
input  = torch.tensor([[5,2,3],
                       [1,5,3],
                       [0,7,3],
                       [3,1,6]], dtype=torch.float)

#target = torch.from_numpy(randint(0,3,20)).long()
#input  = torch.from_numpy(randint(1,10,(20,4))).float()


print('input:  ', input, '\n')
print('target: ', target, '\n')
# Initialize new variables
pixel_count = target.shape[0]

# have 2 loops
y_star = torch.zeros_like(target)
input1 = input.clone()
for pixel, (column1, gt) in enumerate(zip(input1, target)):
    column1 += 1
    print(column1)
    column1[gt] -= 1
    y_star[pixel] = torch.argmax(column1)
    print('pixel: ', pixel+1, ', scores: ', column1, ', gt: ', gt)

score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
score_y_star = torch.sum(input.gather(1, y_star.unsqueeze(1)))

print('gt score: ', score_y, ', y_star score: ', score_y_star)

loss = torch.sum(torch.ne(y_star, target), dtype=torch.float) + score_y_star - score_y

print('loss: ', loss/pixel_count, '\n')

# remove the inner loop
y_star = torch.zeros_like(target)
input2 = input.clone()
input2 += 1
for pixel, (column2, gt) in enumerate(zip(input2, target)):
    print(column2)
    column2[gt] -= 1 
    y_star[pixel] = torch.argmax(column2)
    print('pixel: ', pixel+1, ', scores: ', column2, ', gt: ', gt)

score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
score_y_star = torch.sum(input.gather(1, y_star.unsqueeze(1)))

print('gt score: ', score_y, ', y_star score: ', score_y_star)

loss = torch.sum(torch.ne(y_star, target), dtype=torch.float) + score_y_star - score_y

print('loss: ', loss/pixel_count, '\n')

# remove the outer loop
y_star_delta = torch.zeros_like(target)
input3 = input.clone()
arange = torch.arange(pixel_count, device=input3.device)
input3 += 1
input3[arange, target] -= 1
print('scores: ', input3)
y_star = torch.argmax(input3, 1)

score_y = torch.sum(input3.gather(1, target.unsqueeze(1)))
score_y_star_delta = torch.sum(input3.gather(1, y_star.unsqueeze(1)))

print('gt score: ', score_y, ', y_star_delta score: ', score_y_star_delta)

loss = score_y_star_delta - score_y

print('loss: ', loss/pixel_count)