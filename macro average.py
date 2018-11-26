# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:23:02 2018

@author: hejme
"""

import torch
from numpy.random import randint

target = torch.tensor([0,0,2,2], dtype=torch.long)
input  = torch.tensor([[5,2,3],
                       [1,5,3],
                       [0,7,3],
                       [3,1,6]], dtype=torch.float)
p, c = input.size()

#target = torch.from_numpy(randint(0,3,20)).long()
#input  = torch.from_numpy(randint(1,10,(20,4))).float()


print('Input:\n', torch.transpose(input, 0, 1).long(), '\nTarget:\n', target, '\n')
# Initialize new variables
pixel_count = p

# have 1 loop
y_star = torch.zeros_like(target)
input1 = input.clone()

macro = torch.zeros([c])    
prediction = input1.data.max(1)[1]
gt_class = torch.max(target)

loss = 0
for i in range(c):
    print('Class: ', i)
    tar_class = torch.eq(target.float(), i)
    pred_class = torch.eq(prediction.float(), i)
    print('Target:     ', tar_class,
          '\nPrediction: ', pred_class)
    incorrect = torch.ne(pred_class, tar_class)
    print('Incorrect:  ', incorrect, '\n')
    input1[:,i] += incorrect.float()/c
    macro[i] = torch.sum(incorrect.float())/pixel_count

y_star = torch.argmax(input1, 1)
delta = torch.mean(macro)

print('Macro:\n', macro, '\nDelta:\n', delta, '\ny_star:\n', y_star, '\n')

score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
score_y_star = torch.sum(input.gather(1, y_star.unsqueeze(1)))

print('gt score: ', score_y, ', y_star score: ', score_y_star)

loss = score_y_star - score_y + delta

print('loss: ', loss)
