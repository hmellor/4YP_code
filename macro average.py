# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:23:02 2018

@author: hejme
"""

import torch

target = torch.tensor([0,1,2,2], dtype=torch.long)
input  = torch.tensor([[5,2,3],
                       [1,5,3],
                       [0,7,3],
                       [3,1,6]], dtype=torch.float)
p, c = input.size()


print('Input:\n', input.transpose(0, 1), '\nTarget:\n', target, '\n')

delta = torch.ones_like(input)
arange = torch.arange(p, device=input.device)
delta[arange, target] -= 1
print(r'1[y_bar[i]!=y[i]]:','\n', delta.transpose(0,1), '\n')
unique = torch.unique(target)
print('Classes in target:', unique, '\nNumber of classes in target:', unique.size(0))
for k in unique:
    print('\nClass:', k)
    print('Pixels affected:      ', (target==k).float())
    print('c({k},y) factored in: ', torch.div((target==k).float() , torch.sum(target==k).float()))
    print('U(y) also factored in:', torch.div((target==k).float() , torch.sum(target==k).float() * unique.size(0)))
    delta[target==k,:] /= torch.sum(target==k).float() * unique.size(0)
    
print('\nDelta:\n', delta.transpose(0,1))

## have 1 loop
#y_star = torch.zeros_like(target)
#
#macro = torch.ones([c])    
#gt_class = torch.max(target)
#
#delta = torch.zeros(1, dtype=torch.float)
#for k in torch.unique(target):
#    delta_k= torch.zeros(1, dtype=delta.dtype)
#    print('Class in GT: ', k)
#    for i in range(p):
#        print('  Pixel: ', i)
#        print('    GT: ', target[i], ', y_bar: ', input[i,k])
#        
#        if torch.eq(target[i], k).float():
#            delta_k += torch.eq(target[i], k).float()
#            print('    GT[i]==Pixel: ', torch.eq(target[i], k).float())
#            print('    Delta_k = ', delta_k)
#        
#    delta += delta_k / torch.sum(torch.eq(target, k), dtype=torch.float)
#delta /= torch.sum(torch.unique(target), dtype=torch.float)
#
#print(delta)



#loss = 0
#for i in range(c):
#    print('Class: ', i)
#    tar_class = torch.eq(target.float(), i)
#    pred_class = torch.eq(prediction.float(), i)
#    print('Target:     ', tar_class,
#          '\nPrediction: ', pred_class)
#    incorrect = torch.ne(pred_class, tar_class)
#    print('Incorrect:  ', incorrect, '\n')
#    input1[:,i] += incorrect.float()/c
#    macro[i] = torch.sum(incorrect.float())/p
#y_star = torch.argmax(input1, 1)
#delta = torch.mean(macro)
#
#print('Macro:\n', macro, '\nDelta:\n', delta, '\ny_star:\n', y_star, '\n')
#
#score_y = torch.sum(input.gather(1, target.unsqueeze(1)))
#score_y_star = torch.sum(input.gather(1, y_star.unsqueeze(1)))
#
#print('gt score: ', score_y, ', y_star score: ', score_y_star)
#
#loss = score_y_star - score_y + delta
#
#print('loss: ', loss)
