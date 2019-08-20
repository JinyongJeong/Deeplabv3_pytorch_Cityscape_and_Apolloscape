# camera-ready

import torch
import torch.nn as nn

import numpy as np

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        [128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        [  0,  0,142],
        [  0,  0, 70],
        [  0, 60,100],
        [  0, 80,100],
        [  0,  0,230],
        [119, 11, 32],
        [81,  0, 81]
        }

    img = np.array(label_to_color)[img]
    return img

def label_img_to_color_apolloscape(img):
    label_to_color= [
        [128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        ]

    img = np.array(label_to_color)[img]

    return img
