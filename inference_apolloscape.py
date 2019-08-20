# camera-ready

import sys
import os
default_path = os.path.dirname(os.path.abspath(__file__))

from datasets_apolloscape import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append(os.path.join(default_path,'model'))
from deeplabv3_apolloscape import DeepLabV3

sys.path.append(os.path.join(default_path,'utils'))
from utils import add_weight_decay
from utils import label_img_to_color
from utils import label_img_to_color_apolloscape

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import time
import glob

def getEpoch(checkpoint_name):
    filename_w_ext = os.path.basename(checkpoint_name)
    filename, file_extension = os.path.splitext(filename_w_ext)
    filenames = filename.split("_")
    return filenames[3]

# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "2"
eval_batch_size = 1

network = DeepLabV3(model_id, project_dir=default_path).cuda()

#check last checkpoint
data_list = glob.glob(os.path.join(network.checkpoints_dir,'model_'+model_id+'_*.pth'))

#find latest checkpoint
start_epoch = 0
for name in list(data_list):
    if start_epoch < int(getEpoch(name)):
        start_epoch = int(getEpoch(name))
if start_epoch != 0:
    network.load_state_dict(torch.load(os.path.join(network.checkpoints_dir,"model_" + model_id +"_epoch_" + str(start_epoch) + ".pth")))
    print("Recorver check point of epoch: " + str(start_epoch)) 
else:
    print("Can't find checkpoint for loading")
    quit()

val_dataset = DatasetVal()

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=eval_batch_size, shuffle=False,
                                         num_workers=30)

############################################################################
# inference:
############################################################################
network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

save_path = os.path.join(default_path,'inference/apolloscape')
if not os.path.exists(save_path):
    os.makedirs(save_path)

img_index = 0
print("Start inference")
for step, (imgs, label_imgs) in enumerate(val_loader):
    print("Eval step: " + str(step))
    
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))

        #label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

        # compute the loss:
        #outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        outputs = torch.argmax(outputs, dim=1)

        #pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
        pred_label_imgs = outputs.data.cpu().numpy()

        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        for i in range(pred_label_imgs.shape[0]):
            pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
            img = imgs[i] # (shape: (3, img_h, img_w))
            img = img.data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
            img = img*np.array([0.229, 0.224, 0.225])
            img = img + np.array([0.485, 0.456, 0.406])
            img = img*255.0
            img = img.astype(np.uint8)  
            pred_label_img_color = label_img_to_color_apolloscape(pred_label_img)
            overlayed_img = 0.35*img + 0.65*pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)
            save_file_path = os.path.join(save_path, str(img_index) + '.png')
            cv2.imwrite(save_file_path, overlayed_img)
            img_index += 1

