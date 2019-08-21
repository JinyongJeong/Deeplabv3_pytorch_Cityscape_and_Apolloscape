# camera-ready

import sys
import os
default_path = os.path.dirname(os.path.abspath(__file__))

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
model_id = "3"
eval_batch_size = 1
logs_dir = os.path.join(default_path, 'training_logs')
checkpoints_dir = os.path.join(default_path, 'training_logs', 'model_' + str(model_id), 'checkpoints') 
model_dir = os.path.join(default_path, 'training_logs', 'model_' + str(model_id)) 


#network = DeepLabV3(model_id, project_dir=default_path).cuda()
network = DeepLabV3(model_id, project_dir=default_path)
network = nn.DataParallel(network)
network = network.cuda()
#check last checkpoint
data_list = glob.glob(os.path.join(checkpoints_dir,'model_'+model_id+'_*.pth'))

#find latest checkpoint
start_epoch = 0
for name in list(data_list):
    if start_epoch < int(getEpoch(name)):
        start_epoch = int(getEpoch(name))
if start_epoch != 0:
    network.load_state_dict(torch.load(os.path.join(checkpoints_dir,"model_" + model_id +"_epoch_" + str(start_epoch) + ".pth")))
    print("Recorver check point of epoch: " + str(start_epoch)) 
else:
    print("Can't find checkpoint for loading")
    quit()

############################################################################
# inference:
############################################################################
network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

save_path = os.path.join(default_path,'inference/urban_dataset')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# get data list
source_img_path = '/data/urban_dataset/urban39-pankyo/image/stereo_left'
img_list = sorted(glob.glob(os.path.join(source_img_path, '*.png')))

print(len(img_list))
print("Start inference")

img_index = 0
for img_path in img_list:
    print("inference image: " , str(img_index) , "/" , str(len(img_list)))
    with torch.no_grad():
        img_raw = cv2.imread(img_path, -1)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BAYER_BG2RGB)
        #cv2.imwrite( os.path.join(save_path, str(img_index) + '.png'), img_raw)
        #img_index += 1
        #continue
        img_raw = img_raw/255.0
        img_raw = img_raw - np.array([0.485, 0.456, 0.406])
        img_raw = img_raw/np.array([0.229, 0.224, 0.225]) # (shape: (560, 1280, 3))
        img_raw = np.transpose(img_raw, (2, 0, 1)) # (shape: (3, 560, 1280))
        img_raw = img_raw.astype(np.float32)

        # convert numpy -> torch:
        img_raw = torch.from_numpy(img_raw) # (shape: (3, 560, 1280))
        img_raw = img_raw[None, :]

        imgs = Variable(img_raw).cuda()
        #imgs = imgs[None, :,:,:]
        imgs = imgs.float()
        #imgs = imgs.transpose(2,1,0)
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

    

