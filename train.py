# camera-ready

import sys
import os
default_path = os.path.dirname(os.path.abspath(__file__))

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append(os.path.join(default_path,'model'))
from deeplabv3 import DeepLabV3

sys.path.append(os.path.join(default_path,'utils'))
from utils import add_weight_decay
from utils import label_img_to_color


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
model_id = "1"

num_epochs = 500
batch_size = 3
learning_rate = 0.0001
checkpoint_save_stride = 10

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



train_dataset = DatasetTrain(cityscapes_data_path=os.path.join(default_path,'data/cityscapes'),
                             cityscapes_meta_path=os.path.join(default_path,'data/cityscapes/meta'))
val_dataset = DatasetVal(cityscapes_data_path=os.path.join(default_path,'data/cityscapes'),
                         cityscapes_meta_path=os.path.join(default_path,'data/cityscapes/meta'))

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=1)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=1)

params = add_weight_decay(network, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)

with open(os.path.join(default_path,'data/cityscapes/meta/class_weights.pkl'), "rb") as file: # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

epoch_losses_train = []
epoch_losses_val = []
for epoch in range(start_epoch, num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(train_loader):
        #current_time = time.time()
        #print(imgs.shape)
        #print(label_imgs.shape)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
        label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

        outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
        
        # compute the loss:
        loss = loss_fn(outputs, label_imgs)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        # optimization step:
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)


    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)
    print ("####")

    ############################################################################
    # val:
    ############################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

            outputs = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            #print (time.time() - current_time)
            outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
            pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            for i in range(pred_label_imgs.shape[0]):
                if i == 0:
                    pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                    img_id = img_ids[i]
                    img = imgs[i] # (shape: (3, img_h, img_w))

                    img = img.data.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                    img = img*np.array([0.229, 0.224, 0.225])
                    img = img + np.array([0.485, 0.456, 0.406])
                    img = img*255.0
                    img = img.astype(np.uint8)

                    pred_label_img_color = label_img_to_color(pred_label_img)
                    overlayed_img = 0.35*img + 0.65*pred_label_img_color
                    overlayed_img = overlayed_img.astype(np.uint8)

                    cv2.imwrite(network.model_dir + "/test.png", overlayed_img)


    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
    plt.close(1)

    # save the model weights to disk:
    if epoch%checkpoint_save_stride == 0:
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)
