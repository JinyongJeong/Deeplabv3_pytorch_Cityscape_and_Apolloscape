# camera-ready
import atexit
import sys
import os
import datetime
import smtplib
from email.mime.text import MIMEText

default_path = os.path.dirname(os.path.abspath(__file__))

from datasets_apolloscape_imgaug_road import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append(os.path.join(default_path,'model'))
from deeplabv3_apolloscape_class_8 import DeepLabV3

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
import socket
def getEpoch(checkpoint_name):
    filename_w_ext = os.path.basename(checkpoint_name)
    filename, file_extension = os.path.splitext(filename_w_ext)
    filenames = filename.split("_")
    return filenames[3]



# NOTE! NOTE! change this to not overwrite all log data when you train the model:
model_id = "11"

num_epochs = 500
train_batch_size = 13
eval_batch_size = 1
learning_rate = 0.001

eval_stride = 10
checkpoint_save_stride = 5
logs_dir = os.path.join(default_path, 'training_logs')
checkpoints_dir = os.path.join(default_path, 'training_logs', 'model_' + str(model_id), 'checkpoints') 
model_dir = os.path.join(default_path, 'training_logs', 'model_' + str(model_id)) 

def exit_handler():
    print('*'*30)
    print("Program exit")
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(socket.gethostname())
    subject = 'DeeplabV3_report_{}'.format(end_time)
    body = 'Complete training \n Hostname {} \n IP Address {} \n model id {} \n learning rate {} \n'.format(hostname,ip_address, model_id, learning_rate)

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.login('deeplabv3.jjy0923@gmail.com','bmhrbnrgipufjsci')

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['To'] = 'jjy0923@gmail.com'
    smtp.sendmail('deeplabv3.jjy0923','jjy0923@gmail.com', msg.as_string())
    smtp.quit()
    print('*'*30)
    
atexit.register(exit_handler)



if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Single GPU
#network = DeepLabV3(model_id, project_dir=default_path).cuda()

# Multi GPU
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



train_dataset = DatasetTrain()
val_dataset = DatasetVal()

num_train_batches = int(len(train_dataset)/train_batch_size)
num_val_batches = int(len(val_dataset)/eval_batch_size)
print("num_train_dataset", len(train_dataset))
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=train_batch_size, shuffle=True,
                                           num_workers=30, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=eval_batch_size, shuffle=False,
                                         num_workers=30, drop_last=True)

params = add_weight_decay(network, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)

#with open(os.path.join(default_path,'data/apolloscapes/class_weights.pkl'), "rb") as file: # (needed for python3)
#    class_weights = np.array(pickle.load(file))
#class_weights = torch.from_numpy(class_weights)
#class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

with open(os.path.join(default_path,'data/apolloscapes/class_prob.pkl'), "rb") as file: # (needed for python3)
    class_prob = np.array(pickle.load(file))
class_weights = 1/np.log(1.02 + class_prob)

class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()



# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

epoch_losses_train = []
epoch_losses_val = []
for epoch in range(start_epoch, num_epochs+1):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch, num_epochs))

    ############################################################################
    # train:
    ############################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    for step, (imgs, label_imgs) in enumerate(train_loader):
        #current_time = time.time()
        print("Train Epoch: " + str(epoch) + " step: " + str(step))
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
    with open("%s/epoch_losses_train.pkl" % model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % model_dir)
    plt.close(1)
    print ("####")

    ############################################################################
    # val:
    ############################################################################

    if epoch%eval_stride == 0:
        network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (imgs, label_imgs) in enumerate(val_loader):
            print("Eval Epoch: " + str(epoch) + " step: " + str(step))
            
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

                        cv2.imwrite(model_dir + "/test.png", overlayed_img)


        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        with open("%s/epoch_losses_val.pkl" % model_dir, "wb") as file:
            pickle.dump(epoch_losses_val, file)
        print ("val loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % model_dir)
        plt.close(1)

    # save the model weights to disk:
    if epoch%checkpoint_save_stride == 0:
        checkpoint_path = checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)
