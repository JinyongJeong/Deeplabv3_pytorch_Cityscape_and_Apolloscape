# camera-ready
import pickle
import numpy as np
import cv2
import os
import glob
import random
from collections import namedtuple
default_path = os.path.dirname(os.path.abspath(__file__))

# (NOTE! this is taken from the official Cityscapes scripts:)


################################################################################
# convert all labels to label imgs with trainId pixel values (and save to disk):
################################################################################

eval_data_rate = 0.05
image_overwrite = True

# original dataset image size
origin_img_h = 560
origin_img_w = 1280

# ROI region
new_img_roi_h = 320
new_img_roi_w = 1280
new_img_center_h = 400
new_img_center_w = 640

new_img_h = 320
new_img_w = 1280

total_num_images = 0

source_img_path = '/data/urban_dataset/urban39-pankyo/image/stereo_left_color'
source_label_path =  '/data/urban_dataset/urban39-pankyo/image/stereo_left_trainID'

#source_img = glob.glob(os.path.join(source_img_path, '*.png'))
source_label = glob.glob(os.path.join(source_label_path, '*.png'))

random.shuffle(source_label)
#train_label_data = source_label[0:round(len(source_label)*(1-eval_data_rate))]
#eval_label_data = source_label[round(len(source_label)*(1-eval_data_rate)) : len(source_label)]
#print("Number of training img: ", str(len(train_label_data)))
#print("Number of evaluation img: ", str(len(eval_label_data)))


crop_left_color_path = os.path.join(source_img_path, './../resize_stereo_left_color')
crop_left_label_path = os.path.join(source_label_path, './../resize_stereo_left_trainID')
if not os.path.exists(crop_left_color_path):
    os.makedirs(crop_left_color_path)
if not os.path.exists(crop_left_label_path):
    os.makedirs(crop_left_label_path)

train_label_data = []
eval_label_data = []
number_of_label_data = (1- eval_data_rate) * len(source_label)

image_index = 0;
for image in list(source_label):
        save_filename = os.path.basename(image)
        print(save_filename)
        print("processing... " + str(image_index) + "/" + str(len(source_label)));
        color_img = cv2.imread(os.path.join(source_img_path, save_filename), cv2.IMREAD_COLOR)
        label_img = cv2.imread(os.path.join(source_label_path, save_filename),cv2.IMREAD_GRAYSCALE)
        if color_img is None  or label_img is None:
            print("Skip data:")
            continue
        color_img = color_img[max(0,new_img_center_h - round(new_img_roi_h/2)):min(origin_img_h-1,new_img_center_h + round(new_img_roi_h/2)), max(0,new_img_center_w - round(new_img_roi_w/2)):min(origin_img_w-1,new_img_center_w + round(new_img_roi_w/2))]
        label_img = label_img[max(0,new_img_center_h - round(new_img_roi_h/2)):min(origin_img_h-1,new_img_center_h + round(new_img_roi_h/2)), max(0,new_img_center_w - round(new_img_roi_w/2)):min(origin_img_w-1,new_img_center_w + round(new_img_roi_w/2))]

        cv2.imwrite(os.path.join(crop_left_color_path, save_filename), color_img)
        cv2.imwrite(os.path.join(crop_left_label_path, save_filename), label_img)
        if image_index < number_of_label_data:
            train_label_data.append(os.path.join(crop_left_label_path, save_filename))
        else:
            eval_label_data.append(os.path.join(crop_left_label_path, save_filename))
        image_index = image_index + 1
       
       
# Save train & eval path
with open(os.path.join(source_img_path, './../') + "/train_data_path.pkl","wb") as file:
    for data_path in train_label_data:
        pickle.dump(data_path, file)
with open(os.path.join(source_img_path, './../') + "/eval_data_path.pkl","wb") as file:
    for data_path in eval_label_data:
        pickle.dump(data_path, file)

################################################################################
# compute the class weigths:
################################################################################
print ("computing class weights")

num_classes = 8

trainid_to_count = {}
for trainid in range(num_classes):
    trainid_to_count[trainid] = 0

step = 0
for image in list(train_label_data):
    step = step + 1
    label_img = cv2.imread(image, -1)
    for trainid in range(num_classes):
        trainid_mask = np.equal(label_img, trainid)
        trainid_count = np.sum(trainid_mask)

        # add to the total count:
        trainid_to_count[trainid] += trainid_count

# compute the class weights according to the enet paper:
class_weights = []
class_prob = []
total_count = sum(trainid_to_count.values())
print("==========inverse log weight==========")
print("total count: "+ str(total_count))
for trainid, count in trainid_to_count.items():
    trainid_prob = float(count)/float(total_count)
    trainid_weight = 1/np.log(1.02 + trainid_prob)
    class_weights.append(trainid_weight)
    class_prob.append(trainid_prob)
    print("trainid: " + str(trainid) + " count: " + str(count) + " prob: " + str(trainid_prob) + " weight: " + str(trainid_weight))

print (class_weights)
print (class_prob)

with open(os.path.join(source_img_path, './../') + "/class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
with open(os.path.join(source_img_path, './../') + "/class_prob.pkl", "wb") as file:
    pickle.dump(class_prob, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)


