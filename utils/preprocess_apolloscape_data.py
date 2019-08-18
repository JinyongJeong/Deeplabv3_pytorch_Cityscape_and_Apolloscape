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
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)

labels = [
    #           name     id trainId      category  catId hasInstances ignoreInEval            color
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
    Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
    Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),
    Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
    Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
    Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
    Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
    Label(    'noise' , 249 ,   255 ,    'ignored' ,  12 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,   255 ,    'ignored' ,  12 ,      False ,       True , (255, 255, 255) ),
]

# create a function which maps id to trainId:
# name to label object
name_to_label      = { label.name    : label for label in labels           }
# id to label object
id_to_label        = { label.id      : label for label in labels           }
# trainId to label object
trainId_to_label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category_to_labels = {}
for label in labels:
    category = label.category
    if category in category_to_labels:
        category_to_labels[category].append(label)
    else:
        category_to_labels[category] = [label]

id_to_trainId = {label.id: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

apolloscape_data_path = os.path.join(default_path,'./../data/apolloscapes')
data_path = os.path.join(apolloscape_data_path,'Labels_*/Label/Record*/Camera 5')
data_paths = glob.glob(data_path)

#make color map for fast conversion
#In our application, we use categoryId for trainID
color_map = np.ndarray(shape=(256*256*256), dtype='int32')
color_map[:] = 0
for label in labels:
    #rgb = label.color[0] * 65536 + label.color[1] * 256 + label.color[2]
    rgb = label.color[2] * 65536 + label.color[1] * 256 + label.color[0]
    color_map[rgb] = label.categoryId

################################################################################
# convert all labels to label imgs with trainId pixel values (and save to disk):
################################################################################

eval_data_rate = 0.1
image_overwrite = False

total_num_images = 0
#count total image number
random.shuffle(data_paths)
train_data_paths = data_paths[0:round(len(data_paths)*(1-eval_data_rate))]
eval_data_paths = data_paths[round(len(data_paths)*(1-eval_data_rate)) : len(data_paths)]
print("Number of training path: ", str(len(train_data_paths)))
print("Number of evaluation path: ", str(len(eval_data_paths)))
print("Number of total path: ", str(len(data_paths)))

for path in list(data_paths):
    image_path = os.path.join(path, '*.png')
    images = glob.glob(image_path)
    total_num_images = total_num_images + len(images)
    
image_index = 0
for path in list(data_paths):
    save_path = path.replace("Labels_", "Trainid_")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_path = os.path.join(path, '*.png')
    images = glob.glob(image_path)
    print("Converting image path: ", path)
    print("Number of images: ", len(images))
    for image in list(images):
        image_index = image_index + 1
        save_filename = os.path.basename(image)
        save_file = os.path.join(save_path, save_filename)
        if not image_overwrite:
            if os.path.exists(save_file):
                print("Converted file exist " + str(image_index) +"/" + str(total_num_images))
                continue
        print("processing..." + str(image_index) + "/" + str(total_num_images))
        label_img = cv2.imread(image, cv2.IMREAD_COLOR)
        TrainId_img = label_img.dot(np.array([65536, 256, 1], dtype='int32'))
        TrainId_img = color_map[TrainId_img]

        cv2.imwrite(save_file, TrainId_img);
        print(os.path.join(save_path,save_filename))
       

################################################################################
# compute the class weigths:
################################################################################
print ("computing class weights")

num_classes = 13

trainId_to_count = {}
for trainId in range(num_classes):
    trainId_to_count[trainId] = 0

step = 0
for train_data_path in list(train_data_paths):
    train_data_path = train_data_path.replace("Labels_","Trainid_")
    image_path = os.path.join(train_data_path, '*.png')
    images = glob.glob(image_path)
    for image in list(images):
        step = step + 1
        if step % 10 == 0:
            print("Compute class weight: " + str(step) + "/" + str(round(total_num_images * (1-eval_data_rate))))
        label_img = cv2.imread(image, -1)
        for trainId in range(num_classes):
            trainId_mask = np.equal(label_img, trainId)
            trainId_count = np.sum(trainId_mask)

            # add to the total count:
            trainId_to_count[trainId] += trainId_count

# compute the class weights according to the ENet paper:
class_weights = []
total_count = sum(trainId_to_count.values())
print("total count: "+ str(total_count))
for trainId, count in trainId_to_count.items():
    trainId_prob = float(count)/float(total_count)
    trainId_weight = 1/np.log(1.02 + trainId_prob)
    class_weights.append(trainId_weight)
    print("trainId: " + str(trainId) + " count: " + str(count) + " prob: " + str(trainId_prob) + " weight: " + str(trainId_weight))

print (class_weights)

with open(apolloscape_data_path + "/class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)

# Save train & eval path
with open(apolloscape_data_path + "/train_data_path.pkl","wb") as file:
    for data_path in train_data_paths:
        pickle.dump(data_path, file)
with open(apolloscape_data_path + "/eval_data_path.pkl","wb") as file:
    for data_path in eval_data_paths:
        pickle.dump(data_path, file)

