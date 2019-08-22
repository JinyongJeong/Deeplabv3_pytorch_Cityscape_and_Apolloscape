# camera-ready
import torch
import torch.utils.data
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import cv2
import os
import pickle
import glob
import time
default_path = os.path.dirname(os.path.abspath(__file__))
apolloscape_data_path = os.path.join(default_path,'data/apolloscapes')
train_data_path_file = os.path.join(apolloscape_data_path,'train_data_path.pkl')
eval_data_path_file = os.path.join(apolloscape_data_path,'eval_data_path.pkl')
train_data_path = []
eval_data_path = []
with open(train_data_path_file, 'rb') as f:
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        train_data_path.append(data)
with open(eval_data_path_file, 'rb') as f:
     while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        eval_data_path.append(data)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):


        self.img_h = 560
        self.img_w = 1280

        ia.seed(2)
        self.seq = iaa.Sequential([
#                iaa.CropToFixedSize(width=400, height=400, position='uniform'), #crop
                iaa.PerspectiveTransform(scale=(0, 0.1), keep_size=True, cval=0),   #perspective
                iaa.Affine(scale={"x": (0.8, 1.2), "y":(0.8,1.2)},  #scale, rotation
                    rotate=(-20, 20), 
                    cval=0),
                iaa.CropToFixedSize(width=256, height=256, position='center'),  #set size
                iaa.Fliplr(0.5),    #flip
                iaa.Multiply((0.5, 1.5)),    #brightness
                iaa.Sharpen(alpha=(0.0,0.5), lightness=(0.8, 1.2)),
                sometimes(iaa.OneOf([
                    iaa.AverageBlur(k=(1,3)),
                    iaa.MedianBlur(k=(1,3)),
                    iaa.GaussianBlur(sigma=(0,0.1))
                    ])),
                sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
                sometimes(iaa.Grayscale((0.0, 0.8)))
                ])

                


        self.examples = []
        for train_dir in train_data_path:

            file_dir = os.path.join(train_dir,"*.png")
            file_list = glob.glob(file_dir)

            for file_path in file_list:
                img_path = file_path.replace('Labels_', 'ColorImage_resize_')
                img_path = img_path.replace('Label','ColorImage')
                img_path = img_path.replace('_bin.png','.jpg')
                label_path = file_path.replace('Labels_', 'Trainid_')
                
                if os.path.exists(img_path) and os.path.exists(label_path):
                    example = {}
                    example["img_path"] = img_path
                    example["label_img_path"] = label_path
                    self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        img_path = example["img_path"]
        img = cv2.imread(img_path, -1) # (shape: (560, 1280, 3)

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (560, 1280))
        tic = time.clock()

        img = img[None,:]
        label_img = label_img[None,:]
        toc = time.clock()
        print("1: " , str(toc - tic))
        tic = time.clock()
        
        #random crop for initial
        initial_crop_size = 400
        start_x = np.random.randint(low=0, high=(img.shape[2] - initial_crop_size))
        end_x = start_x + initial_crop_size
        start_y = np.random.randint(low=0, high=(img.shape[1] - initial_crop_size))
        end_y = start_y + initial_crop_size
        img = img[:,start_y:end_y, start_x:end_x,:] # (shape: (1, 400, 400, 3))
        label_img = label_img[:,start_y:end_y, start_x:end_x] # (shape: (1, 400, 400))

        img, label_img = self.seq(images=img, segmentation_maps=label_img)
        
        toc = time.clock()
        print("2:", str(toc - tic))
        tic = time.clock()
        img = img[0]
        label_img = label_img[0]
        img_basename = os.path.basename(img_path)
        label_basename = os.path.basename(label_img_path)
        test_img_save_path = os.path.join(default_path,'test', img_basename)
        test_label_save_path = os.path.join(default_path, 'test', label_basename)

        toc = time.clock()
        print("3:", str(toc - tic))

        cv2.imwrite(test_img_save_path, img)
        cv2.imwrite(test_label_save_path, label_img)

#        ########################################################################
#        # randomly scale the img and the label:
#        ########################################################################
#        scale = np.random.uniform(low=0.7, high=1.5)
#        new_img_h = int(scale*self.img_h)
#        new_img_w = int(scale*self.img_w)
#
#        # resize img without interpolation (want the image to still match
#        # label_img, which we resize below):
#        img = cv2.resize(img, (new_img_w, new_img_h),
#                         interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w, 3))
#
#        # resize label_img without interpolation (want the resulting image to
#        # still only contain pixel values corresponding to an object class):
#        label_img = cv2.resize(label_img, (new_img_w, new_img_h),
#                               interpolation=cv2.INTER_NEAREST) # (shape: (new_img_h, new_img_w))
#        ########################################################################
#        # # # # # # # # debug visualization START
#        # print (scale)
#        # print (new_img_h)
#        # print (new_img_w)
#        #
#        # cv2.imshow("test", img)
#        # cv2.waitKey(0)
#        #
#        # cv2.imshow("test", label_img)
#        # cv2.waitKey(0)
#        # # # # # # # # debug visualization END
#        ########################################################################
#        # select a 256x256 random crop from the img and label:
#        ########################################################################
#        start_x = np.random.randint(low=0, high=(new_img_w - 256))
#        end_x = start_x + 256
#        start_y = np.random.randint(low=0, high=(new_img_h - 256))
#        end_y = start_y + 256
#
#        img = img[start_y:end_y, start_x:end_x] # (shape: (256, 256, 3))
#        label_img = label_img[start_y:end_y, start_x:end_x] # (shape: (256, 256))
#        ########################################################################
#        # flip the img and the label with 0.5 probability:
#        flip = np.random.randint(low=0, high=2)
#        if flip == 1:
#            img = cv2.flip(img, 1)
#            label_img = cv2.flip(label_img, 1)
#        
#        # brightness augmentation
#        factor = 0.5
#        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#        hsv = np.array(hsv, dtype=np.float64)
#        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
#        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
#        img = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
#        # # # # # # # # debug visualization START
#        # print (img.shape)
#        # print (label_img.shape)
#        #
#        # cv2.imshow("test", img)
#        # cv2.waitKey(0)
#        #
#        # cv2.imshow("test", label_img)
#        # cv2.waitKey(0)
#        # # # # # # # # debug visualization END
#
#        # Data augmentation using imgaug
#        #img = np.transpose(img, (2,0,1))
#        #print("img: ")
#        #print(img.shape)
#        img, label_img = self.seq(image=img, segmentation_maps=label_img)
#        #img = np.transpose(img, (2,0,1))
#        #print("after img: ")
#        #print(img.shape)
#        
#
#
#        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img) # (shape: (256, 256))
        return (img, label_img)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self):

        self.img_h = 560
        self.img_w = 1280

        self.examples = []
        for eval_dir in eval_data_path:

            file_dir = os.path.join(eval_dir,"*.png")
            file_list = sorted(glob.glob(file_dir))

            for file_path in file_list:
                img_path = file_path.replace('Labels_', 'ColorImage_resize_')
                img_path = img_path.replace('Label','ColorImage')
                img_path = img_path.replace('_bin.png','.jpg')
                label_path = file_path.replace('Labels_', 'Trainid_')
                if os.path.exists(img_path) and os.path.exists(label_path):
                    example = {}
                    example["img_path"] = img_path
                    example["label_img_path"] = label_path
                    self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]


        img_path = example["img_path"]
        #print(img_path)
        img = cv2.imread(img_path, -1) # (shape: (560, 1280, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1) # (shape: (560, 1280))
        
        # # # # # # # # debug visualization START
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (560, 1280, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 560, 1280))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 560, 1280))
        label_img = torch.from_numpy(label_img) # (shape: (560, 1280))

        return (img, label_img)

    def __len__(self):
        return self.num_examples


