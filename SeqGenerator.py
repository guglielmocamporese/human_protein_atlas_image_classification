# Import Packages
import numpy as np
import pandas as pd
import cv2
from imgaug import augmenters as iaa
from keras.utils import Sequence

class SeqGenerator(Sequence):
    def __init__(self, df, img_size, data_dir, mode='train', batch_size=32, seed=123, augmentation=False, img_mode='rgby'):
        self.df = df
        self.img_size = img_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode 
        self.img_mode = img_mode
        if self.mode=='train':
            self.shuffle = True
            self.augmentation = augmentation
        if self.mode=='valid':
            self.shuffle = True
            self.augmentation = False
        if self.mode=='test':
            self.shuffle = False
            self.augmentation = False
        self.seed = seed
        self.n_samples = self.df.shape[0]
        self.n_batch = int(np.ceil(self.n_samples / self.batch_size))
        self.on_epoch_end()
        
    def __len__(self):
        return self.n_batch
    
    def __getitem__(self, index):
        if self.mode=='train':
            return self.get_batch_train(index)
        if self.mode=='valid':
            return self.get_batch_valid(index)
        if self.mode=='test':
            return self.get_batch_test(index)
        
    def on_epoch_end(self):
        self.idx = np.arange(self.n_samples)
        if self.shuffle:
            np.random.seed(seed=self.seed)
            np.random.shuffle(self.idx)
            self.seed += 1
        return
    
    def get_batch_train(self, idx_cur):
        if idx_cur < self.n_batch-1:
            idx_batch = np.copy(self.idx[idx_cur*self.batch_size:(idx_cur+1)*self.batch_size])
            x_batch = np.array([self.read_img(img_id, train=True, mode=self.img_mode, normalize=True) for img_id in self.df['Id'].values[idx_batch]])
            if self.augmentation:
                x_batch = self.augment_img(x_batch)
            y_batch = np.stack(self.df['y_one_hot'].values[idx_batch], axis=0)
            return x_batch, y_batch
        else:
            # Last Mini-Batch
            idx_batch = np.copy(self.idx[idx_cur*self.batch_size:])
            x_batch = np.array([self.read_img(img_id, train=True, mode=self.img_mode, normalize=True) for img_id in self.df['Id'].values[idx_batch]])
            if self.augmentation:
                x_batch = self.augment_img(x_batch)
            y_batch = np.stack(self.df['y_one_hot'].values[idx_batch], axis=0)
            return x_batch, y_batch
    
    def get_batch_valid(self, idx_cur):
        idx_batch = np.copy(self.idx[idx_cur*self.batch_size:(idx_cur+1)*self.batch_size])
        x_batch = np.array([self.read_img(img_id, train=True, mode=self.img_mode, normalize=True) for img_id in self.df['Id'].values[idx_batch]])
        if self.augmentation:
            x_batch = self.augment_img(x_batch)
        y_batch = np.stack(self.df['y_one_hot'].values[idx_batch], axis=0)
        self.on_epoch_end()
        return x_batch, y_batch
    
    def get_batch_test(self, idx_cur):
        if idx_cur < self.n_batch-1:
            idx_batch = np.copy(self.idx[idx_cur*self.batch_size:(idx_cur+1)*self.batch_size])
            x_batch = np.array([self.read_img(img_id, train=False, mode=self.img_mode, normalize=True) for img_id in self.df['Id'].values[idx_batch]])
            if self.augmentation:
                x_batch = self.augment_img(x_batch)
            return x_batch
        else:
            # Last Mini-Batch
            idx_batch = np.copy(self.idx[idx_cur*self.batch_size:])
            x_batch = np.array([self.read_img(img_id, train=False, mode=self.img_mode, normalize=True) for img_id in self.df['Id'].values[idx_batch]])
            if self.augmentation:
                x_batch = self.augment_img(x_batch)
            return x_batch


    ### Downsample Function
    def downsample(self, img):
        if self.img_size==img.shape[0]:
            return img
        else:
            return cv2.resize(img, tuple([self.img_size, self.img_size]))

    ### Load Image Function
    def read_img(self, img_id, train=True, mode='rgby', normalize=True):
        # Choose if Train / Test Image
        train_test_dir = 'train/'
        if not train:
            train_test_dir = 'test/'
        
        # Select Mode
        channel_list = ['red', 'green', 'blue', 'yellow']
        if mode=='rgby':
            channels = channel_list
        if mode=='rgb':
            channels = channel_list[:-1]
        if mode=='g':
            channels = channel_list[0:1]
        if mode=='mean_rgby':
            channels = channel_list
        
        # Normalization
        if normalize:
            img = np.stack([self.downsample(cv2.imread(self.data_dir+train_test_dir+img_id+'_{}.png'.format(channel), cv2.IMREAD_GRAYSCALE)
                                        .astype('float32')/255.)
                                for channel in channels], axis=-1)
            if mode[:4]=='mean':
                return np.mean(img, axis=-1)
            else:
                return img
        else:
            img = np.stack([self.downsample(cv2.imread(self.data_dir+train_test_dir+img_id+'_{}.png'.format(channel), cv2.IMREAD_GRAYSCALE)
                                        .astype('float32'))
                                for channel in channels], axis=-1)
            if mode[:4]=='prod':
                return np.mean(img, axis=-1)
            else:
                return img

    ### Augment Function
    def augment_img(self, images):
        # Augment a Batch of Images of Shape = [batch_size, h, w, c]
        aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)
        return aug.augment_images(images)