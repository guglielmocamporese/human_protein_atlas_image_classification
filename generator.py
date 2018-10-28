import numpy as np
from hpaic.utilities import *
from keras.utils import Sequence

# Generators
class MyDataGenerator(Sequence):
    def __init__(self, df, data_dir, batch_size=32, mode='train', shuffle=True, seed=0, img_mode='rgby', target_size=512, train_dir=True, kernel='colab'):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.n_samples = self.df.shape[0]
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(1.*self.n_samples/self.batch_size))
        self.img_mode = img_mode
        self.target_size = target_size
        self.train_dir = train_dir
        self.kernel = kernel
        
        self.on_epoch_end()

    def __len__(self):
        return self.n_batches
        
    def __getitem__(self, n_batch):
        if self.mode=='train':
            x_batch, y_batch = self.get_batch(n_batch)
            return x_batch, y_batch
        
        if self.mode=='valid':
            x_batch, y_batch = self.get_batch(0)
            self.on_epoch_end()
            return x_batch, y_batch
        
        if self.mode=='test':
            if self.n_batch < self.n_batches:
                x_batch = self.get_batch(n_batch)
                return x_batch
            else:
                raise StopIteration
            
            
    def get_batch(self, n_batch):
        read_img_fun = None
        if self.kernel=='colab':
            read_img_fun = read_img_from_zip
        if self.kernel=='kaggle':
            read_img_fun = read_img

        idx_batch = []
        if n_batch < self.n_batches-1:
            idx_batch = np.copy(self.idx[n_batch*self.batch_size:(n_batch+1)*self.batch_size])
        else:
            idx_batch = np.copy(self.idx[n_batch*self.batch_size:])
            self.on_epoch_end()
        
        x_batch = np.array([read_img_fun(self.data_dir, img_id, self.target_size, mode=self.img_mode, train=self.train_dir)
                            for img_id in self.df['Id'].values[idx_batch]])

        if self.mode=='train' or self.mode=='valid':
            y_batch = np.stack(self.df['target_oh'].values[idx_batch], axis=0)
            return x_batch, y_batch
        if self.mode=='test':
            return x_batch
        
        
    def on_epoch_end(self):
        self.idx = np.arange(self.n_samples)
        if self.shuffle:
            np.random.seed(seed=self.seed)
            np.random.shuffle(self.idx)
            self.seed+=1
        self.n_batch = 0
        return