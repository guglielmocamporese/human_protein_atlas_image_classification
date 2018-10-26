import numpy as np
from utilities import *

# Generators
class MyDataGenerator():
    def __init__(self, df, data_dir, batch_size=32, mode='train', shuffle=True, seed=0, img_mode='rgby', target_size=512, train_dir=True):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.n_samples = self.df.shape[0]
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(1.*self.n_samples/self.batch_size))
        self.n_batch = 0
        self.img_mode = img_mode
        self.target_size = target_size
        self.train_dir = train_dir
        
        self.on_epoch_end()
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.mode=='train':
            x_batch, y_batch = self.get_batch(self.n_batch)
            self.n_batch = (self.n_batch + 1) % self.n_batches
            return x_batch, y_batch
        
        if self.model=='valid':
            x_batch, y_batch = self.get_batch(0)
            self.on_epoch_end()
            return x_batch, y_batch
        
        if self.mode=='test':
            if self.n_batch < self.n_batches:
                x_batch, y_batch = self.get_batch(self.n_batch)
                self.n_batch+=1
                return x_batch, y_batch
            else:
                raise StopIteration
            
            
    def get_batch(self, n_batch):
        idx_batch = []
        if n_batch < self.n_batches-1:
            idx_batch = np.copy(self.idx[n_batch*self.batch_size:(n_batch+1)*self.batch_size])
        else:
            idx_batch = np.copy(self.idx[n_batch*self.batch_size:])
            self.on_epoch_end()
        
        x_batch = np.array([read_img_from_zip(self.data_dir, img_id, self.target_size, mode=self.img_mode, train=self.train_dir)
                            for img_id in self.df['Id'].values[idx_batch]])
        y_batch = np.stack(self.df['target_oh'].values[idx_batch], axis=0)
        return x_batch, y_batch
        
        
    def on_epoch_end(self):
        self.idx = np.arange(self.n_samples)
        if self.shuffle:
            np.random.seed(seed=self.seed)
            np.random.shuffle(self.idx)
            self.seed+=1
        self.n_batch = 0
        return