import numpy as np
from zipfile import ZipFile
from keras.preprocessing.image import load_img
from io import BytesIO
from IPython.display import clear_output

def download_dataset():
    '''
        This function is used to download and manage the dataset in the colab envoirment.
    '''
    # File Configuration
    !mkdir ~/.kaggle
    !cp kaggle.json ~/.kaggle/

    !chmod 600 ~/.kaggle/kaggle.json

    # Download Dataset
    !kaggle competitions download -c human-protein-atlas-image-classification

    # Create Folders
    !mkdir dataset

    # Manage Dataset
    !mv train.zip dataset
    !mv test.zip dataset
    !mv train.csv dataset
    !mv sample_submission.csv dataset

    # Remove Useless Data
    !rm sample_data/*
    !rmdir sample_data

    clear_output(wait=True)
    return

def read_img_from_zip(data_dir, img_id, target_size, mode='rgby', train=True):
    # Select Train or Test
    archive = ZipFile(data_dir+'train.zip', 'r')
    if not train:
        archive = ZipFile(data_dir+'test.zip', 'r')
        
    if mode=='rgby':
        img_data_r = archive.read('{}_red.png'.format(img_id))
        img_data_g = archive.read('{}_green.png'.format(img_id))
        img_data_b = archive.read('{}_blue.png'.format(img_id))
        img_data_y = archive.read('{}_yellow.png'.format(img_id))

        bytes_io_r = BytesIO(img_data_r)
        bytes_io_g = BytesIO(img_data_g)
        bytes_io_b = BytesIO(img_data_b)
        bytes_io_y = BytesIO(img_data_y)

        img_r = np.array(load_img(bytes_io_r, grayscale=True, target_size=[target_size, target_size]))
        img_g = np.array(load_img(bytes_io_g, grayscale=True, target_size=[target_size, target_size]))
        img_b = np.array(load_img(bytes_io_b, grayscale=True, target_size=[target_size, target_size]))
        img_y = np.array(load_img(bytes_io_y, grayscale=True, target_size=[target_size, target_size]))
        return np.stack([img_r, img_g, img_b, img_y], axis=-1) / 255.
    if mode=='rgb':
        img_data_r = archive.read('{}_red.png'.format(img_id))
        img_data_g = archive.read('{}_green.png'.format(img_id))
        img_data_b = archive.read('{}_blue.png'.format(img_id))

        bytes_io_r = BytesIO(img_data_r)
        bytes_io_g = BytesIO(img_data_g)
        bytes_io_b = BytesIO(img_data_b)

        img_r = np.array(load_img(bytes_io_r, grayscale=True, target_size=[target_size, target_size]))
        img_g = np.array(load_img(bytes_io_g, grayscale=True, target_size=[target_size, target_size]))
        img_b = np.array(load_img(bytes_io_b, grayscale=True, target_size=[target_size, target_size]))
        return np.stack([img_r, img_g, img_b], axis=-1) / 255.
    if mode=='rgby':
        img_data_g = archive.read('{}_green.png'.format(img_id))

        bytes_io_g = BytesIO(img_data_g)

        img_g = np.array(load_img(bytes_io_g, grayscale=True, target_size=[target_size, target_size]))
        return np.stack([img_g], axis=-1) / 255.
    
def read_img(data_dir, img_id, target_size, mode='rgby', train=True):
    # Select Train or Test
    img_dir = data_dir+'train/'
    if not train:
        img_dir = data_dir+'test/'
    
    if mode=='rgby':
        img_r = np.array(load_img(img_dir+'{}_red.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        img_g = np.array(load_img(img_dir+'{}_green.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        img_b = np.array(load_img(img_dir+'{}_blue.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        img_y = np.array(load_img(img_dir+'{}_yellow.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        return np.stack([img_r, img_g, img_b, img_y], axis=-1) / 255.
    if mode=='rgb':
        img_r = np.array(load_img(img_dir+'{}_red.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        img_g = np.array(load_img(img_dir+'{}_green.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        img_b = np.array(load_img(img_dir+'{}_blue.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        return np.stack([img_r, img_g, img_b], axis=-1) / 255.
    if mode=='g':
        img_g = np.array(load_img(img_dir+'{}_green.png'.format(img_id), grayscale=True, target_size=[target_size, target_size]))
        return np.stack([img_g], axis=-1) / 255.
    