# Import Packeages for the Model
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.backend import tf as ktf

def big_model():
	
    # Pre-Trained Model
    model_pt = PTModel(include_top=False, weights='imagenet', input_shape=[299,299,3])
    
    img_in = Input([*SHAPE[:-1], 1])
    x = concatenate([img_in, img_in, img_in], axis=-1)
    x = Lambda(lambda image: ktf.image.resize_images(image, (299, 299)))(x)
    x = BatchNormalization()(x)
    x = model_pt(x)
    x = Conv2D(32, [1,1], activation='relu')(x)
    
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(N_CLASSES, activation='sigmoid')(x)
    
    model = Model(img_in, x)
    
    return model



