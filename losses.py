import keras.backend as K
import tensorflow as tf

''' 
Focal Loss
    loss for binary classification that accounts for imbalance-class batch
    implementation from: https://arxiv.org/pdf/1708.02002.pdf
'''
def focal_loss(gamma=2, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        alpha_t = y_true*alpha + (1-y_true)*(1-alpha)
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred) + K.epsilon()
        FL = - alpha_t * K.pow((1-p_t),gamma) * K.log(p_t)
        return K.mean(FL)
    return focal_loss_fixed

'''
    Macro F1-Score Loss
'''
def f1_loss(y_true, y_pred):
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


'''
    Mixed Loss
'''
def mixed_loss(y_true, y_pred):
    return focal_loss(gamma=2, alpha=0.25)(y_true, y_pred) + f1_loss(y_true, y_pred)