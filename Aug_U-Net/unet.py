import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras import metrics
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from my_iou import *
# from my_loss import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights = None,input_size = (128,128,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128->64
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 64->32
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 32->16
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # 16->8

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    # sgd = SGD(lr=0.0001,decay=1e-5,momentum=0.9,nesterov=True)
    sgd = SGD(lr=0.0001,decay=1e-5)
    # Adam(lr = 1e-4)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['acc'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [my_iou_metric])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
# def iou(y_true, y_pred, label: int):
#     """
#     Return the Intersection over Union (IoU) for a given label.
#     Args:
#         y_true: the expected y values as a one-hot
#         y_pred: the predicted y values as a one-hot or softmax output
#         label: the label to return the IoU for
#     Returns:
#         the IoU for the given label
#     """
#     # extract the label values using the argmax operator then
#     # calculate equality of the predictions and truths to the label
#     y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
#     y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
#     # calculate the |intersection| (AND) of the labels
#     intersection = K.sum(y_true * y_pred)
#     # calculate the |union| (OR) of the labels
#     union = K.sum(y_true) + K.sum(y_pred) - intersection
#     # avoid divide by zero - if the union is zero, return 1
#     # otherwise, return the intersection over union
#     return K.switch(K.equal(union, 0), 1.0, intersection / union)
# def IoU(y_true, y_pred, eps=1e-6):
#     if np.max(y_true) == 0.0:
#         return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
#     return -K.mean( (intersection + eps) / (union + eps), axis=0)
