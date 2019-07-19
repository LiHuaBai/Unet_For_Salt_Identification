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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def ResUnet(pretrained_weights = None,input_size = (128,128,1)):
    inputs = Input(input_size)
    shortcut_x1 = inputs
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    shortcut_x1 = Conv2D(64,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x1)
    conv1 = Add()([conv1,shortcut_x1])
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128->64
    shortcut_x2 = pool1
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    shortcut_x2 = Conv2D(128,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x2)
    conv2 = Add()([conv2,shortcut_x2])
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 64->32
    shortcut_x3 = pool2
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    shortcut_x3 = Conv2D(256,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x3)
    conv3 = Add()([conv3,shortcut_x3])
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 32->16
    shortcut_x4 = pool3
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    shortcut_x4 = Conv2D(512,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x4)
    conv4 = Add()([conv4,shortcut_x4])
    conv4 = Activation('relu')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # 16->8

    shortcut_x5 = pool4
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    shortcut_x5 = Conv2D(1024,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x5)
    conv5 = Add()([conv5,shortcut_x5])
    conv5 = Activation('relu')(conv5)
    drop5 = Dropout(0.5)(conv5)


    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    shortcut_x6 = up6
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    shortcut_x6 = Conv2D(512,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x6)
    conv6 = Add()([conv6,shortcut_x6])
    conv6 = Activation('relu')(conv6)


    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    shortcut_x7 = up7
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    shortcut_x7 = Conv2D(256,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x7)
    conv7 = Add()([conv7,shortcut_x7])
    conv7 = Activation('relu')(conv7)


    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    shortcut_x8 = up8
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    shortcut_x8 = Conv2D(128,3, padding = 'same', kernel_initializer = 'he_normal')(shortcut_x8)
    conv8 = Add()([conv8,shortcut_x8])
    conv8 = Activation('relu')(conv8)

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
