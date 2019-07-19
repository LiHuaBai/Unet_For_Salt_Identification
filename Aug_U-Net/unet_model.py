import numpy as np
import pandas as pd
from random import randint
# from sklearn.model_selection import train_test_split
#
# from skimage.transform import resize
from my_iou import *
from my_loss import *
from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

img_size_ori = 101  # 需先变成128*128后再计算
img_size_target = 128 # 实际要求输出也是101*101
def Unet(input_shape=(128, 128, 1)):
    input_layer = Input(input_shape)
    start_neurons = 64
    #128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    # pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(convm)
    # convm = Dropout(0.5)(convm)
    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(convm) # 目前是通过反向理解输出大小，即先想好什么样的输入按设定步长和卷积核大小，正卷积后会得到一直的大小。这样的输入即反卷积的输出
    uconv4 = concatenate([deconv4, conv4])
    # uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    # uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    # uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer = 'he_normal')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    # uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same", kernel_initializer = 'he_normal')(uconv1)

    # uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    model = Model(input_layer, output_layer)

    sgd = SGD(lr=0.0001,decay=1e-5)
    # Adam(lr = 1e-4)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['acc'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = [tversky_loss], metrics = [my_iou_metric])
    return model
