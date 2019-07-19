import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img
img_path = "images_aug"
msk_path = "masks_aug"
test_img_path = "images_test"
test_msk_path = "masks_test"


def write_imgpath():
    fp_train = open(train_path, 'w')
    fp_dev = open(dev_path,'w')
    fp_test = open(test_path, 'w')
    cnt = 0
    for file in os.listdir(img_path):
        cnt += 1
        if cnt%4==0:
            fp_dev.write(file+'\n')
        elif cnt%5==0:
            cnt = 0
            fp_test.write(file+'\n')
        else:
            fp_train.write(file+'\n')

    fp_train.close()
    fp_dev.close()
    fp_test.close()

def process_line(line):
    img_file = os.path.join(img_path, line)
    msk_file = os.path.join(msk_path, line)
    img = np.array(load_img(img_file,grayscale=True))/255
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    # msk = cv2.imread(msk_file)
    # msk = cv2.cvtColor(msk,cv2.COLOR_BGR2GRAY)
    msk = np.array(load_img(msk_file,grayscale=True))/255
    msk = cv2.resize(msk, (128, 128), interpolation=cv2.INTER_CUBIC)
    return img,msk

def process_line1(line):
    img_file = os.path.join(test_img_path, line)
    msk_file = os.path.join(test_msk_path, line)
    # img = cv2.imread(img_file)
    img = np.array(load_img(img_file,grayscale=True))/255
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    # msk = cv2.imread(msk_file)
    # msk = cv2.cvtColor(msk,cv2.COLOR_BGR2GRAY)
    msk = np.array(load_img(msk_file,grayscale=True))/255
    msk = cv2.resize(msk, (128, 128), interpolation=cv2.INTER_CUBIC)
    return img,msk

def get_train_data():
    list_x = []
    list_y = []
    cnt = 0
    for file in os.listdir(img_path):
        cnt += 1
        if cnt%4 == 0:
            cnt = 0
        else:
            x,y = process_line(file)
            list_x.append(x.reshape(128,128,1))
            list_y.append(y.reshape(128,128,1))
    return np.array(list_x),np.array(list_y)

def get_dev_data():
    list_x = []
    list_y = []
    cnt = 0
    for file in os.listdir(img_path):
        cnt += 1
        if cnt%4 == 0:
            x,y = process_line(file)
            list_x.append(x.reshape(128,128,1))
            list_y.append(y.reshape(128,128,1))
            cnt = 0
    return np.array(list_x),np.array(list_y)

def get_test_data():
    list_x = []
    list_y = []
    for file in os.listdir(test_img_path):
        x,y = process_line1(file)
        list_x.append(x.reshape(128,128,1))
        list_y.append(y.reshape(128,128,1))
    return np.array(list_x),np.array(list_y)

# data_test,name = predicted_data()
# result = model.predict(data_test,batch_size = 32)
# #
# resultsave("masks_predict",result,name)

def predicted_data():
    list_x = []
    img_file_name = []
    for file in os.listdir(test_img_path):
        x,y = process_line1(file)
        img_file_name.append(file)
        list_x.append(x.reshape(128,128,1))
    return np.array(list_x),img_file_name

def resultsave(savefilepath,data,file_name):
    for i in range(len(file_name)):
        img = data[i]
        path = os.path.join(savefilepath, file_name[i])
        img.reshape(128,128)
        img = cv2.resize(img, (101, 101), interpolation=cv2.INTER_CUBIC)
        img = img*255
        # ret,thresh1=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
        # 二分化图像，目前只在最后输出图像时做
        ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # print("path = ",path,"img = ",img)
        cv2.imwrite(path,binary)

def maskcopy(file_name):
    for i in range(len(file_name)):
        path = os.path.join(msk_path, file_name[i])
        img = cv2.imread(path)
        path = os.path.join("maskscopy", file_name[i])
        cv2.imwrite(path,img)

if __name__ == "__main__":
    # x_train,y_train = get_train_data()
    # x_dev,y_dev = get_dev_data()
    x_test,y_test = get_test_data()
    # x_test,name = predicted_data()
    print("x_test.shape = ",x_test.shape)
    print("y_test.shape = ",y_test.shape)
    # print("x_dev.shape = ",x_dev.shape)
    # print("y_dev.shape = ",y_dev.shape)
    # print("x_test.shape = ",x_test.shape)
    # print("len(name) = ",len(name))
