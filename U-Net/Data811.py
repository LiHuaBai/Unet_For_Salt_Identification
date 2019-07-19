import os
import cv2
import numpy as np
from keras.preprocessing.image import load_img
img_path = "images"
msk_path = "masks"
train_path = "train_img_path.txt"
dev_path = "dev_img_path.txt"
test_path = "test_img_path.txt"

# 811分数据集

def write_imgpath():
    fp_train = open(train_path, 'w')
    fp_dev = open(dev_path,'w')
    fp_test = open(test_path, 'w')
    cnt = 0
    for file in os.listdir(img_path):
        # print("file = ",file)
        # input()
        cnt += 1
        if cnt%9==0:
            fp_dev.write(file+'\n')
        elif cnt%10==0:
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

def getdata(kind):
    list_x = []
    list_y = []
    count=0
    if kind == 0:
        path = train_path
    elif kind == 1:
        path = dev_path
    else:
        path = test_path
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '')
            x,y = process_line(line)
            list_x.append(x.reshape(128,128,1))
            list_y.append(y.reshape(128,128,1))
            count+=1
        return np.array(list_x),np.array(list_y)

def generator_data(batch_size,kind):
    list_x = []
    list_y = []
    count=0
    if kind == 0:
        path = train_path
    else:
        path = test_path
    while True:
        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')
                x,y = process_line(line)
                list_x.append(x.reshape(128,128,1))
                list_y.append(y.reshape(128,128,1))
                count+=1
                if count>=batch_size:
                    yield (np.array(list_x),np.array(list_y))
                    count=0
                    list_x=[]
                    list_y=[]

def predicted_data():
    list_x = []
    img_file_name = []
    with open(test_path) as f:
        for line in f:
            line = line.replace('\n', '')
            x,y = process_line(line)
            img_file_name.append(line)
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
    write_imgpath()
