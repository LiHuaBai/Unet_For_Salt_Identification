import numpy as np
from imgaug import augmenters as iaa
import os
import cv2

img_path = "images"
img_aug_path = "images_aug"
test_img_path = "images_test"
test_msk_path = "masks_test"
msk_path = "masks"
msk_aug_path = "masks_aug"
train_path = "train_img_path.txt"
dev_path = "dev_img_path.txt"
test_path = "test_img_path.txt"

# 将.txt中的训练数据保存到数据Aug文件夹
def copytraindata():
    with open(train_path) as f:
        for line in f:
            line = line.replace('\n', '')
            img_file = os.path.join(img_path, line)
            # img_file = os.path.join(msk_path, line)
            img = cv2.imread(img_file)
            img_aug_file = os.path.join(img_aug_path,line)
            cv2.imwrite(img_aug_file,img)

def copytestdata():
    with open(test_path) as f:
        for line in f:
            line = line.replace('\n', '')
            img_file = os.path.join(img_path, line)
            # img_file = os.path.join(msk_path, line)
            img = cv2.imread(img_file)
            img_test_file = os.path.join(test_img_path,line)
            cv2.imwrite(img_test_file,img)

def copytraindata_msk():
    with open(train_path) as f:
        for line in f:
            line = line.replace('\n', '')
            msk_file = os.path.join(msk_path, line)
            msk = cv2.imread(msk_file)
            msk_aug_file = os.path.join(msk_aug_path,line)
            cv2.imwrite(msk_aug_file,msk)


def copytestdata_msk():
    with open(test_path) as f:
        for line in f:
            line = line.replace('\n', '')
            msk_file = os.path.join(msk_path, line)
            # img_file = os.path.join(msk_path, line)
            msk = cv2.imread(msk_file)
            msk_test_file = os.path.join(test_msk_path,line)
            cv2.imwrite(msk_test_file,msk)


def copydevdata():
    with open(dev_path) as f:
        for line in f:
            line = line.replace('\n', '')
            img_file = os.path.join(img_path, line)
            img = cv2.imread(img_file)
            img_aug_file = os.path.join(img_aug_path,line)
            cv2.imwrite(img_aug_file,img)

def copydevdata_msk():
    with open(dev_path) as f:
        for line in f:
            line = line.replace('\n', '')
            msk_file = os.path.join(msk_path, line)
            msk = cv2.imread(msk_file)
            msk_aug_file = os.path.join(msk_aug_path,line)
            cv2.imwrite(msk_aug_file,msk)

# 镜像变化后改名，保存在数据Aug文件夹
def aug_img():
    seq = iaa.Sequential([
        # iaa.Crop(px=(0, 16)), # 从每侧裁剪图像0到16px（随机选择）
        iaa.Fliplr(1), # 水平翻转图像
        # iaa.GaussianBlur(sigma=(0, 3.0)) # 使用0到3.0的sigma模糊图像
    ])
    imglist=[]
    # cnt = 0
    for file in os.listdir(img_aug_path):
        # cnt += 1
        # print(cnt)
        img_file = os.path.join(img_aug_path, file)
        img = cv2.imread(img_file)
        imglist.append(img)
        images_aug = seq.augment_images(imglist)
        img_name,en = os.path.splitext(file)
        img_aug_file = img_name + '_aug' + en
        img_aug_file = os.path.join(img_aug_path, img_aug_file)
        cv2.imwrite(img_aug_file,images_aug[0])
        images_aug.clear()
        imglist.clear()

def aug_msk():
    seq = iaa.Sequential([
        # iaa.Crop(px=(0, 16)), # 从每侧裁剪图像0到16px（随机选择）
        iaa.Fliplr(1), # 水平翻转图像
        # iaa.GaussianBlur(sigma=(0, 3.0)) # 使用0到3.0的sigma模糊图像
    ])
    msklist=[]
    # cnt = 0
    for file in os.listdir(msk_aug_path):
        # cnt += 1
        # print(cnt)
        msk_file = os.path.join(msk_aug_path, file)
        msk = cv2.imread(msk_file)
        msklist.append(msk)
        images_aug = seq.augment_images(msklist)
        msk_name,en = os.path.splitext(file)
        msk_aug_file = msk_name + '_aug' + en
        msk_aug_file = os.path.join(msk_aug_path, msk_aug_file)
        cv2.imwrite(msk_aug_file,images_aug[0])
        images_aug.clear()
        msklist.clear()

if __name__ == "__main__":
    aug_msk()
