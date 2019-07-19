import numpy as np
import os
from keras.preprocessing.image import load_img
# import tensorflow as tf
# 输入为np矩阵
mfile = 'maskscopy'
pfile = 'predicts_ResUnet_dataAug'
test_path = 'test_img_path.txt'
test_img = '/0a1742c740.png'
def get_iou(mask,pred):
    Tsum = np.sum(mask)
    Psum = np.sum(pred)
    if Tsum == 0:
        if Psum == 0:
            iou = 1
        else:
            iou = 1
        return iou
    # 简单对应位置上的数量积
    intersection = np.sum(mask * pred)
    union = Tsum + Psum - intersection
    # print("Tsum = ",Tsum," Psum = ",Psum,"intersection = ",intersection," union = ",union)
    iou = intersection / union
    return iou
def cal_pic_iou(mask_file,pred_file):
    mask = np.array(load_img(mask_file,color_mode = "grayscale"))/255
    pred = np.array(load_img(pred_file,color_mode = "grayscale"))/255
    return get_iou(mask,pred)

def cal_all_img():
    Tcnt = 0
    Fcnt = 0
    for file in os.listdir(mfile):
        mask_file = os.path.join(mfile, file)
        pred_file = os.path.join(pfile, file)
        iou = cal_pic_iou(mask_file,pred_file)
        # if iou < 0:
        #     continue
        if iou >= 0.5:
            Tcnt += 1
        else:
            Fcnt += 1
        print("Tcnt = ",Tcnt," Fcnt = ",Fcnt)

if __name__ == "__main__":
    cal_all_img()
