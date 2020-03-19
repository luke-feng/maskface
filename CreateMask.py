import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os


dir_path = '/Users/chaofeng/Documents/photo/6/vs_train'


def readImg(path, file_name):
    '''
    read a image and return an array.
    '''
    file = path + '/' + file_name
    im = np.array(Image.open(file))
    return im

def getStartEnd(depts):
    start = []
    end = []
    start.append(random.randint(0,200-3*depts))
    start.append(random.randint(0,200-3*depts))
    end.append(random.randint(start[0]+depts, 200-depts))
    end.append(random.randint(start[1]+depts, 200-depts))
    return start, end

def masking(file_name, path, depts):
    im = readImg(path, file_name)

    start, end = getStartEnd(depts)
    mask_area = im[start[0]:end[0], start[1]:end[1]]
    masking = mask_area[::depts, ::depts]
    m_ = masking.shape
    print(m_)
    mask_w = m_[0]
    mask_y = m_[1]
    im_masked = im.copy()
    for i in range(mask_w):
        for j in range(mask_y):
            im_masked[start[0] + i*depts : start[0]+(i+1)*depts, start[1] + j*depts : start[1]+(j+1)*depts] = masking[i,j]
    im2 = Image.fromarray(im_masked.astype("uint8"))
    im2.save(dir_path+'/input/'+file_name)


def readFile():
    path = dir_path + '/y'
    files = os.listdir(path)
    for file in files:
        masking(file, path, 10)
    

readFile()


