"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
import copy
try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def load_test_data(image_path):
    img = imread(image_path)
    img = img/127.5 - 1
    return img

def load_train_data(image_path, low_res_size=64, high_res_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_A = np.concatenate((img_A, np.zeros((img_A.shape[0], high_res_size-low_res_size, img_A.shape[2]))), axis=1)
    img_A = np.concatenate((img_A, np.zeros((high_res_size-low_res_size, img_A.shape[1], img_A.shape[2]))), axis=0)

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (high_res_size, high_res_size, input_c_dim + output_c_dim)
    return img_AB

# -----------------------------
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.
