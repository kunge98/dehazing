import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import data, util, color
import math
import time
import os
import cv2
from PIL import Image
import numpy as np


def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


if __name__ == '__main__':

    img_path = "../../images/indoor.png"

    if os.path.isfile(img_path):
        # print("path {} is existence;".format(img_path))
        img = Image.open(img_path)
        Img = img.convert('L')
        img = np.array(img)
        # print(img, img.shape)

    start_time = time.time()
    img = homomorphic_filter(img)
    end_time = time.time()
    # print("new img shape is {}".format(img.shape))
    print("消耗时间{}".format(end_time-start_time))

    name = img_path.split('/')[-1]
    cv2.imwrite('./dehazeing/'+name, img)


