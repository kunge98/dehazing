import cv2
from numpy import nonzero,zeros
from numpy import float32
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import csv
import numpy as np


def replaceZeroes(data):
    min_nonzero = min(data[nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=None, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))#求取C
    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSR(img, sigma_list):
    B, G, R = cv2.split(img)
    weight = 1 / 3.0
    scales_size = 3

    def channel(C, sigma_list):
        for i in range(0, scales_size):
            C = replaceZeroes(C)
            C = C.astype(float32) / 255
            L_C = cv2.GaussianBlur(C, (5, 5), sigma_list[i])##L(x,y)=I(x,y)∗G(x,y)
            #print(sigma_list[i])
            h, w = C.shape[:2]
            log_R_C = zeros((h, w), dtype=float32)
            L_C = replaceZeroes(L_C)
            L_C = L_C.astype(float32) / 255
            log_C = cv2.log(C)##logI(x,y)
            log_L_C = cv2.log(L_C)##logL(x,y)
            log_R_C += weight * cv2.subtract(log_C, log_L_C)##=logR(x,y)=w(logI(x,y)−logL(x,y))

        minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(log_R_C)
        for i in range(h):
            for j in range(w):
                log_R_C[i, j] = (log_R_C[i, j] - minvalue) * 255.0 / (maxvalue - minvalue)  ##R(x,y)=(value-min)(255-0)/(max-min)

        C_uint8 = cv2.convertScaleAbs(log_R_C)
        return C_uint8

    B_uint8 = channel(B, sigma_list)
    G_uint8 = channel(G, sigma_list)
    R_uint8 = channel(R, sigma_list)

    image = cv2.merge((B_uint8, G_uint8, R_uint8))
    return image


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0
    img_retinex = MSR(img, sigma_list)#先做MSR处理
    img_color = colorRestoration(img, alpha, beta)#计算色彩恢复C
    img_msrcr = G * (img_retinex * img_color + b)#MSRCR处理

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255 #转换为实数域

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))#图片格式恢复
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)#色彩平衡处理

    return img_msrcr


if __name__ == '__main__':

    img_path = '../images/trees.png'
    img = cv2.imread(img_path)
    img = MSRCR(img,[15,80,250])

    name = img_path.split('/')[-1]
    cv2.imwrite('./dehazeing/msrcr/'+name, img)

    # cv2.imshow('1', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('111')
