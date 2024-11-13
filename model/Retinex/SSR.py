# SSR
import cv2
from numpy import nonzero,zeros
from numpy import float32
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import csv
import time

def replaceZeroes(data):
    min_nonzero = min(data[nonzero(data)])##data中不为0数字的位置中的最小值
    data[data == 0] = min_nonzero##data中为0的位置换为最小值
    return data


def SSR(img, sigma):
    B, G, R = cv2.split(img)
    def channel(C):
        L_C = cv2.GaussianBlur(C, (5, 5), sigma)##L(x,y)=I(x,y)∗G(x,y)
        h, w = C.shape[:2]
        C = replaceZeroes(C)
        C = C.astype(float32) / 255
        L_C = replaceZeroes(L_C)
        L_C = L_C.astype(float32) / 255
        dst_C = cv2.log(C)  ##logI(x,y)
        dst_L_C = cv2.log(L_C)  ##logL(x,y)
        log_R_C = cv2.subtract(dst_C, dst_L_C)  ##logR(x,y)=logI(x,y)−logL(x,y)
        minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(log_R_C)  ##量化处理
        for i in range(h):
            for j in range(w):
                log_R_C[i, j] = (log_R_C[i, j] - minvalue) * 255.0 / (maxvalue - minvalue)  ##R(x,y)=(value-min)(255-0)/(max-min)
        C_uint8 = cv2.convertScaleAbs(log_R_C)
        return C_uint8

    B_uint8 = channel(B)
    G_uint8 = channel(G)
    R_uint8 = channel(R)

    image = cv2.merge((B_uint8, G_uint8, R_uint8))
    return image


if __name__ == '__main__':

    img_path = '../../images/'
    img = cv2.imread(img_path)
    start_time = time.time()
    img = SSR(img,60)
    end_time = time.time()
    print('转换消耗时间{}'.format(end_time-start_time))
    name = img_path.split('/')[-1]
    cv2.imwrite('./dehazeing/ssr/'+name, img)

    # cv2.imshow('1', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('111')