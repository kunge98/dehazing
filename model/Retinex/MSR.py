# MSR
import cv2
from numpy import nonzero,zeros
from numpy import float32
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import csv



def replaceZeroes(data):
    min_nonzero = min(data[nonzero(data)])##data中不为0数字的位置中的最小值
    data[data == 0] = min_nonzero##data中为0的位置换为最小值
    return data


def MSR(img, sigma_list):
    B, G, R = cv2.split(img)
    weight = 1 / 3.0
    scales_size = 3


    def channel(C, sigma_list):
        for i in range(0, scales_size):
            C = replaceZeroes(C)
            C = C.astype(float32) / 255
            L_C = cv2.GaussianBlur(C, (5, 5), sigma_list[i])##L(x,y)=I(x,y)∗G(x,y)
            # print(sigma_list[i])
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


if __name__ == '__main__':

    img_path = '../images/trees.png'
    img = cv2.imread(img_path)
    img = MSR(img,[15,80,250])

    name = img_path.split('/')[-1]
    cv2.imwrite('./dehazeing/msr/'+name, img)

    # cv2.imshow('1', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('111')