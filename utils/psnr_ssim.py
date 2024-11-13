import numpy
import math
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np


def PSNR(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def SSIM(img1, img2):
    return ssim(img1, img2, multichannel=True)


if __name__ == '__main__':

    original_path = '../other_img/foggyHouse.jpg'
    dehaze_path = '../model/black_channel/dehazeing/foggyHouse.jpg'
    # dehaze_path = '../model/FFA-Net/dehazeing/foggyHouse_FFA_inference.png'

    original = cv2.imread(original_path)  # numpy.adarray
    contrast = cv2.imread(dehaze_path)

    psnrValue = PSNR(original, contrast)
    ssimValue = SSIM(original, contrast)
    print('原图与去雾图的psnr比值', psnrValue)
    print('原图与去雾图的ssim比值', ssimValue)

