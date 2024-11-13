import cv2
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':


    img_path = '../../images/foggy_bench.jpg'
    img = cv2.imread(img_path)

    start_time = time.time()

    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    blue_equ = cv2.equalizeHist(blue)
    green_equ = cv2.equalizeHist(green)
    red_equ = cv2.equalizeHist(red)
    equ = cv2.merge([blue_equ, green_equ, red_equ])

    end_time = time.time()

    print('消耗时间{}'.format(end_time-start_time))
    # cv2.imshow("1",img)
    cv2.imshow("2",equ)
    # plt.figure("原始图像直方图")
    # plt.hist(img.ravel(), 256)
    # plt.figure("均衡化图像直方图")
    # plt.hist(equ.ravel(), 256)
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
    name = img_path.split('/')[-1]
    cv2.imwrite('./dehazeing/'+name, equ)
