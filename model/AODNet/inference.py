import torch
import torch.optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from scipy import misc


def dehaze_image(image_name):

    data_hazy = Image.open(image_name)
    data_hazy = data_hazy.convert('RGB')
    data_hazy = np.array(data_hazy) / 255.0
    original_img = data_hazy.copy()

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.unsqueeze(0)

    dehaze_net = torch.load('./saved_models/AODNet.pth', map_location=torch.device('cpu'))

    clean_image = dehaze_net(data_hazy).detach().numpy().squeeze()
    clean_image = np.swapaxes(clean_image, 0, 1)
    clean_image = np.swapaxes(clean_image, 1, 2)
    print(type(clean_image),'++++++++++++++++++++++++')

    # clean_image = clean_image * 255.0
    # print(np.unique(clean_image),clean_image.max(), clean_image.min())
    # clean_image = Image.fromarray(clean_image)
    # clean_image.save('./1.png')

    plt.plot()
    # plt.show(clean_image)
    plt.savefig('./1.jpg',clean_image)
    plt.imshow(clean_image)
    plt.show()

    # plt.subplot(1, 2, 1)
    # plt.imshow(original_img)
    # plt.axis('off')
    # plt.title('original')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot()
    # plt.imshow(clean_image)
    # plt.axis('off')
    # plt.title('dehaze')
    # plt.show()



# def dehaze_image2(image_name):
#     img = cv2.imread(image_name)
#
#     # print(img)
#
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     data_hazy = np.array(img) / 255.0
#     original_img = data_hazy.copy()
#
#
#     # data_hazy = Image.open(image_name)
#     # data_hazy = data_hazy.convert('RGB')
#     # data_hazy = np.array(data_hazy) / 255.0
#     # original_img = data_hazy.copy()
#     #
#     data_hazy = torch.from_numpy(data_hazy).float()
#     data_hazy = data_hazy.permute(2, 0, 1)
#     data_hazy = data_hazy.unsqueeze(0)
#
#     dehaze_net = torch.load('./saved_models/AODNet.pth', map_location=torch.device('cpu'))
#
#     clean_image = dehaze_net(data_hazy).detach().numpy().squeeze()
#     clean_image = np.swapaxes(clean_image, 0, 1)
#     clean_image = np.swapaxes(clean_image, 1, 2)
#     print(type(clean_image),'++++++++++++++++++++++++')
#
#     cv2.imwrite('./1.jpg', clean_image)
#     # plt.plot()
#     # plt.imshow(clean_image)
#     # plt.show()
#     # img = Image.fromarray(clean_image)
#
#     # img.savefig('./1.jpg')
#     # plt.subplot(1, 2, 1)
#     # plt.imshow(original_img)
#     # plt.axis('off')
#     # plt.title('original')
#
#     # plt.subplot(1, 2, 2)
#     # plt.plot()
#     # plt.imshow(clean_image)
#     # plt.axis('off')
#     # plt.title('dehaze')
#     # plt.show()
#     # img = Image.fromarray(clean_image)
#     # img.save('./1.jpg')
#     # return img


if __name__ == '__main__':

    img_name = '../../images/foggy_bench.jpg'
    dehaze_image(img_name)
    # img = dehaze_image(img_name)