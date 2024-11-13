#!/usr/bin/env python
# encoding: utf-8
import time

from PIL import Image
import numpy as np


def haze_removal(image, windowSize=24, w0=0.4, t0=0.1):

    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    darkImage = darkImage.astype(np.double)

    t = 1 - w0 * (darkImage / maxDarkChannel)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    result = Image.fromarray(J)

    return result


if __name__ == '__main__':

    img_path = '../../images/foggy_bench.jpg'
    image = np.array(Image.open(img_path))
    imageSize = image.shape
    start_time = time.time()
    result = haze_removal(image)
    end_time = time.time()
    print("消耗时间{}".format(end_time-start_time))
    result.show()
    name = img_path.split('/')[-1]
    result.save('./dehazeing/' + name)
