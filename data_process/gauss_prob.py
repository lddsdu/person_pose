import numpy as np
import cv2
import math
from pooling import avg_pool_n_times


x = 300
sigma = 10

calico = np.zeros(shape=(x, x, 1), dtype=np.float32)
calico = np.squeeze(calico)


def gauss2d(x, y, mu1=0, mu2=0, sigma1=0.03, sigma2=0.03, p=0):
    a = 2 * math.pi * sigma1 * sigma2 * math.pow(1 - p ** 2, 0.5)
    b = math.pow(math.e, (-1.0 / (2 - 2 * p ** 2)) * ((x - mu1) ** 2 / sigma1 ** 2 - 2 * p * (x-mu1) * (y - mu2) / sigma1 / sigma2 + (y - mu2) ** 2 / sigma2 ** 2))
    return 1.0 / a * b


peak = [x / 2, x / 2]
for x, line in enumerate(calico):
    for y, pixel in enumerate(line):
        data = gauss2d(x - peak[0] + 0.5, y - peak[1] + 0.5, sigma1=sigma, sigma2=sigma)
        calico[x, y] = data


max_pixel = np.max(calico)
calico = calico * 255 / max_pixel
calico = calico.astype(np.uint8)

# for line in calico:
#     print line

calico = np.expand_dims(calico, -1)
cv2.imwrite("gauss.png", calico)
# cv2.waitKey()


avg_pool_n_times()
