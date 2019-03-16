import math


def gauss2d(x, y, mu1=0, mu2=0, sigma1=0.03, sigma2=0.03, p=0):
    a = 2 * math.pi * sigma1 * sigma2 * math.pow(1 - p ** 2, 0.5)
    b = math.pow(math.e, (-1.0 / (2 - 2 * p ** 2)) * ((x - mu1) ** 2 / sigma1 ** 2 - 2 * p * (x-mu1) * (y - mu2) / sigma1 / sigma2 + (y - mu2) ** 2 / sigma2 ** 2))
    return 1.0 / a * b


print gauss2d(0.5, 0.5)
print gauss2d(0, 0)
print gauss2d(-0.5, -0.5)

print gauss2d(0, 0, sigma1=0.3, sigma2=0.3)

print gauss2d(0.5, 0.5, sigma1=0.3, sigma2=0.3)
print gauss2d(-0.5, -0.5, sigma1=0.3, sigma2=0.3)
print gauss2d(0.5, -0.5, sigma1=0.3, sigma2=0.3)
print gauss2d(-0.5, 0.5, sigma1=0.3, sigma2=0.3)

