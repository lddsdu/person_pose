import tensorflow as tf
import cv2
import numpy as np
pool_index = 0


def avg_pool(bottom, name=None):
    global pool_index
    pool_index += 1
    if name is None:
        name = "pool_{:0>2}".format(pool_index)
    ret = tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    return ret


def avg_pool_n_times(bottom, n):
    x = bottom
    for _ in range(n):
        x = avg_pool(x)
    return x


def main():
    session = tf.InteractiveSession()
    origin = tf.placeholder(tf.float32, shape=(1, 300, 300, 1), name="segment")
    origin_image = cv2.imread("gauss.png")
    origin_image = np.expand_dims(origin_image, 0)
    origin_image = origin_image[:, :, :, 0:1]
    x = origin
    for _ in range(2):
        x = avg_pool(x)

    pool_result = session.run(x, feed_dict={origin: origin_image})
    pool_result = np.squeeze(pool_result, 0)
    print pool_result.shape
    cv2.imwrite("/tmp/pool_gauss_avg.jpg", pool_result)

    pool_result = np.squeeze(pool_result)
    for line in pool_result:
        print line


if __name__ == '__main__':
    main()
