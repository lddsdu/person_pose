import tensorflow as tf
import cv2
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pooling import avg_pool_n_times
from vgg_feature_extractor import Vgg19


def get_gram_matrix(feature):
    shape = tf.shape(feature)
    feature_flat_hw = tf.reshape(feature, [shape[0], -1, shape[-1]])
    feature_flat_hw_t = tf.transpose(feature_flat_hw, [0, 2, 1])
    gram_matrix = tf.matmul(feature_flat_hw_t, feature_flat_hw)
    return gram_matrix


def main():
    with tf.name_scope("input"):
        image = tf.placeholder(tf.float32, shape=(1, 300, 200, 3), name="image")
        segment = tf.placeholder(tf.float32, shape=(1, 300, 200, 1), name="segment")

    image_feature = Vgg19(vgg19_npy_path="/dl_data/dressyou-about/Augment-InstaGAN-Tensorflow/vgg-weights/vgg19.npy",
                          trainable=False).build(image)
    segment_pool = avg_pool_n_times(segment, 2)

    print image_feature.shape
    print segment_pool.shape
    feature = image_feature * segment_pool
    gram = get_gram_matrix(feature)

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    img = np.expand_dims(cv2.imread("short.png"), 0) / 127.5 - 1.
    seg = np.expand_dims(cv2.imread("segment.png")[:, :, 0:1], 0) / 255.
    g = session.run(gram, feed_dict={image: img, segment: seg})
    print g.shape
    print g

    g2 = g.copy()
    g2 = g2 * 0.1
    print np.sum(((g2 - g) / 75 / 50) ** 2)


if __name__ == '__main__':
    main()
