# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import os

from vgg_feature_extractor import Vgg19
from vgg_feature_contrast import get_gram_matrix
from pooling import avg_pool_n_times
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def build_model():
    with tf.name_scope("input"):
        image = tf.placeholder(tf.float32, shape=(1, 300, 300, 3), name="image")
        segment = tf.placeholder(tf.float32, shape=(1, 300, 300, 1), name="segment")

    image_feature = Vgg19(vgg19_npy_path="/dl_data/dressyou-about/Augment-InstaGAN-Tensorflow/vgg-weights/vgg19.npy",
                          trainable=False).build(image)
    segment_pool = avg_pool_n_times(segment, 2)
    feature_matrix = segment_pool * image_feature
    gram_mat = get_gram_matrix(feature_matrix)
    return image, segment, gram_mat


def gen_mask(image):
    image = np.sum(image, 2, keepdims=True)
    seg = np.zeros(image.shape)
    image = np.squeeze(image)
    threshould = 255 * 3 - 10
    for x, line in enumerate(image):
        for y, pixel in enumerate(line):
            if pixel < threshould:
                seg[x, y, 0] = 1
    seg = seg * 255.
    return seg.astype(dtype=np.uint8)


def mask():
    image_filename = ["0{}.jpg".format(x) for x in range(1, 5)]
    for f in image_filename:
        img = cv2.imread(os.path.join("image", f))
        print img.shape
        mask = gen_mask(img)
        cv2.imwrite(os.path.join("image", f[:-4] + "_seg.jpg"), mask)


if __name__ == '__main__':
    mask()
    image_placeholder, segment_placeholder, gram_mat = build_model()

    image_filename = ["0{}.jpg".format(x) for x in range(1, 5)]
    image_seg_filename = ["0{}_seg.jpg".format(x) for x in range(1, 5)]

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    gram_mats = []
    for f, g in zip(image_filename, image_seg_filename):
        print f
        print g
        img = cv2.imread(os.path.join("image", f))
        seg = cv2.imread(os.path.join("image", g))
        img = cv2.resize(img, (300, 300)) / 127.5 - 1.
        seg = cv2.resize(seg, (300, 300))
        seg = seg[:, :, 0:1] / 255.
        img = np.expand_dims(img, 0)
        seg = np.expand_dims(seg, 0)

        gm = session.run(gram_mat, feed_dict={image_placeholder: img, segment_placeholder: seg})
        gram_mats.append(gm)

    print gram_mats[0].shape

    rele = []
    for i in range(4):
        x = gram_mats[i]
        rele.append([])
        for j in range(4):
            y = gram_mats[j]
            v = np.sum(((x - y) / (75 * 50)) ** 2)
            rele[-1].append(v)

    for l in rele:
        print l
