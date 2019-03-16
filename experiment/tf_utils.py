# -*- coding: utf-8 -*-
import tensorflow as tf
import time


def gram_matrix(feature):
    shape = tf.shape(feature)
    feature_flat_hw = tf.reshape(feature, [shape[0], -1, shape[-1]])
    feature_flat_hw_t = tf.transpose(feature_flat_hw, [0, 2, 1])
    gram_matrix = tf.matmul(feature_flat_hw_t, feature_flat_hw)
    return gram_matrix


def summary_file_name():
    template = "%Y-%m-%d %H:%M:%S"
    a = time.strftime(template, time.localtime())
    return a


if __name__ == '__main__':
    print summary_file_name()