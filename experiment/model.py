# -*- coding: utf-8 -*-

from experiment.ops import *
from vgg import Vgg19
from tf_utils import gram_matrix
from pooling import avg_pool_n_times


class PoseGen:
    def __init__(self, opt):
        batch_size, height, width, channel, n_joint = \
            opt.batch_size, opt.height, opt.width, opt.channel, opt.n_joint
        self.batch_size, self.height, self.width, self.channel, self.n_joint = \
            batch_size, height, width, channel, n_joint

        # number of filter
        self.ngf = opt.ngf
        self.npf = opt.npf  # number of pose encoder filter

        # model
        self.norm = opt.norm
        self.vgg19 = Vgg19(trainable=False)
        self.is_training = opt.is_training
        self.generator_reuse = False
        self.discriminator_reuse = False
        self.pose_encoder_reuse = False
        self.use_sigmoid = opt.use_sigmoid

        # hyper parameters
        self.lambda_img = opt.lambda_img
        self.lambda_pose = opt.lambda_pose
        self.lambda_identity = opt.lambda_identity
        self.lambda_style = opt.lambda_style

        self.fake_image = None
        self.hat_image = None
        self.real_image_pose_map = None
        self.fake_image_pose_map = None
        self.hat_image_pose_map = None

    def build(self):
        with tf.name_scope("input"):
            # 输入为原始的image, pose信息
            original_image = tf.placeholder(tf.float32,
                                            shape=(self.batch_size, self.height, self.width, self.channel),
                                            name="original_image")
            from_pose_map = tf.placeholder(tf.float32,
                                           shape=(self.batch_size, self.height, self.width, self.n_joint),
                                           name="from_pose_map")
            to_pose_map = tf.placeholder(tf.float32,
                                         shape=(self.batch_size, self.height, self.width, self.n_joint),
                                         name="to_pose_map")
            # 为了维护训练的稳定性，fake_image从ImagePool中直接获取
            fake_image = tf.placeholder(tf.float32,
                                        shape=(self.batch_size, self.height, self.width, self.channel),
                                        name="fake_image")

        input = tf.concat([original_image, to_pose_map], axis=3)
        self.real_image_pose_map = self.pose_encoder(original_image)

        # first stage
        self.fake_image = self.generator(input, "generator")
        self.fake_image_pose_map = self.pose_encoder(self.fake_image)

        # second stage
        fake_input = tf.concat([self.fake_image, from_pose_map], axis=3)
        self.hat_image = self.generator(fake_input, "generator")
        self.hat_image_pose_map = self.pose_encoder(self.hat_image)

        # losses
        # identity loss = content_loss + patch_style_loss
        ct_loss = self.content_loss(original_image, self.hat_image)         # content loss
        patch_style_loss = self.patch_style_loss(self.vgg19.build(original_image),
                                                 from_pose_map,
                                                 self.vgg19.build(fake_image),
                                                 self.fake_image_pose_map)
        l_identity = ct_loss + self.lambda_style * patch_style_loss

        # pose loss = real + fake + hat
        real_pose_loss = self.pose_loss(from_pose_map, self.real_image_pose_map)
        fake_pose_loss = self.pose_loss(to_pose_map, self.fake_image_pose_map)
        hat_pose_loss = self.pose_loss(from_pose_map, self.hat_image_pose_map)
        l_pose = self.lambda_pose * (real_pose_loss + fake_pose_loss + hat_pose_loss)

        # discriminator loss
        d_loss = self.discriminator_loss(original_image, fake_image) + self.discriminator_loss(original_image, self.hat_image)
        # generator loss
        g_loss = self.generator_loss(self.fake_image) + self.generator_loss(self.hat_image)

        loss_total = l_identity + l_pose

        return original_image, from_pose_map, to_pose_map, fake_image, loss_total, d_loss, g_loss

    def generator(self, bottom, name="generator"):
        """ generator component,
        网络进行4次conv, 然后是８层res block, 最后进行三层 conv_transpose, 最后恢复"""
        with tf.variable_scope(name, reuse=self.generator_reuse):
            # down
            c7s1_32 = c7s1_k(bottom, self.ngf, is_training=self.is_training, norm=self.norm, name="c7s1_32", reuse=self.generator_reuse)
            d64 = dk(c7s1_32, 2 * self.ngf, is_training=self.is_training, norm=self.norm, name="d64", reuse=self.generator_reuse)
            d128 = dk(d64, 4 * self.ngf, is_training=self.is_training, norm=self.norm, name="d128", reuse=self.generator_reuse)
            d256 = dk(d128, 8 * self.ngf, is_training=self.is_training, norm=self.norm, name="d256", reuse=self.generator_reuse)

            # 8-layer res block
            res_output = n_res_blocks(d256, reuse=self.generator_reuse, n=8)

            # up
            u128 = uk(res_output, 4 * self.ngf, is_training=self.is_training, norm=self.norm, reuse=self.generator_reuse, name="u128")
            u64 = uk(u128, 2 * self.ngf, is_training=self.is_training, norm=self.norm, reuse=self.generator_reuse, name="u64")
            u32 = uk(u64, 1 * self.ngf, is_training=self.is_training, norm=self.norm, reuse=self.generator_reuse, name="u32")
            output = c7s1_k(u32, 3, norm=None, activation="tanh", reuse=self.generator_reuse, name="output")

        self.generator_reuse = True

        return output

    def discriminator(self, bottom, name="discriminator"):
        """path discriminator

        path discriminate 将图像变换到原来的 1 / 16 * 1 / 16
        """
        with tf.variable_scope(name_or_scope=name, reuse=self.discriminator_reuse):
            c64 = Ck(bottom, 64, reuse=self.discriminator_reuse, norm=None,
                     is_training=self.is_training, name="c64")
            c128 = Ck(c64, 128, reuse=self.discriminator_reuse, norm=None,
                      is_training=self.is_training, name="c128")
            c256 = Ck(c128, 256, reuse=self.discriminator_reuse, norm=None,
                      is_training=self.is_training, name="c256")
            c512 = Ck(c256, 512, reuse=self.discriminator_reuse, norm=None,
                      is_training=self.is_training, name="c512")
            output = last_conv(c512, reuse=self.discriminator_reuse, use_sigmoid=self.use_sigmoid, name="output")
        self.discriminator_reuse = True
        return output

    def pose_encoder(self, bottom, name="encoder"):
        """使用了res block的结构"""
        with tf.variable_scope(name_or_scope=name, reuse=self.pose_encoder_reuse):
            # down
            c7s1_32 = c7s1_k(bottom, self.npf, is_training=self.is_training, norm=self.norm, name="c7s1_32", reuse=self.pose_encoder_reuse)
            d128 = dk(c7s1_32, self.npf * 2, is_training=self.is_training, norm=self.norm, name="d128", reuse=self.pose_encoder_reuse)
            d256 = dk(d128, self.npf * 4, is_training=self.is_training, norm=self.norm, name="d256", reuse=self.pose_encoder_reuse)
            d512 = dk(d256, self.npf * 8, is_training=self.is_training, norm=self.norm, name="d512")

            # res block
            res_ouput = n_res_blocks(d512, reuse=self.pose_encoder_reuse, n=3)

            # up
            u256 = uk(res_ouput, self.npf * 4, is_training=self.is_training, norm=self.norm, reuse=self.pose_encoder_reuse, name="u256")
            u128 = uk(u256, self.npf * 2, is_training=self.is_training, norm=self.norm, reuse=self.pose_encoder_reuse, name="u128")
            u64 = uk(u128, self.npf, is_training=self.is_training, norm=self.norm, reuse=self.pose_encoder_reuse, name="u64")
            # 输出为概率分布，所以将其映射到 [0, 1]空间内
            output = c7s1_k(u64, self.n_joint, activation="sigmoid", reuse=self.generator_reuse, name="output")

        self.pose_encoder_reuse = True
        return output

    def vgg_feature_extractor(self, bottom, name=None):
        vgg_feature = self.vgg19.build(bottom)
        return vgg_feature

    # losses
    def pose_loss(self, pose_target, pose_hat, name="pose_loss"):
        """
        pose loss
        compute the difference between pose_target and pose_hat
        :param pose_target: batch_size, height, width, 18(n_joint)
        :param pose_hat: batch_size, height, width, 18(n_joint)
        :param name:
        :return: difference between target and hat
        """
        with tf.name_scope(name):
            loss = tf.reduce_mean((pose_hat - pose_target) ** 2)
        return loss

    def content_loss(self, target, img_hat):
        target_feature = self.vgg_feature_extractor(target)
        img_hat_feature = self.vgg_feature_extractor(img_hat)
        loss_content = tf.reduce_mean((target_feature - img_hat_feature) ** 2)
        return loss_content

    def patch_style_loss(self, target_vgg_feature, target_pose, hat_vgg_feature, hat_pose):
        """
        path style loss to control the are of the same joint same
        给定vgg提取的特征和pose的map，求得style loss
        # Parameters:
            target_vgg_feature: batch_size, height', width', channel'
            target_pose: batch_size, height, width, 18
            hat_vgg_feature: batch_size, height', width', channel'
            hat_pose: batch_size, height, width, 18
        """
        target_pose_pooled = avg_pool_n_times(target_pose, 3)
        hat_pose_pooled = avg_pool_n_times(hat_pose, 3)
        target_pose_list = tf.split(target_pose_pooled, self.n_joint, axis=3)
        hat_pose_list = tf.split(hat_pose_pooled, self.n_joint, axis=3)

        print target_vgg_feature.shape, target_pose_pooled.shape,\
            hat_vgg_feature.shape, hat_pose_pooled.shape
        patch_loss = 0.

        for t, h in zip(target_pose_list, hat_pose_list):
            gram1 = gram_matrix(target_vgg_feature * t)
            gram2 = gram_matrix(hat_vgg_feature * h)
            patch_loss += tf.reduce_mean(((gram1 - gram2) / (self.height * self.width)) ** 2)

        return patch_loss / self.n_joint

    def generator_loss(self, fake_image):
        loss = tf.reduce_mean(tf.squared_difference(self.discriminator(fake_image), 1.))
        return loss

    def discriminator_loss(self, real_image, fake_image):
        error_real = tf.reduce_mean(tf.squared_difference(self.discriminator(real_image), 1.))
        error_fake = tf.reduce_mean(tf.square(self.discriminator(fake_image)))
        loss = (error_real + error_fake) / 2.
        return loss
