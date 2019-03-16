# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from model import PoseGen
import argparse
import easydict
from tf_utils import summary_file_name
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


parser = argparse.ArgumentParser()

# TODO 这部分的超参将严重影响最后的效果,需要仔细的调整
# hyper parameters
parser.add_argument("--lambda_identity", default=1., type=float, help="")
parser.add_argument("--lambda_style", default=0.3, type=float, help="")
parser.add_argument("--lambda_img", default=1., type=float, help="")
parser.add_argument("--lambda_pose", default=700., type=float, help="")

# filter
parser.add_argument("--ngf", default=64, help="")
parser.add_argument("--npf", default=64, help="")

# model
parser.add_argument("--use_sigmoid", default=False, type=bool, help="")
parser.add_argument("--is_training", default=True, type=bool, help="")
parser.add_argument("--norm", default="instance", type=str, help="")

# input
parser.add_argument("--batch_size", default=10, help="")
parser.add_argument("--height", default=256, help="")
parser.add_argument("--width", default=256, help="")
parser.add_argument("--channel", default=3, help="")
parser.add_argument("--n_joint", default=18, help="")
parser.add_argument("--lr", default=2e-4, help="")


def main():
    args = parser.parse_args()
    opt = easydict.EasyDict(args.__dict__)
    pg = PoseGen(opt)

    # TODO test model
    original_image, from_pose_map, to_pose_map, fake_image, \
    loss_total, d_loss, g_loss = pg.build()

    # summary
    tf.summary.image("real_image", original_image, max_outputs=1)
    tf.summary.image("fake_image", pg.fake_image, max_outputs=1)
    tf.summary.image("history_fake_image", fake_image, max_outputs=1)
    tf.summary.image("rec_image", pg.hat_image, max_outputs=1)

    tf.summary.scalar("loss/d_loss", d_loss)
    tf.summary.scalar("loss/g_loss", d_loss)
    tf.summary.scalar("loss/loss_other", loss_total)
    summary_merged = tf.summary.merge_all()

    all_variables = tf.trainable_variables()
    encoder_variables = filter(lambda x: "encoder" in x.name, all_variables)
    generator_variables = filter(lambda x: "generator" in x.name, all_variables)
    discriminator_variables = filter(lambda x: "discriminator" in x.name, all_variables)
    print len(all_variables), len(encoder_variables), len(generator_variables), len(discriminator_variables)
    assert len(all_variables) == sum([len(x) for x in [encoder_variables, generator_variables, discriminator_variables]])

    # optimizer
    d_optm = tf.train.AdamOptimizer(learning_rate=opt.lr).minimize(d_loss, var_list=discriminator_variables)
    g_optm = tf.train.AdamOptimizer(learning_rate=opt.lr / 2.).\
        minimize(g_loss + loss_total, var_list=generator_variables + encoder_variables)

    # session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(summary_file_name())

    # train

    # infer

    # save model


if __name__ == '__main__':
    main()
