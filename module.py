from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

""" Discriminator for SRGAN """
def discriminator_SRGAN(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is H x W x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='d_h0_conv'))
        # h0 is (H x W x self.df_dim)
        # TODO: maybe use batchnorm instead of instance norm?
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim, s=3, name='d_h1_conv'), 'd_bn1'))
        # h1 is (H/2 x W/2 x self.df_dim)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim*2, ks=3, s=1, name='d_h2_conv'), 'd_bn2'))
        # h2 is (H/2 x W/2 x self.df_dim*2)
        h3 = lrelu(batch_norm(conv2d(h2, options.df_dim*2, ks=3, name='d_h3_conv'), 'd_bn3'))
        # h3 is (H/4 x W/4 x self.df_dim*2)
        h4 = lrelu(batch_norm(conv2d(h3, options.df_dim*4, ks=3, s=1, name='d_h4_conv'), 'd_bn4'))
        # h4 is (H/4 x W/4 x self.df_dim*4)
        h5 = lrelu(batch_norm(conv2d(h4, options.df_dim*4, ks=3, name='d_h5_conv'), 'd_bn5'))
        # h5 is (H/8 x W/8 x self.df_dim*4)
        h6 = lrelu(batch_norm(conv2d(h5, options.df_dim*8, ks=3, s=1, name='d_h6_conv'), 'd_bn6'))
        # h6 is (H/8 x W/8 x self.df_dim*8)
        h7 = lrelu(batch_norm(conv2d(h6, options.df_dim*8, ks=3, name='d_h7_conv'), 'd_bn7'))
        # h7 is (H/16 x W/16 x self.df_dim*8)
        h8 = lrelu(linear(h7, 1024, scope='d_lin8'))
        # h8 is (H*W/16/16 x 1024)
        h9 = sigmoid(linear(h8, 1, scope='d_lin9'))
        # h9 is (H*W/16/16 x 1)
        return h9


""" Generator for SRGAN """
def generator_SRGAN(image, options, A2B=True, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is H x W x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            # x is (H x W x dim)
            y = prelu_tf(batch_norm(conv2d(x, dim, ks, s, name=name+'_c1'), name+'_bn1'), name=name+'_prelu1')
            y = batch_norm(conv2d(x, dim, ks, s, name=name+'_c2'), name+'_bn2')
            # y is (H x W x dim)
            return y + x

        c1 = prelu_tf(conv2d(image, 64, ks=9, s=1, name='g_e1_c'), name='g_e1_prelu')
        # c1 is (H x W x 64)

        r1 = residule_block(c1, 64, name='g_r1')
        r2 = residule_block(r1, 64, name='g_r2')
        r3 = residule_block(r2, 64, name='g_r3')
        r4 = residule_block(r3, 64, name='g_r4')
        r5 = residule_block(r4, 64, name='g_r5')
        r6 = residule_block(r5, 64, name='g_r6')
        r7 = residule_block(r6, 64, name='g_r7')
        r8 = residule_block(r7, 64, name='g_r8')
        r9 = residule_block(r8, 64, name='g_r9')
        r10 = residule_block(r9, 64, name='g_r10')
        r11 = residule_block(r10, 64, name='g_r11')
        r12 = residule_block(r11, 64, name='g_r12')
        r13 = residule_block(r12, 64, name='g_r13')
        r14 = residule_block(r13, 64, name='g_r14')
        r15 = residule_block(r14, 64, name='g_r15')
        r16 = residule_block(r15, 64, name='g_r16')
        # r16 is (H x W x 64)

        c2 = batch_norm(conv2d(r16, 64, s=1, name='g_e2_c'), 'g_e2_bn')
        c2 = c2 + c1
        # c2 is (H x W x 64)

        scale = options.scale if A2B else 1 / options.scale
        if (scale >= 1):
            # c3 = prelu_tf(pixelShuffler(conv2d(c2, 256, s=1, name='g_e3_c'), scale=scale), name='g_e3_prelu')
            c3 = prelu_tf(tf.depth_to_space(conv2d(c2, 256, s=1, name='g_e3_c'), scale), name='g_e3_prelu')
            # c3 is (H*scale x W*scale x 256)
            # c4 = prelu_tf(pixelShuffler(conv2d(c3, 256, s=1, name='g_e4_c'), scale=scale), name='g_e4_prelu')
            c4 = prelu_tf(tf.depth_to_space(conv2d(c3, 256, s=1, name='g_e4_c'), scale), name='g_e4_prelu')
            # c4 is (H*scale*scale x W*scale*scale x 256)
        else:
            # c3 = prelu_tf(pixelShuffler(conv2d(c2, 256, s=1, name='g_e3_c'), scale=scale), name='g_e3_prelu')
            c3 = prelu_tf(conv2d(c2, 256, s=options.scale, name='g_e3_c'), name='g_e3_prelu')
            # c3 is (H*scale x W*scale x 256)
            # c4 = prelu_tf(pixelShuffler(conv2d(c3, 256, s=1, name='g_e4_c'), scale=scale), name='g_e4_prelu')
            c4 = prelu_tf(conv2d(c3, 256, s=options.scale, name='g_e4_c'), name='g_e4_prelu')
            # c4 is (H*scale*scale x W*scale*scale x 256)

        pred = conv2d(c4, options.output_c_dim, ks=9, s=1, name='g_pred_c')
        # pred is (H*scale*scale x W*scale*scale x options.output_c_dim)

        return pred


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def perceptual_criterion(in_, target):
    return tf.reduce_mean(tf.reduce_sum(tf.square(in_ - target), axis=[3]))


def VGG19(input, type, reuse, name, scope):
    # Define the feature to extract according to the type of perceptual
    with tf.name_scope(name) as scope_name:
        if type == 'VGG54':
            target_layer = scope + '/vgg_19/conv5/conv5_4'
        elif type == 'VGG22':
            target_layer = scope + '/vgg_19/conv2/conv2_2'
        else:
            raise NotImplementedError('Unknown perceptual type')
        _, output = vgg_19(input, is_training=False, reuse=reuse, scope=scope+'/vgg_19')
        output = output[target_layer]

        return output
