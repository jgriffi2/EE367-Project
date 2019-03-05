from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


# def discriminator(image, options, reuse=False, name="discriminator"):
#
#     with tf.variable_scope(name):
#         # image is 256 x 256 x input_c_dim
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse is False
#
#         h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
#         # h0 is (128 x 128 x self.df_dim)
#         h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
#         # h1 is (64 x 64 x self.df_dim*2)
#         h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
#         # h2 is (32x 32 x self.df_dim*4)
#         h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
#         # h3 is (32 x 32 x self.df_dim*8)
#         h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
#         # h4 is (32 x 32 x 1)
#         return h4

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
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim, s=3, name='d_h1_conv'), 'd_bn1'))
        # h1 is (H/2 x W/2 x self.df_dim)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*2, ks=3, s=1, name='d_h2_conv'), 'd_bn2'))
        # h2 is (H/2 x W/2 x self.df_dim*2)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*2, ks=3, name='d_h3_conv'), 'd_bn3'))
        # h3 is (H/4 x W/4 x self.df_dim*2)
        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim*4, ks=3, s=1, name='d_h4_conv'), 'd_bn4'))
        # h4 is (H/4 x W/4 x self.df_dim*4)
        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim*4, ks=3, name='d_h5_conv'), 'd_bn5'))
        # h5 is (H/8 x W/8 x self.df_dim*4)
        h6 = lrelu(instance_norm(conv2d(h5, options.df_dim*8, ks=3, s=1, name='d_h6_conv'), 'd_bn6'))
        # h6 is (H/8 x W/8 x self.df_dim*8)
        h7 = lrelu(instance_norm(conv2d(h6, options.df_dim*8, ks=3, name='d_h7_conv'), 'd_bn7'))
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
            y = prelu_tf(instance_norm(conv2d(x, dim, ks, s, name=name+'_c1'), name+'_bn1'), name=name+'_prelu1')
            y = instance_norm(conv2d(x, dim, ks, s, name=name+'_c2'), name+'_bn2')
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

        c2 = instance_norm(conv2d(r16, 64, s=1, name='g_e2_c'), 'g_e2_bn')
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


def generator_unet(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

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
