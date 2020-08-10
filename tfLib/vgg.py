from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

class Vgg(object):

    def __init__(self):
        self.content_layer_name = ["vgg_16/conv5/conv5_3"]
        self.style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                        "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

    def content_loss(self, endpoints_mixed, content_layers):

        loss = 0
        for layer in content_layers:
            feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
            size = tf.size(feat_a)
            loss += tf.nn.l2_loss(feat_a - feat_b) * 2 / tf.to_float(size)

        return loss

    def style_loss(self, endpoints_mixed, style_layers):

        loss = 0
        for layer in style_layers:
            feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
            size = tf.size(feat_a)
            loss += tf.nn.l2_loss(
                self.gram(feat_a) - self.gram(feat_b)) * 2 / tf.to_float(size)

        return loss

    def gram(self, layer):

        shape = tf.shape(layer)
        num_images = shape[0]
        width = shape[1]
        height = shape[2]
        num_filters = shape[3]
        features = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
        denominator = tf.to_float(width * height * num_filters)
        grams = tf.matmul(features, features, transpose_a=True) / denominator

        return grams

    def vgg_content_loss(self, fake, real):
        """
        build the sub graph of perceptual matching network
        return:
            c_loss: content loss
            s_loss: style loss
        """
        _, endpoints_mixed = self.vgg_16(tf.concat([fake, real], 0))
        c_loss = self.content_loss(endpoints_mixed, self.content_layer_name)

        return c_loss

    def vgg_style_loss(self, fake, real):
        """
        build the sub graph of perceptual matching network
        return:
            c_loss: content loss
        """
        _, endpoints_mixed = self.vgg_16(tf.concat([fake, real], 0))
        s_loss = self.style_loss(endpoints_mixed, self.style_layers)

        return s_loss

    def percep_loss(self, fake, real):
        return self.vgg_style_loss(fake, real) + self.vgg_content_loss(fake, real)

    def vgg_16(self, inputs, scope='vgg_16'):

        # repeat_net = functools.partial(slim.repeat, )
        with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                    outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Convert end_points_collection into a end_point dict
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

        return net, end_points


