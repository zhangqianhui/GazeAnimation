from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def gradient_penalty(f, real, fake, mode='wgan-gp'):

    def _gradient_penalty(i_f, i_real, i_fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.compat.v1.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.compat.v1.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(i_real, i_fake)
        pred = i_f(x)
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        i_gp = tf.reduce_mean((norm - 1.)**2)

        return i_gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    else:
        raise Exception("your mode is not correct")

    return gp
