from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from Dataset import save_images
import functools
from tfLib.ops import *
from tfLib.ops import instance_norm as IN
from tfLib.vgg import Vgg
from tfLib.advloss import *
from tfLib.loss import L1
import tensorflow.contrib.slim as slim
import os
import random

class Gaze_GAN(object):

    # build model
    def __init__(self, dataset, opt):

        self.dataset = dataset
        self.opt = opt
        # placeholder
        self.x_left_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.x_right_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.y_left_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.y_right_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])

        self.x = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        self.xm = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.output_nc])
        self.y = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        self.ym = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])

        self.alpha = tf.placeholder(tf.float32, [self.opt.batch_size, 1])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

        self.vgg = Vgg()

    def build_model(self):

        def build_x_model():
            
            xc = self.x * (1 - self.xm)  # corrputed images
            xl_left, xl_right = self.crop_resize(self.x, self.x_left_p, self.x_right_p)
            xl_left_fp = self.Gr(xl_left, use_sp=False)
            xl_right_fp = self.Gr(xl_right, use_sp=False)
            xl_fp_content = tf.concat([xl_left_fp, xl_right_fp], axis=-1)
            xo = self.Gx(xc, self.xm, xl_fp_content, use_sp=False)

            return xc, xl_left, xl_right, xo

        def build_y_model():

            yc = self.y * (1 - self.ym)
            yl_left, yl_right = self.crop_resize(self.y, self.y_left_p, self.y_right_p)
            yl_fp = self.encode(tf.concat([yl_left, yl_right], axis=-1))
            yl_content_left = self.Gr(yl_left, use_sp=False)
            yl_content_right = self.Gr(yl_right, use_sp=False)
            yl_content = tf.concat([yl_content_left, yl_content_right], axis=-1)
            yo = self.Gy(yc, self.ym, yl_fp, yl_content, use_sp=False)

            yo_left, yo_right = self.crop_resize(yo, self.y_left_p, self.y_right_p)
            yo_fp = self.encode(tf.concat([yo_left, yo_right], axis=-1))

            yo_content_left = self.Gr(yo_left, use_sp=False)
            yo_content_right = self.Gr(yo_right, use_sp=False)

            yo_content = tf.concat([yo_content_left, yo_content_right], axis=-1)

            return yc, yl_left, yl_right, yl_fp, yl_content, yo, yo_fp, yo_content

        def build_yx_model():
            
            y2x = self.Gx(self.yc, self.ym, self.yl_content, use_sp=False)  # output
            y2x_left, y2x_right = self.crop_resize(y2x, self.y_left_p, self.y_right_p)

            y2x_content_left = self.Gr(y2x_left, use_sp=False)
            y2x_content_right = self.Gr(y2x_right, use_sp=False)
            
            #
            y2x_content = tf.concat([y2x_content_left, y2x_content_right], axis=-1)

            y2x_fp = self.encode(tf.concat([y2x_left, y2x_right], axis=-1))
            # Learn the angle related features
            y2x_ = self.Gy(self.yc, self.ym, y2x_fp, y2x_content, use_sp=False)

            y2x_left_, y2x_right_ = self.crop_resize(y2x_, self.y_left_p, self.y_right_p)

            y2x_content_left_ = self.Gr(y2x_left_, use_sp=False)
            y2x_content_right_ = self.Gr(y2x_right_, use_sp=False)

            y2x_content_ = tf.concat([y2x_content_left_, y2x_content_right_], axis=-1)

            y2x_fp_ = self.encode(tf.concat([y2x_left_, y2x_right_], axis=-1))

            return y2x, y2x_left, y2x_right, y2x_fp, y2x_content, y2x_, y2x_fp_, y2x_content_

        self.xc, self.xl_left, self.xl_right, self.xo = build_x_model()
        self.yc, self.yl_left, self.yl_right, self.yl_fp, \
                self.yl_content, self.yo, self.yo_fp, self.yo_content = build_y_model()

        self.y2x, self.y2x_left, self.y2x_right, self.y2x_fp, self.y2x_content, self.y2x_, \
                            self.y2x_fp_, self.y2x_content_ = build_yx_model()

        self._xl_left, self._xl_right = self.crop_resize(self.xo, self.x_left_p, self.x_right_p)
        self._yl_left, self._yl_right = self.crop_resize(self.yo, self.y_left_p, self.y_right_p)
        self._y2x_left_, self._y2x_right_ = self.crop_resize(self.y2x_, self.y_left_p, self.y_right_p)

        self.dx_logits = self.D(self.x, self.xl_left, self.xl_right, scope='Dx')
        self.gx_logits = self.D(self.xo, self._xl_left, self._xl_right, scope='Dx')

        self.dy_logits = self.D(self.y, self.yl_left, self.yl_right, scope='Dy')
        self.gy_logits = self.D(self.yo, self._yl_left, self._yl_right, scope='Dy')

        # self.dyx_logits = self.D(self.x, self.y2x_left, self.y2x_right)
        self.gyx_logits = self.D(self.y2x_, self._y2x_left_, self._y2x_right_, scope='Dx')
        d_loss_fun, g_loss_fun = get_adversarial_loss(self.opt.loss_type)

        self.dx_gan_loss = d_loss_fun(self.dx_logits, self.gx_logits)
        self.gx_gan_loss = g_loss_fun(self.gx_logits)

        self.dy_gan_loss = d_loss_fun(self.dy_logits, self.gy_logits)
        self.gy_gan_loss = g_loss_fun(self.gy_logits)

        self.dyx_gan_loss = d_loss_fun(self.dx_logits, self.gyx_logits)
        self.gyx_gan_loss = g_loss_fun(self.gyx_logits)

        self.recon_loss_x = self.Local_L1(self.xo, self.x)
        self.recon_loss_y = self.Local_L1(self.yo, self.y)
        self.recon_loss_y_angle = self.Local_L1(self.y2x, self.y2x_)

        self.percep_loss_x = self.vgg.percep_loss(self.xl_left, self._xl_left) \
                             + self.vgg.percep_loss(self.xl_right, self._xl_right)

        self.percep_loss_y = self.vgg.percep_loss(self.yl_left, self._yl_left) + \
                             self.vgg.percep_loss(self.yl_right, self._yl_right) + \
                             self.vgg.percep_loss(self.y2x_left, self._y2x_left_) + \
                             self.vgg.percep_loss(self.y2x_right, self._y2x_right_)

        # fp loss
        self.recon_fp_content = L1(self.y2x_content, self.y2x_content_) + L1(self.yl_content, self.yo_content)

        # self.recon_fp_angle = L1(self.y2x_fp, self.y2x_fp_) + L1(self.yl_fp, self.yo_fp)
        self.Dx_loss = self.dx_gan_loss + self.dyx_gan_loss
        self.Dy_loss = self.dy_gan_loss
        self.Gx_loss = self.gx_gan_loss + self.opt.lam_r * self.recon_loss_x + self.opt.lam_p * self.percep_loss_x
        self.Gy_loss = self.gy_gan_loss + self.gyx_gan_loss + self.opt.lam_r * self.recon_loss_y \
                       + self.opt.lam_r * self.recon_loss_y_angle + self.recon_fp_content + self.opt.lam_p * self.percep_loss_y

    def build_test_model(self):

        def build_x_model():
            xc = self.x * (1 - self.xm)  # corrputed images
            xl_left, xl_right = self.crop_resize(self.x, self.x_left_p, self.x_right_p)
            xl_left_fp = self.Gr(xl_left, use_sp=False)
            xl_right_fp = self.Gr(xl_right, use_sp=False)
            xl_fp_content = tf.concat([xl_left_fp, xl_right_fp], axis=-1)
            xl_fp = self.encode(tf.concat([xl_left, xl_right], axis=-1))
            xo = self.Gx(xc, self.xm, xl_fp_content, use_sp=False)

            return xc, xl_left, xl_right, xl_fp, xl_fp_content, xo

        def build_y_model():

            yc = self.y * (1 - self.ym)
            yl_left, yl_right = self.crop_resize(self.y, self.y_left_p, self.y_right_p)
            yl_fp = self.encode(tf.concat([yl_left, yl_right], axis=-1))
            yl_content_left = self.Gr(yl_left, use_sp=False)
            yl_content_right = self.Gr(yl_right, use_sp=False)
            yl_content = tf.concat([yl_content_left, yl_content_right], axis=-1)
            yo = self.Gy(yc, self.ym, yl_fp, yl_content, use_sp=False)

            return yc, yl_left, yl_right, yl_fp, yl_content, yo

        def build_yx_model():

            y2x = self.Gx(self.yc, self.ym, self.yl_content, use_sp=False)  # output
            y2x_left, y2x_right = self.crop_resize(y2x, self.y_left_p, self.y_right_p)

            y2x_content_left = self.Gr(y2x_left, use_sp=False)
            y2x_content_right = self.Gr(y2x_right, use_sp=False)

            y2x_content = tf.concat([y2x_content_left, y2x_content_right], axis=-1)
            y2x_fp = self.encode(tf.concat([y2x_left, y2x_right], axis=-1))

            # Learn the angle related features
            y2x_ = self.Gy(self.yc, self.ym, y2x_fp, y2x_content, use_sp=False)

            return y2x, y2x_left, y2x_right, y2x_fp, y2x_content, y2x_

        self.xc, self.xl_left, self.xl_right, self.xl_fp, self.xl_content, self.xo = build_x_model()
        self.yc, self.yl_left, self.yl_right, self.yl_fp, self.yl_content, self.yo = build_y_model()

        self._xl_left, self._xl_right = self.crop_resize(self.xo, self.x_left_p, self.x_right_p)
        self._yl_left, self._yl_right = self.crop_resize(self.yo, self.y_left_p, self.y_right_p)

        yo_content_left = self.Gr(self._yl_left, use_sp=False)
        yo_content_right = self.Gr(self._yl_right, use_sp=False)

        self.yo_content = tf.concat([yo_content_left, yo_content_right], axis=-1)
        self.y2x, self.y2x_left, self.y2x_right, self.y2x_fp, self.y2x_content, self.y2x_ = build_yx_model()

        self._y2x_left, self._y2x_right = self.crop_resize(self.y2x_, self.x_left_p, self.x_right_p)
        y2x_content_left_ = self.Gr(self._y2x_left, use_sp=False)
        y2x_content_right_ = self.Gr(self._y2x_right, use_sp=False)
        self.y2x_content_ = tf.concat([y2x_content_left_, y2x_content_right_], axis=-1)

        self.y2x_fp_inter = self.y2x_fp * self.alpha + (1 - self.alpha) * self.yl_fp
        self.y2x_content_inter = self.y2x_content * self.alpha + (1 - self.alpha) * self.yl_content
        self._y2x_inter = self.Gy(self.yc, self.ym, self.y2x_fp_inter, self.y2x_content_inter, use_sp=False)

    def crop_resize(self, input, boxes_left, boxes_right):

        shape = [int(item) for item in input.shape.as_list()]
        return tf.image.crop_and_resize(input, boxes=boxes_left, box_ind=list(range(0, shape[0])),
                                        crop_size=[int(shape[-3] / 4), int(shape[-2] / 4)]), \
               tf.image.crop_and_resize(input, boxes=boxes_right, box_ind=list(range(0, shape[0])),
                                        crop_size=[int(shape[-3] / 4), int(shape[-2] / 4)])

    def Local_L1(self, l1, l2):
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(l1 - l2), axis=[1, 2, 3])
                              / (self.opt.crop_w * self.opt.crop_h * self.opt.output_nc))
        return loss

    def test(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            print('Load checkpoint')
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Succeed!')
            else:
                print('Do not exists any checkpoint, Load Failed!')
                exit()

            trainbatch, trainmask, _, _, testbatch, testmask = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = self.opt.test_num
            for j in range(batch_num):
                x_img, x_img_pos, y_img, y_img_pos = sess.run([trainbatch, trainmask, testbatch, testmask])
                x_m, x_left_pos, x_right_pos = self.get_Mask_and_pos(x_img_pos)
                y_m, y_left_pos, y_right_pos = self.get_Mask_and_pos(y_img_pos)

                f_d = {self.x: x_img,
                       self.xm: x_m,
                       self.x_left_p: x_left_pos,
                       self.x_right_p: x_right_pos,
                       self.y: y_img,
                       self.ym: y_m,
                       self.y_left_p: y_left_pos,
                       self.y_right_p: y_right_pos
                       }

                output = sess.run([self.x, self.xc, self.xo, self.yc,
                                   self.y, self.yo, self.y2x, self.y2x_], feed_dict=f_d)
                output_concat = self.Transpose(np.array([output[0], output[1], output[2],
                                                         output[3], output[4], output[5], output[6], output[7]]))
                local_output = sess.run([self.xl_left, self.xl_right, self.yl_left, self.yl_right,
                                         self._xl_left, self._xl_right, self._yl_left, self._yl_right, self.y2x_left,
                                         self.y2x_right], feed_dict=f_d)
                local_output_concat = self.Transpose(
                    np.array([local_output[0], local_output[1], local_output[2], local_output[3],
                              local_output[4], local_output[5], local_output[6], local_output[7],
                              local_output[8], local_output[9]]))

                inter_results = [y_img, np.ones(shape=[self.opt.batch_size,
                                                       self.opt.img_size, self.opt.img_size, 3])]
                inter_results1 = [y_img, np.ones(shape=[self.opt.batch_size,
                                                        self.opt.img_size, self.opt.img_size, 3])]
                inter_results2 = [y_img, np.ones(shape=[self.opt.batch_size,
                                                        self.opt.img_size, self.opt.img_size, 3])]
                inter_results3 = []

                for i in range(0, 11):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results.append(output)

                for i in range(0, 15):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results1.append(output)

                for i in range(11, 22):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results2.append(output)

                for i in range(-10, 0):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results3.append(output)

                save_images(output_concat,
                            '{}/{:02d}.jpg'.format(self.opt.test_sample_dir, j))
                save_images(local_output_concat,
                            '{}/{:02d}_local.jpg'.format(self.opt.test_sample_dir, j))
                save_images(self.Transpose(np.array(inter_results)),
                            '{}/{:02d}inter1.jpg'.format(self.opt.test_sample_dir, j))
                save_images(self.Transpose(np.array(inter_results1)),
                            '{}/{:02d}inter1_1.jpg'.format(self.opt.test_sample_dir, j))

            coord.request_stop()
            coord.join(threads)

    def train(self):

        self.t_vars = tf.trainable_variables()
        self.dx_vars = [var for var in self.t_vars if 'Dx' in var.name]
        self.dy_vars = [var for var in self.t_vars if 'Dy' in var.name]
        self.gx_vars = [var for var in self.t_vars if 'Gx' in var.name]
        self.gy_vars = [var for var in self.t_vars if 'Gy' in var.name]
        self.e_vars = [var for var in self.t_vars if 'encode' in var.name]
        self.gr_vars = [var for var in self.t_vars if 'Gr' in var.name]
        self.vgg_vars = [var for var in self.t_vars if 'vgg_16' in var.name]

        assert len(self.t_vars) == len(self.dx_vars + self.dy_vars + self.gx_vars
                                      + self.gy_vars + self.e_vars + self.gr_vars + self.vgg_vars)

        self.saver = tf.train.Saver(max_to_keep=10)
        self.p_saver = tf.train.Saver(self.gr_vars)
        opti_Dx = tf.train.AdamOptimizer(self.opt.lr_d * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Dx_loss, var_list=self.dx_vars)
        opti_Dy = tf.train.AdamOptimizer(self.opt.lr_d * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Dy_loss, var_list=self.dy_vars)
        opti_Gx = tf.train.AdamOptimizer(self.opt.lr_g * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Gx_loss, var_list=self.gx_vars)
        opti_Gy = tf.train.AdamOptimizer(self.opt.lr_g * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Gy_loss, var_list=self.gy_vars + self.e_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            start_step = 0

            variables_to_restore = slim.get_variables_to_restore(include=['vgg_16'])
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.opt.vgg_path)

            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                try:
                    ckpt = tf.train.get_checkpoint_state(self.opt.pretrain_path)
                    print(ckpt)
                    self.p_saver.restore(sess, ckpt.model_checkpoint_path)
                except:
                    print(" PAM ckpt path may not be correct")

            step = start_step
            lr_decay = 1

            print("Start read dataset")
            train_images_x, train_eye_pos_x, train_images_y, train_eye_pos_y, \
                                test_images, test_eye_pos = self.dataset.input()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            real_test_batch, real_test_pos = sess.run([test_images, test_eye_pos])

            while step <= self.opt.niter:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.opt.niter - step) / float(self.opt.niter - 20000)

                x_data, x_p_data = sess.run([train_images_x, train_eye_pos_x])
                y_data, y_p_data = sess.run([train_images_y, train_eye_pos_y])

                flip_random = random.uniform(0, 1)
                if flip_random > 0.5:
                    x_data = np.flip(x_data, axis=2)
                    y_data = np.flip(y_data, axis=2)
                    x_p_data[:,0] = self.opt.img_size - x_p_data[:, 0]
                    x_p_data[:,2] = self.opt.img_size - x_p_data[:, 2]

                xm_data, x_left_p_data, x_right_p_data = self.get_Mask_and_pos(x_p_data)
                ym_data, y_left_p_data, y_right_p_data = self.get_Mask_and_pos(y_p_data)

                f_d = {self.x: x_data,
                       self.xm: xm_data,
                       self.x_left_p: x_left_p_data,
                       self.x_right_p: x_right_p_data,
                       self.y: y_data,
                       self.ym: ym_data,
                       self.y_left_p: y_left_p_data,
                       self.y_right_p: y_right_p_data,
                       self.lr_decay: lr_decay}

                sess.run(opti_Dx, feed_dict=f_d)
                sess.run(opti_Dy, feed_dict=f_d)
                sess.run(opti_Gx, feed_dict=f_d)
                sess.run(opti_Gy, feed_dict=f_d)

                if step % 500 == 0:
                    output_loss = sess.run(
                        [self.Dx_loss + self.Dy_loss, self.Gx_loss, self.Gy_loss, self.opt.lam_r * self.recon_loss_x,
                         self.opt.lam_r * self.recon_loss_y], feed_dict=f_d)
                    print(
                        "step %d D_loss=%.4f, Gx_loss=%.4f, Gy_loss=%.4f, Recon_loss_x=%.4f, Recon_loss_y=%.4f, lr_decay=%.4f" %
                        (
                            step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], output_loss[4],
                            lr_decay))

                if np.mod(step, 2000) == 0:
                    o_list = sess.run([self.xl_left, self.xl_right, self.xc, self.xo,
                                       self.yl_left, self.yl_right, self.yc, self.yo,
                                       self.y2x, self.y2x_], feed_dict=f_d)

                    batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_test_pos)
                    # for test
                    f_d = {self.x: real_test_batch, self.xm: batch_masks,
                           self.x_left_p: batch_left_eye_pos, self.x_right_p: batch_right_eye_pos,
                           self.y: real_test_batch, self.ym: batch_masks,
                           self.y_left_p: batch_left_eye_pos, self.y_right_p: batch_right_eye_pos,
                           self.lr_decay: lr_decay}

                    t_o_list = sess.run([self.xc, self.xo, self.yc, self.yo], feed_dict=f_d)
                    train_trans = self.Transpose(
                        np.array([x_data, o_list[2], o_list[3], o_list[6], o_list[7], o_list[8],
                                  o_list[9]]))
                    l_trans = self.Transpose(np.array([o_list[0], o_list[1], o_list[4], o_list[5]]))
                    test_trans = self.Transpose(np.array([real_test_batch, t_o_list[0],
                                                          t_o_list[1], t_o_list[2], t_o_list[3]]))
                    save_images(l_trans, '{}/{:02d}_lo_{}.jpg'.format(self.opt.sample_dir, step, self.opt.exper_name))
                    save_images(train_trans,
                                '{}/{:02d}_tr_{}.jpg'.format(self.opt.sample_dir, step, self.opt.exper_name))
                    save_images(test_trans,
                                '{}/{:02d}_te_{}.jpg'.format(self.opt.sample_dir, step, self.opt.exper_name))

                if np.mod(step, 20000) == 0:
                    self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
            # summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print("Model saved in file: %s" % save_path)

    def Transpose(self, list):
        refined_list = np.transpose(np.array(list), axes=[1, 2, 0, 3, 4])
        refined_list = np.reshape(refined_list, [refined_list.shape[0] * refined_list.shape[1],
                                                 refined_list.shape[2] * refined_list.shape[3], -1])
        return refined_list

    def D(self, x, xl_left, xl_right, scope='D'):

        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            xg_fp = self.global_d(x)
            xl_fp = self.local_d(tf.concat([xl_left, xl_right], axis=-1))
            # Concatenation
            ful = tf.concat([xg_fp, xl_fp], axis=1)
            ful = tf.nn.relu(fc(ful, output_size=1024, scope='fc1'))
            logits = fc(ful, output_size=1, scope='fc2')

            return logits

    def local_d(self, x):

        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("d1", reuse=tf.AUTO_REUSE):
            for i in range(self.opt.n_layers_d):
                output_dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 512)
                x = lrelu(conv2d_base(x, output_dim=output_dim, scope='d{}'.format(i)))
            x = tf.reshape(x, shape=[self.opt.batch_size, -1])
            fp = fc(x, output_size=output_dim, scope='fp')
            return fp

    def global_d(self, x):

        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("d2", reuse=tf.AUTO_REUSE):
            # Global Discriminator Dg
            for i in range(self.opt.n_layers_d):
                dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 256)
                x = lrelu(conv2d_base(x, output_dim=dim, scope='d{}'.format(i)))
            x = tf.reshape(x, shape=[self.opt.batch_size, -1])
            fp = fc(x, output_size=dim, scope='fp')

            return fp

    def Gy(self, input_x, img_mask, fp_local, fp_content, use_sp=False):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        conv2d_final = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp, output_dim=self.opt.output_nc)
        with tf.variable_scope("Gy", reuse=tf.AUTO_REUSE):

            x = tf.concat([input_x, img_mask], axis=3)
            u_fp_list = []
            x = lrelu(IN(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_g):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i + 1), 256)
                x = lrelu(IN(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='IN{}'.format(i)))
                u_fp_list.append(x)

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')
            bottleneck = tf.concat([bottleneck, fp_local, fp_content], axis=1)

            h, w = x.shape.as_list()[-3], x.shape.as_list()[-2]
            de_x = lrelu(fc(bottleneck, output_size=256 * h * w, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.opt.batch_size, h, w, 256])

            ngf = c_dim
            for i in range(self.opt.n_layers_g):
                c_dim = np.maximum(int(ngf / np.power(2, i)), 16)
                de_x = tf.concat([de_x, u_fp_list[len(u_fp_list) - (i + 1)]], axis=3)
                de_x = tf.nn.relu(IN(de_conv(de_x,
                                                        output_shape=[self.opt.batch_size, h * pow(2, i + 1),
                                                                      w * pow(2, i + 1), c_dim], use_sp=use_sp,
                                                        scope='deconv{}'.format(i)), scope='IN_{}'.format(i)))
            de_x = conv2d_final(de_x, scope='output_conv')

            return input_x + tf.nn.tanh(de_x) * img_mask

    def Gx(self, input_x, img_mask, fp, use_sp=False):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        conv2d_final = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp, output_dim=self.opt.output_nc)
        with tf.variable_scope("Gx", reuse=tf.AUTO_REUSE):

            x = tf.concat([input_x, img_mask], axis=3)
            u_fp_list = []
            x = lrelu(IN(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_g):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i + 1), 256)
                x = lrelu(IN(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='IN{}'.format(i)))
                u_fp_list.append(x)

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')

            bottleneck = tf.concat([bottleneck, fp], axis=-1)
            h, w = x.shape.as_list()[-3], x.shape.as_list()[-2]
            de_x = lrelu(fc(bottleneck, output_size=256 * h * w, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.opt.batch_size, h, w, 256])
            ngf = c_dim
            for i in range(self.opt.n_layers_g):
                c_dim = np.maximum(int(ngf / np.power(2, i)), 16)
                de_x = tf.concat([de_x, u_fp_list[len(u_fp_list) - (i + 1)]], axis=3)
                de_x = tf.nn.relu(IN(de_conv(de_x, output_shape=[self.opt.batch_size, h * pow(2, i + 1),
                                                                            w * pow(2, i + 1), c_dim], use_sp=use_sp,
                                                        scope='deconv{}'.format(i)), scope='IN_{}'.format(i)))

            de_x = conv2d_final(de_x, scope='output_conv')
            return input_x + tf.nn.tanh(de_x) * img_mask

    def Gr(self, input_x, use_sp=False):
        print(input_x.shape)
        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        with tf.variable_scope("Gr", reuse=tf.AUTO_REUSE):
            x = input_x
            x = lrelu(IN(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_r):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i + 1), 128)
                x = lrelu(IN(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='IN{}'.format(i)))

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')

            return bottleneck

    def encode(self, x):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2)
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            nef = self.opt.nef
            x = tf.nn.relu(IN(conv2d_first(x, output_dim=nef, scope='e_c1'), scope='e_in1'))
            for i in range(self.opt.n_layers_e):
                x = tf.nn.relu(IN(conv2d_base(x, output_dim=min(nef * pow(2, i + 1), 128), scope='e_c{}'.format(i + 2)),
                                  scope='e_in{}'.format(i + 2)))
            bottleneck = tf.reshape(x, [self.opt.batch_size, -1])
            content = fully_connect(bottleneck, output_size=2, scope='e_ful1')

            return content

    def get_Mask_and_pos(self, eye_pos):

        def get_pos(eye_pos):
            o_eye_pos = np.zeros(shape=(self.opt.batch_size, 4), dtype=np.int32)
            o_eye_pos[:, 3] = (eye_pos[:, 0] + self.opt.crop_w / 2)
            o_eye_pos[:, 2] = (eye_pos[:, 1] + self.opt.crop_h / 2)
            o_eye_pos[:, 1] = (eye_pos[:, 0] - self.opt.crop_w / 2)
            o_eye_pos[:, 0] = (eye_pos[:, 1] - self.opt.crop_h / 2)

            return o_eye_pos

        def get_Mask(left_eye_pos, right_eye_pos):
            batch_mask = np.zeros(shape=(self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.output_nc))
            # x, y = np.meshgrid(range(img_size), range(img_size))
            for i in range(self.opt.batch_size):
                batch_mask[i,
                left_eye_pos[i][0]:left_eye_pos[i][2],
                left_eye_pos[i][1]:left_eye_pos[i][3], :] = 1
                batch_mask[i,
                right_eye_pos[i][0]:right_eye_pos[i][2],
                right_eye_pos[i][1]:right_eye_pos[i][3], :] = 1

            return batch_mask

        left_eye_pos = get_pos(eye_pos[:, 0:2])
        right_eye_pos = get_pos(eye_pos[:, 2:4])
        mask = get_Mask(left_eye_pos, right_eye_pos)

        return mask, left_eye_pos / float(self.opt.img_size), \
               right_eye_pos / float(self.opt.img_size)




