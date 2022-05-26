from __future__ import absolute_import
from __future__ import division

import os
from Dataset import Dataset

from GazeGAN import Gaze_GAN
from config.train_options import TrainOptions

import setproctitle
setproctitle.setproctitle("GazeGAN")
opt = TrainOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)


if __name__ == "__main__":

    print('Running...')
    dataset = Dataset(opt)
    gaze_gan = Gaze_GAN(dataset, opt)
    gaze_gan.build_model()
    # gaze_gan.build_test_model()
    gaze_gan.train()
    print('Done.')
