from __future__ import absolute_import
from __future__ import division


import os
from Dataset import Dataset

from GazeGAN import Gaze_GAN
from config.test_options import TestOptions

opt = TestOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)


if __name__ == "__main__":

    print('Running...')
    dataset = Dataset(opt)
    gaze_gan = Gaze_GAN(dataset, opt)
    gaze_gan.build_test_model()
    gaze_gan.test()
    print('Done.')
