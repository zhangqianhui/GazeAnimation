from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from Dataset import Dataset
from GazeGAN2 import Gaze_GAN as Gaze_GAN2
from GazeGAN3 import Gaze_GAN as Gaze_GAN3
from GazeGAN5 import Gaze_GAN as Gaze_GAN5
from GazeGAN6 import Gaze_GAN as Gaze_GAN6
from GazeGAN7 import Gaze_GAN as Gaze_GAN7
from GazeGAN8 import Gaze_GAN as Gaze_GAN8
from GazeGAN9 import Gaze_GAN as Gaze_GAN9
from GazeGAN10 import Gaze_GAN as Gaze_GAN10
from GazeGAN11_pretrain import Gaze_GAN as Gaze_GAN11_pre
from GazeGAN11 import Gaze_GAN as Gaze_GAN11
from GazeGAN12 import Gaze_GAN as Gaze_GAN12
from GazeGAN13 import Gaze_GAN as Gaze_GAN13
from GazeGAN14 import Gaze_GAN as Gaze_GAN14
from GazeGAN15 import Gaze_GAN as Gaze_GAN15
from GazeGAN16 import Gaze_GAN as Gaze_GAN16
from GazeGAN17 import Gaze_GAN as Gaze_GAN17
from GazeGAN18 import Gaze_GAN as Gaze_GAN18
from GazeGAN19 import Gaze_GAN as Gaze_GAN19
from GazeGAN20 import Gaze_GAN as Gaze_GAN20
from GazeGAN21 import Gaze_GAN as Gaze_GAN21
from GazeGAN22 import Gaze_GAN as Gaze_GAN22
from GazeGAN23 import Gaze_GAN as Gaze_GAN23
from GazeGAN24 import Gaze_GAN as Gaze_GAN24
from GazeGAN25 import Gaze_GAN as Gaze_GAN25
from GazeGAN26 import Gaze_GAN as Gaze_GAN26

from config.train_options import TrainOptions

import setproctitle
setproctitle.setproctitle("jichao: GazeGAN")

print("1")
opt = TrainOptions().parse()
print("2")

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    dataset = Dataset(opt)
    if '28_1_3' in opt.exper_name:
        gaze_gan = Gaze_GAN3(dataset, opt)
    elif '28_1_5' in opt.exper_name:
        gaze_gan = Gaze_GAN5(dataset, opt)
    elif '28_1_6' in opt.exper_name:
        gaze_gan = Gaze_GAN6(dataset, opt)
    elif '28_1_7' in opt.exper_name:
        gaze_gan = Gaze_GAN7(dataset, opt)
    elif '28_1_8' in opt.exper_name:
        gaze_gan = Gaze_GAN8(dataset, opt)
    elif '28_1_9' in opt.exper_name:
        gaze_gan = Gaze_GAN9(dataset, opt)
    elif '28_1_10' in opt.exper_name:
        gaze_gan = Gaze_GAN10(dataset, opt)
    elif '28_1_11' in opt.exper_name:
        gaze_gan = Gaze_GAN11(dataset, opt)
    elif '28_1_12' in opt.exper_name:
        gaze_gan = Gaze_GAN12(dataset, opt)
    elif '28_1_13' in opt.exper_name:
        gaze_gan = Gaze_GAN13(dataset, opt)
    elif '28_1_14' in opt.exper_name:
        gaze_gan = Gaze_GAN14(dataset, opt)
    elif '28_1_15' in opt.exper_name:
        gaze_gan = Gaze_GAN15(dataset, opt)
    elif '28_1_16' in opt.exper_name:
        gaze_gan = Gaze_GAN16(dataset, opt)
    elif '28_1_17' in opt.exper_name:
        gaze_gan = Gaze_GAN17(dataset, opt)
    elif '28_1_18' in opt.exper_name:
        gaze_gan = Gaze_GAN18(dataset, opt)
    elif '28_1_19' in opt.exper_name:
        gaze_gan = Gaze_GAN19(dataset, opt)
    elif '28_1_20' in opt.exper_name:
        gaze_gan = Gaze_GAN19(dataset, opt)
    elif '28_1_21' in opt.exper_name:
        gaze_gan = Gaze_GAN19(dataset, opt)
    elif '28_1_22' in opt.exper_name:
        gaze_gan = Gaze_GAN20(dataset, opt)
    elif '28_1_23' in opt.exper_name:
        gaze_gan = Gaze_GAN20(dataset, opt)
    elif '28_1_24' in opt.exper_name:
        gaze_gan = Gaze_GAN20(dataset, opt)
    elif '6_22_1' in opt.exper_name:
        gaze_gan = Gaze_GAN21(dataset, opt)
    elif '6_22_2' in opt.exper_name:
        gaze_gan = Gaze_GAN22(dataset, opt)
    elif '6_22_3' in opt.exper_name:
        gaze_gan = Gaze_GAN22(dataset, opt)
    elif '6_22_4' in opt.exper_name:
        gaze_gan = Gaze_GAN21(dataset, opt)
    elif '6_22_5' in opt.exper_name:
        gaze_gan = Gaze_GAN23(dataset, opt)
    elif '6_22_6' in opt.exper_name:
        gaze_gan = Gaze_GAN24(dataset, opt)
    elif '6_22_7' in opt.exper_name:
        gaze_gan = Gaze_GAN25(dataset, opt)
    elif '6_22_8' in opt.exper_name:
        gaze_gan = Gaze_GAN26(dataset, opt)
    else:
        gaze_gan = Gaze_GAN2(dataset, opt)

    gaze_gan.build_model()
    #gaze_gan.build_test_model()
    gaze_gan.train()