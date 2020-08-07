from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_model_freq', type=int, default=10000, help='frequency of saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=200000, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50000, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for Adam in d')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for Adam in g')
        parser.add_argument('--lr_r', type=float, default=0.0005, help='initial learning rate for Adam in r')
        parser.add_argument('--lr_sr', type=float, default=4e-4, help='initial learning rate for Adam in sr')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
        parser.add_argument('--loss_type', type=str, default='softplus', choices=['hinge', 'gan', 'wgan', 'softplus', 'lsgan'], help='using type of gan loss')
        parser.add_argument('--lam_gp', type=float, default=10.0, help='weight for gradient penalty of gan')
        parser.add_argument('--lam_p', type=float, default=100.0, help='perception loss in g')
        parser.add_argument('--lam_r', type=float, default=1.0, help='weight for recon loss in g')
        parser.add_argument('--lam_ss', type=float, default=1, help='self-supervised loss in g')
        parser.add_argument('--lam_fp', type=float, default=0.1, help='fp loss for g')
        parser.add_argument('--use_sp', action='store_true', help='use spectral normalization')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--crop_w', type=int, default=50, help='the size of cropped eye region')
        parser.add_argument('--crop_h', type=int, default=30, help='the size of crooped eye region')
        parser.add_argument('--crop_w_p', type=int, default=180, help='the padding version for cropped size')
        parser.add_argument('--crop_h_p', type=int, default=128, help='the padding version for crooped size')
        parser.add_argument('--test_num', type=int, default=300, help='the number of test samples')

        self.isTrain = True
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt