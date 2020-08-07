from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--use_sp', action='store_true', help='use spetral normalization')
        parser.add_argument('--crop_w', type=int, default=160, help='the size of cropped eye region')
        parser.add_argument('--crop_h', type=int, default=92, help='the size of crooped eye region')
        parser.add_argument('--crop_w_p', type=int, default=200, help='the padding version for cropped size')
        parser.add_argument('--crop_h_p', type=int, default=128, help='the padding version for crooped size')
        parser.add_argument('--test_num', type=int, default=300, help='the number of test samples')
        
        self.isTrain = False
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt