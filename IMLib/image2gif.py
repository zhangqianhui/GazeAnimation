from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imageio

filename = '../test_dir9/183inter.jpg'
with imageio.get_writer('../test_dir9/30.gif', mode='I') as writer:
    image = imageio.imread(filename)
    w = image.shape[1]
    im_list = []
    for i in range(w//256):
        image0 = image[:, i*256:(i+1)*256,:]
        im_list.append(image0)
    im_list_ = im_list[1:] + list(reversed(im_list[1:]))
    for i in range(len(im_list_)):
        writer.append_data(im_list_[i])
