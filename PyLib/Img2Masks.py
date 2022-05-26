from __future__ import absolute_import
from __future__ import division

import os
import cv2
import numpy as np

faces_folder_path = './GazeDataNew/'
mask_path = os.path.join(faces_folder_path, 'masks')
txt_path = os.path.join(faces_folder_path, '0.txt')

batch_size = 1
crop_w = 160
crop_h = 92
img_size = 1024
output_nc = 3


def get_Mask_and_pos(eye_pos):

    def get_pos(i_eye_pos):
        o_eye_pos = np.zeros(shape=(batch_size, 4), dtype=np.int32)
        o_eye_pos[:, 3] = (i_eye_pos[:, 0] + crop_w / 2)
        o_eye_pos[:, 2] = (i_eye_pos[:, 1] + crop_h / 2)
        o_eye_pos[:, 1] = (i_eye_pos[:, 0] - crop_w / 2)
        o_eye_pos[:, 0] = (i_eye_pos[:, 1] - crop_h / 2)

        return o_eye_pos

    def get_Mask(i_left_eye_pos, i_right_eye_pos):
        batch_mask = np.zeros(shape=(batch_size, img_size, img_size, output_nc))
        # x, y = np.meshgrid(range(img_size), range(img_size))

        for i in range(batch_size):
            batch_mask[i, i_left_eye_pos[i][0]:i_left_eye_pos[i][2], i_left_eye_pos[i][1]:i_left_eye_pos[i][3], :] = 1
            batch_mask[i, i_right_eye_pos[i][0]:i_right_eye_pos[i][2], i_right_eye_pos[i][1]:i_right_eye_pos[i][3], :] = 1

        return batch_mask

    left_eye_pos = get_pos(eye_pos[:, 0:2])
    right_eye_pos = get_pos(eye_pos[:, 2:4])
    mask = get_Mask(left_eye_pos, right_eye_pos)

    return mask


os.makedirs(mask_path, exist_ok=True)
num = 0

fp = open(txt_path, 'r')
for item in fp:

    an_id = item.split(' ')
    pos = [int(value) for value in an_id[1:5]]
    # print(pos)
    pos = np.reshape(pos, newshape=[1, 4])
    mask_g = get_Mask_and_pos(pos)
    cv2.imwrite(os.path.join(mask_path, an_id[0] + '.jpg'), np.squeeze(mask_g) * 255.0)
