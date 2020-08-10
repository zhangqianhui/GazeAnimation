from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from IMLib.utils import *

class Dataset(object):

    def __init__(self, config):
        super(Dataset, self).__init__()

        self.data_dir = config.data_dir
        self.attr_0_txt = '0.txt'
        self.attr_1_txt = '1.txt'
        self.height, self.width= config.img_size, config.img_size
        self.channel = config.output_nc
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads
        self.test_number = config.test_num

        self.train_images_list, self.train_eye_pos, \
        self.train_images_list2, self.train_eye_pos2, \
                self.test_images_list, self.test_eye_pos, \
                    self.test_images_list0, self.test_eye_pos0 = self.readfilenames()

        print(len(self.train_images_list), len(self.train_eye_pos),
              len(self.train_images_list2), len(self.train_eye_pos2),
              len(self.train_images_list2), len(self.train_eye_pos2),
              len(self.test_images_list), len(self.test_eye_pos),
                len(self.test_images_list0), len(self.test_eye_pos0))

        assert len(self.train_images_list) == len(self.train_eye_pos)
        assert len(self.train_images_list2) == len(self.train_eye_pos2)
        assert len(self.test_images_list) == len(self.test_eye_pos)
        assert len(self.test_images_list0) == len(self.test_eye_pos0)

        assert len(self.train_images_list) > 0

    def readfilenames(self):

        train_eye_pos = []
        train_images_list = []
        fh = open(os.path.join(self.data_dir, self.attr_1_txt))

        for f in fh.readlines():
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg")):
                train_images_list.append(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg"))
                train_eye_pos.append([int(value) for value in filenames[1:5]])
        fh.close()

        fh = open(os.path.join(self.data_dir, self.attr_0_txt))
        test_images_list = []
        test_eye_pos = []

        for f in fh.readlines():
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "0/"+filenames[0]+".jpg")):
                test_images_list.append(os.path.join(self.data_dir,"0/"+filenames[0]+".jpg"))
                test_eye_pos.append([int(value) for value in filenames[1:5]])
                #print test_eye_pos
        fh.close()

        train_images_list2 = test_images_list[0:-self.test_number]
        train_eye_pos2 = test_eye_pos[0:-self.test_number]

        test_images_list = test_images_list[-self.test_number:]
        test_eye_pos = test_eye_pos[-self.test_number:]

        test_images_list0 = train_images_list[-100:]
        test_eye_pos0 = train_eye_pos[-100:]

        train_images_list = train_images_list[0:-100]
        train_eye_pos = train_eye_pos[0:-100]

        return train_images_list, train_eye_pos, \
               train_images_list2, train_eye_pos2, \
               test_images_list, test_eye_pos, \
               test_images_list0, test_eye_pos0

    def read_images(self, input_queue):

        content = tf.read_file(input_queue)
        # Decode a JPEG-encoded image to a uint8 tensor
        image = tf.image.decode_jpeg(content, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, size=(self.height, self.width))
        return image / 127.5 - 1.0

    def input(self):

        train_images = tf.convert_to_tensor(self.train_images_list, dtype=tf.string)
        train_eye_pos = tf.convert_to_tensor(self.train_eye_pos, dtype=tf.int32)
        train_queue = tf.train.slice_input_producer([train_images, train_eye_pos], shuffle=True)
        train_eye_pos_queue = train_queue[1]
        train_images_queue = self.read_images(input_queue=train_queue[0])

        train_images2 = tf.convert_to_tensor(self.train_images_list2, dtype=tf.string)
        train_eye_pos2 = tf.convert_to_tensor(self.train_eye_pos2, dtype=tf.int32)
        train_queue2 = tf.train.slice_input_producer([train_images2, train_eye_pos2], shuffle=True)
        train_eye_pos_queue2 = train_queue2[1]
        train_images_queue2 = self.read_images(input_queue=train_queue2[0])

        test_images = tf.convert_to_tensor(self.test_images_list, dtype=tf.string)
        test_eye_pos = tf.convert_to_tensor(self.test_eye_pos, dtype=tf.int32)
        test_queue = tf.train.slice_input_producer([test_images, test_eye_pos], shuffle=False)
        test_eye_pos_queue = test_queue[1]
        test_images_queue = self.read_images(input_queue=test_queue[0])

        batch_image1, batch_eye_pos1, \
                batch_image2, batch_eye_pos2= tf.train.shuffle_batch([train_images_queue, train_eye_pos_queue,
                                                            train_images_queue2, train_eye_pos_queue2],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=500)

        batch_image3, batch_eye_pos3 = tf.train.batch([test_images_queue, test_eye_pos_queue],
                                                batch_size=self.batch_size,
                                                capacity=100,
                                                num_threads=1)

        return batch_image1, batch_eye_pos1, batch_image2, batch_eye_pos2, batch_image3, \
               batch_eye_pos3
