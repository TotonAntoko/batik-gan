import os
from keras.preprocessing import image
import tensorflow as tf
import numpy as np


class DataPipeline:
    def __init__(self):
        # self.path = path
        self

    def load_data(self, path, size):
        list_img = list()
        for filename in os.listdir(path):
            # load and resize the image
            img = image.load_img(path + filename, target_size=size)
            # convert to numpy array
            img = image.img_to_array(img)
            # Convert to lower case
            path_file = (path + filename).lower()
            # store
            list_img.append([path_file, img])
        img_sorted = sorted(list_img)
        img_arr = np.asarray([i[1] for i in img_sorted])

        return img_arr

    def concat_image(self, patch_a, patch_b):
        def h(patch_a, patch_b):
            h = tf.concat([patch_a, patch_b, patch_a, patch_b], 1)
            return h
        v = tf.concat([h(patch_a, patch_b), h(patch_b, patch_a),
                       h(patch_a, patch_b), h(patch_b, patch_a)], 2)
        return v

    def normalisasi(self, A, B):
        A = tf.cast(A, dtype=tf.float32)
        A = (A-127.5)/127.5
        B = tf.cast(B, dtype=tf.float32)
        B = (B-127.5)/127.5
        img = self.concat_image(tf.expand_dims(A, axis=0),
                                tf.expand_dims(B, axis=0))[0]
        return A, B, img

    def execute(self):
        # dataset = self.load_data()
        path = 'static/dataset/201710370311030/dataset_batik/'
        data_patch = self.load_data(path + 'Patch/', (32, 32))

        patch_a = data_patch[[i for i in range(32) if i % 2 == 0]]
        patch_b = data_patch[[i for i in range(32) if i % 2 != 0]]

        dataset = tf.data.Dataset.from_tensor_slices((patch_a, patch_b))

        # # AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.map(self.normalisasi, num_parallel_calls=-1).cache()
        dataset = dataset.batch(1, drop_remainder=True).repeat(1)
        dataset = dataset.prefetch(1)
        return dataset
