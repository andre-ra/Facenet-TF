import os
import numpy as np
# import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
Create tf.data.Dataset from a directory of Images.
Preprocess Images : Resize, Flip, Normalize
'''


def preprocess_image(image, image_height, image_width):
    
    image = tf.image.resize(image, [image_height, image_width])
    #image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image


def parse_image_function(image_path, label, image_height, image_width):
      
    image_string = tf.io.read_file(image_path)
    #image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.decode_png(image_string, channels=3)
    image = preprocess_image(image, image_height, image_width)
    return image, label


def get_dataset(dir, params, k_element, phase='train'):

    dir_paths  =  os.listdir(dir)
    dir_paths  =  [os.path.join(dir, dir_path) for dir_path in dir_paths]

    image_paths = []
    image_label = []
    for dir_path in dir_paths:
        for image_path in os.listdir(dir_path):
            if image_path.endswith('.png'):
                img = plt.imread(os.path.join(dir_path,image_path))
                plt.imshow(img, cmap='gray')
                #image_path_short = image_path.rsplit('_',1)[0]
                image_sample_number = int(str(image_path)[1])
                if ((image_sample_number == k_element and phase=='validate') or (image_sample_number != k_element and phase=='train')):
                    image_paths.append(os.path.join(dir_path, image_path))
                    image_label.append(dir_path.split('/')[-1])

    AUTOTUNE   =  tf.data.experimental.AUTOTUNE
    dataset    =  tf.data.Dataset.from_tensor_slices((image_paths, image_label))
    dataset    =  dataset.map(lambda x, y: parse_image_function(x, y, params.image_height, params.image_width))
    dataset    =  dataset.batch(params.batch_size).prefetch(AUTOTUNE)
    
    return dataset, len(image_paths)


if __name__ == "__main__":

    print(get_dataset('../face-data'))
