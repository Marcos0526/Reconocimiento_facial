import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



np.set_printoptions(precision=4)

df = pd.read_csv('archive/list_attr_celeba2.txt', sep= ',' , header = None )


files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))

print(data)


path_to_images = 'archive/img_align_celeba/img_align_celeba/'


def process_file(file_name, attributes):

    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  #
    return image, attributes



labeled_images = data.map(process_file)

print(labeled_images)

for image, attributes in labeled_images.take(1):
    plt.imshow(image)
    plt.show()

exit()
