import argparse
import h5py
import numpy as np
from retipy.retina import Retina

parser = argparse.ArgumentParser()
parser.add_argument("file", help="the h5 file to be opened")
args = parser.parse_args()

file = h5py.File(args.file, 'r')
window = np.array(file['windows'])
print(window.shape)
tags = np.array(file['tags'])
print(tags.shape)

image_zero = []
image_one = []
image_two = []

for i in range(0, tags.shape[0]):

    if tags[i, -1] == 0:
        image_zero.append(i)
    elif tags[i, -1] == 1:
        image_one.append(i)
    elif tags[i, -1] == 2:
        image_two.append(i)

for image_id in image_zero:
    img = Retina(window[image_id, 1, :, :].astype(np.int, copy=False), "0_{}.png".format(image_id))
    img.save_image("tmp/")

for image_id in image_one:
    img = Retina(window[image_id, 1, :, :].astype(np.int, copy=False), "1_{}.png".format(image_id))
    img.save_image("tmp/")

for image_id in image_two:
    img = Retina(window[image_id, 1, :, :].astype(np.int, copy=False), "2_{}.png".format(image_id))
    img.save_image("tmp/")

