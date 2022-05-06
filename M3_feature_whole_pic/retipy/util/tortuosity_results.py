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

data = []
for _ in range(0, tags.shape[1]-1):  # ignore last tag (should be k-means)
    data.append([])


for i in range(0, tags.shape[0]):
    for j in range(0, tags.shape[1] - 1):
        data[j].append((i, tags[i, j]))


for i in range(0, tags.shape[1] - 1):
    ordered = sorted(data[i], key=lambda x: x[-1])

    def save_image(iid):
        current_image = ordered[iid][0]
        img = Retina(window[current_image, 1, :, :].astype(
            np.int, copy=False), "measure{}_rank{}_id{}.png".format(i, iid, current_image))
        img.save_image("measures/")

    for im_id in range(0, 5):
        save_image(im_id)

    for im_id in range(-5, 0):
        save_image(im_id)
