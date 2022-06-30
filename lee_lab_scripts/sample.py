import random, os
from PIL import Image
from glob import glob


 #labelsd = {}
 #bins = {}
 #with open("/data/Kaggle-DR/raw/trainLabels.csv") as fin:
 #  first = True
 #  for l in fin:
 #    if first:
 #      first = False
 #      continue
 #    arr = l.strip().split(",")
 #    labels[arr[0]] = arr[1]
 #    if not arr[1] in bins:
 #      bins[arr[1]] = []
 #    bins[arr[1]].append(arr[0])


froot = "/data/Kaggle-DR/raw/train/"
fend = ".jpeg"
fdest = "/data/anand/Automorph/images/"


for i in bins.keys():
  for fi in random.sample(bins[i], 200):
    f = froot + fi + fend
    print(f)
    os.system("cp %s %s%s-%s.jpg" % (f, fdest, i, fi))

