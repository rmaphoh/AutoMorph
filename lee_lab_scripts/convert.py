from PIL import Image
from glob import glob

for f in glob("/data/anand/Automorph/images/*.jpg"):
    Image.open(f).save(f.split(".jpg")[0]+".png", format="png")

