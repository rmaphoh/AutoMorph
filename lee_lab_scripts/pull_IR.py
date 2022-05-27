import fnmatch
import re
from glob import glob
from shutil import copy

ir = glob("/data/oct-heyex-data/macoct/*/*/*/ir.png")

print(len(ir))

root = "/data/anand/Automorph/images/"
for idx,f in enumerate(ir[:20]):
    fname = "{}.png".format(idx)
    copy(f, root+fname)