from glob import glob
import automorph.config as gv 

images = glob("{}*.png".format(gv.image_dir))
print("{} images found with glob".format(len(images)))

with open("/data/anand/Automorph/resolution_information.csv", "w") as f:
  f.write("fundus,res\n")
  f.writelines("{},1\n".format(x.split('/')[-1]) for x in images)  	
