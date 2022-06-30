from glob import glob
import automorph.config as gv 
from pathlib import Path

images = glob("{}*.png".format(gv.image_dir))
print("{} images found with glob".format(len(images)))

res_csv_pth = Path(__file__).parent / "../automorph/resolution_information.csv" 
with open(res_csv_pth, "w") as f:
  f.write("fundus,res\n")
  f.writelines("{},1\n".format(x.split('/')[-1]) for x in images)  	
