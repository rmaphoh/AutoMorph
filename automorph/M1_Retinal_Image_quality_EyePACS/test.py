from dataset import BasicDataset_OUT

ds = BasicDataset_OUT('/data/anand/Automorph_data/ukb/images/', [912,912], 2, False, 
'/data/anand/Automorph_data/ukb/results/M0/crop_info.csv')

ds.__getitem__(1)