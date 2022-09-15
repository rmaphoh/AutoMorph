from glob import glob
import numpy as np
import pandas as pd
import shutil
import os

def quality_assessment(cfg):

    result_Eyepacs_path = '{}M1/results_ensemble.csv'.format(cfg.results_dir) #TODO replace this in some way?
    
    if not os.path.exists('{}M1/Good_quality/'.format(cfg.results_dir)):
        os.makedirs('{}M1/Good_quality/'.format(cfg.results_dir))
    if not os.path.exists('{}M1/Bad_quality/'.format(cfg.results_dir)):
        os.makedirs('{}M1/Bad_quality/'.format(cfg.results_dir))
    
    result_Eyepacs_ = pd.read_csv(result_Eyepacs_path)
   
    # save the Good_quality
    gq = result_Eyepacs_[result_Eyepacs_['Prediction'] == 0]['Name'].values
    if cfg.quality_thresh == 'good':
        usable = result_Eyepacs_[(result_Eyepacs_['Prediction'] == 1) &
                     (result_Eyepacs_['softmax_bad'] < 0.25)]['Name'].values
    if cfg.quality_thresh == 'usable':
        usable = result_Eyepacs_[(result_Eyepacs_['Prediction'] == 1)]['Name'].values
    if cfg.quality_thresh == 'all':
        usable = result_Eyepacs_[result_Eyepacs_['Prediction'].isin([1,2])]['Name'].values
    gq = np.append(gq, usable)
    gq = list(gq)

    # find the bad quality images
    bq = result_Eyepacs_[~result_Eyepacs_['Name'].isin(gq)]['Name'].values
     
    # drop all images where radius and centre are 0 
    def find_zeros(a):
        if (a['centre_w'] == 0) & (a['centre_h'] == 0) & (a['radius'] == 0):
            return False
        else:
            return True

    # merge the images with the crop data and save them as a csv
    crops = pd.read_csv(cfg.results_dir+'M0/crop_info.csv', usecols=['Name', 'centre_h', 'centre_w', 'radius'])
    crops = crops[crops.apply(find_zeros, axis=1)]
    gq_crops = crops[crops['Name'].isin(gq)]

    gq_crops.to_csv('{}/M1/Good_quality/image_list.csv'.format(cfg.results_dir), index=False)
    bq_crops = crops[crops['Name'].isin(bq)]
    bq_crops.to_csv('{}/M1/Bad_quality/image_list.csv'.format(cfg.results_dir), index=False)

    # print metrics
    print('Gradable cases by EyePACS_QA is {} '.format(len(gq)))
    print('Ungradable cases by EyePACS_QA is {} '.format(len(bq)))


    # if not sparse, then copy and save all the images into a M1/Good_quality or M1/Bad_quality folder
    if not cfg.sparse:
        print('copying good and bad quality images from M0/images into subsequent directories')
    
        for id in gq:

            f = id.split('/')[-1].split('.')[0]+'.png'
            name = cfg.results_dir+"M0/images/"+f
            shutil.copy(name, '{}M1/Good_quality/{}'.format(cfg.results_dir, f))
            
        for id in bq:

            f = id.split('/')[-1].split('.')[0]+'.png'
            name = cfg.results_dir+"M0/images/"+f
            shutil.copy(name, '{}M1/Bad_quality/{}'.format(cfg.results_dir, f))
               