from glob import glob
import numpy as np
import pandas as pd
import shutil
import os
import automorph.config as gv

def quality_assessment():

    result_Eyepacs_path = '{}M1/results_ensemble.csv'.format(gv.results_dir) #TODO replace this in some way?
    
    if not os.path.exists('{}M1/Good_quality/'.format(gv.results_dir)):
        os.makedirs('{}M1/Good_quality/'.format(gv.results_dir))
    if not os.path.exists('{}M1/Bad_quality/'.format(gv.results_dir)):
        os.makedirs('{}M1/Bad_quality/'.format(gv.results_dir))
    
    result_Eyepacs_ = pd.read_csv(result_Eyepacs_path)
   
    # save the Good_quality
    gq = result_Eyepacs_[result_Eyepacs_['Prediction'] == 0]['Name'].values
    usable = result_Eyepacs_[(result_Eyepacs_['Prediction'] == 1) &
                 (result_Eyepacs_['softmax_bad'] < 0.25)]['Name'].values
    gq = np.append(gq, usable)
    gq = list(gq)

    # find the bad quality images
    bq = result_Eyepacs_[~result_Eyepacs_['Name'].isin(gq)]['Name'].values

    # merge the images with the crop data and save them as a csv
    crops = pd.read_csv(gv.results_dir+'M0/crop_info.csv', usecols=['Name', 'centre_h', 'centre_w', 'radius'])
    gq_crops = crops[crops['Name'].isin(gq)]
    gq_crops.to_csv('{}/M1/Good_quality/image_list.csv'.format(gv.results_dir), index=False)
    bq_crops = crops[crops['Name'].isin(bq)]
    bq_crops.to_csv('{}/M1/Bad_quality/image_list.csv'.format(gv.results_dir), index=False)

    # print metrics
    print('Gradable cases by EyePACS_QA is {} '.format(len(gq)))
    print('Ungradable cases by EyePACS_QA is {} '.format(len(bq)))


    # if not sparse, then copy and save all the images into a M1/Good_quality or M1/Bad_quality folder
    if not gv.sparse:
    
        Eyepacs_pre = result_Eyepacs_['Prediction']
        Eyepacs_bad_mean = result_Eyepacs_['softmax_bad']
        Eyepacs_usable_sd = result_Eyepacs_['usable_sd']
        name_list = result_Eyepacs_['Name']
         
        Eye_good = 0
        Eye_bad = 0
 
        for i in range(len(name_list)):
            
            f = name_list[i].split('/')[-1]
            name = gv.image_dir+"M0/images/"+f
            
            if Eyepacs_pre[i]==0:
                Eye_good+=1
                shutil.copy(name, '{}M1/Good_quality/'.format(gv.results_dir))
            elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25):
            #elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25) and (Eyepacs_usable_sd[i]<0.1):
                Eye_good+=1
                shutil.copy(name, '{}M1/Good_quality/'.format(gv.results_dir))        
            else:
                Eye_bad+=1        
                shutil.copy(name, '{}M1/Bad_quality/'.format(gv.results_dir))
    