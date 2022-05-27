from glob import glob
import numpy as np
import pandas as pd
import shutil
import os
import automorph.config as gv

def quality_assessment():

    result_Eyepacs = '{}M1/results_ensemble.csv'.format(gv.results_dir) #TODO replace this in some way?
    
    if not os.path.exists('{}M1/Good_quality/'.format(gv.results_dir)):
        os.makedirs('{}M1/Good_quality/'.format(gv.results_dir))
    if not os.path.exists('{}M1/Bad_quality/'.format(gv.results_dir)):
        os.makedirs('{}M1/Bad_quality/'.format(gv.results_dir))
    
    result_Eyepacs_ = pd.read_csv(result_Eyepacs)
    
    Eyepacs_pre = result_Eyepacs_['Prediction']
    Eyepacs_bad_mean = result_Eyepacs_['softmax_bad']
    Eyepacs_usable_sd = result_Eyepacs_['usable_sd']
    name_list = result_Eyepacs_['Name']
    
    Eye_good = 0
    Eye_bad = 0
    
    for i in range(len(name_list)):
        
        if Eyepacs_pre[i]==0:
            Eye_good+=1
            shutil.copy(name_list[i], '{}M1/Good_quality/'.format(gv.results_dir))
        elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25):
        #elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25) and (Eyepacs_usable_sd[i]<0.1):
            Eye_good+=1
            shutil.copy(name_list[i], '{}M1/Good_quality/'.format(gv.results_dir))        
        else:
            Eye_bad+=1        
            shutil.copy(name_list[i], '{}M1/Bad_quality/'.format(gv.results_dir))
            #shutil.copy(name_list[i], '../Results/M1/Good_quality/')
    
    
    print('Gradable cases by EyePACS_QA is {} '.format(Eye_good))
    print('Ungradable cases by EyePACS_QA is {} '.format(Eye_bad))
    