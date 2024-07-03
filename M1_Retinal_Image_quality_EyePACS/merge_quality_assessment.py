import numpy as np
import pandas as pd
import shutil
import os

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','..')

result_Eyepacs = f'{AUTOMORPH_DATA}/Results/M1/results_ensemble.csv'

if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M1/Good_quality/'):
    os.makedirs(f'{AUTOMORPH_DATA}/Results/M1/Good_quality/')
if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M1/Bad_quality/'):
    os.makedirs(f'{AUTOMORPH_DATA}/Results/M1/Bad_quality/')

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
        shutil.copy(name_list[i], f'{AUTOMORPH_DATA}/Results/M1/Good_quality/')
    elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25):
    #elif (Eyepacs_pre[i]==1) and (Eyepacs_bad_mean[i]<0.25) and (Eyepacs_usable_sd[i]<0.1):
        Eye_good+=1
        shutil.copy(name_list[i], f'{AUTOMORPH_DATA}/Results/M1/Good_quality/')        
    else:
        Eye_bad+=1        
        shutil.copy(name_list[i], f'{AUTOMORPH_DATA}/Results/M1/Bad_quality/')
        #shutil.copy(name_list[i], '../Results/M1/Good_quality/')


print('Gradable cases by EyePACS_QA is {} '.format(Eye_good))
print('Ungradable cases by EyePACS_QA is {} '.format(Eye_bad))
