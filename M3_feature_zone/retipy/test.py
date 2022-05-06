import os
import numpy as np

path_ = '../../Results/M2/artery_vein/Zone_B_disc_centred_artery_process/'
test_ = path_.split('/')[-2]
if 'artery' in path_.split('/')[-2]:
    print('@@@@')

    
def Hubbard_cal(w1,w2):

    w_artery = np.sqrt(0.87*np.square(w1) + 1.01*np.square(w2) - 0.22*w1*w2 - 10.76) 
    w_vein = np.sqrt(0.72*np.square(w1)+0.91*np.square(w2)+450.05)
    
    return w_artery,w_vein

aaa = [80,90,100]
bbb = [8,9,10,11,12,13]

sorted_w1_list_average = sorted(aaa)[-6:]
    
w_first_artery_Hubbard_1, w_first_vein_Hubbard_1 = Hubbard_cal(sorted_w1_list_average[0],sorted_w1_list_average[5])

w_first_artery_Hubbard_2, w_first_vein_Hubbard_2 = Hubbard_cal(sorted_w1_list_average[1],sorted_w1_list_average[4])

w_first_artery_Hubbard_3, w_first_vein_Hubbard_3 = Hubbard_cal(sorted_w1_list_average[2],sorted_w1_list_average[3])
    
CRAE_first_round = sorted([w_first_artery_Hubbard_1,w_first_artery_Hubbard_2,w_first_artery_Hubbard_3])
CRVE_first_round = sorted([w_first_vein_Hubbard_1,w_first_vein_Hubbard_2,w_first_vein_Hubbard_3])
    

w_second_artery_Hubbard_1, w_second_vein_Hubbard_1 = Hubbard_cal(CRAE_first_round[0],CRAE_first_round[2])  
        
CRAE_second_round = sorted([w_second_artery_Hubbard_1,CRAE_first_round[1]])
CRAE_Hubbard,CRVE_Hubbard = Hubbard_cal(CRAE_second_round[0],CRAE_second_round[1])
print(CRAE_Hubbard)

w_second_artery_Hubbard_1, w_second_vein_Hubbard_1 = Hubbard_cal(CRVE_first_round[0],CRVE_first_round[2])  
        
CRVE_second_round = sorted([w_second_vein_Hubbard_1,CRVE_first_round[1]])
CRAE_Hubbard,CRVE_Hubbard = Hubbard_cal(CRVE_second_round[0],CRVE_second_round[1])
print(CRVE_Hubbard)