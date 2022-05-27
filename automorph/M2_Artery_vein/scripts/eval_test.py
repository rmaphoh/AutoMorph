import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.autograd import Function
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import filters
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import os
import cv2

def pixel_values_in_mask(true_vessels, pre_vessels_decode, mask, train_or, dataset):

    
    if train_or=='val':
        true_vessels = np.squeeze(true_vessels)
        pre_vessels_decode = np.squeeze(pre_vessels_decode)

        if dataset=='HRF-AV':
            true_vessels = (true_vessels[mask[0,...] != 0])
            pre_vessels_decode = (pre_vessels_decode[mask[0,...] != 0])
        else:
            true_vessels = (true_vessels[mask!= 0])
            pre_vessels_decode = (pre_vessels_decode[mask!= 0])
            

    return true_vessels.flatten(), pre_vessels_decode.flatten()

def AUC_ROC(true_vessel_arr, pred_vessel_arr, average):

    AUC_ROC=roc_auc_score(true_vessel_arr, pred_vessel_arr, average)
    return AUC_ROC

def threshold_by_otsu(pred_vessels):
    
    threshold=filters.threshold_otsu(pred_vessels)
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1

    return pred_vessels_bin

def AUC_PR(true_vessel_img, pred_vessel_img, average):

    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)

    try:
        acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        G = np.sqrt(sensitivity*specificity)
        F1_score_2 = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou
    
    except:

        return 0,0,0,0,0,0,0,0


def eval_net(epoch, net_G_1, net_G_A_1, net_G_V_1, net_G_2, net_G_A_2, net_G_V_2, net_G_3, net_G_A_3, net_G_V_3, net_G_4, net_G_A_4, net_G_V_4, net_G_5, net_G_A_5, net_G_V_5, net_G_6, net_G_A_6, net_G_V_6, net_G_7, net_G_A_7, net_G_V_7, net_G_8, net_G_A_8, net_G_V_8, dataset, loader, device, mode, train_or):
    """Evaluation without the densecrf with the dice coefficient"""
         
    mask_type = torch.float32 if net_G_1.n_classes == 1 else torch.long
    n_val = len(loader) 
    acc_a,sent_a,spet_a,pret_a,G_t_a,F1t_a,mset_a,iout_a,auc_roct_a,auc_prt_a=0,0,0,0,0,0,0,0,0,0
    acc_v,sent_v,spet_v,pret_v,G_t_v,F1t_v,mset_v,iout_v,auc_roct_v,auc_prt_v=0,0,0,0,0,0,0,0,0,0
    acc_u,sent_u,spet_u,pret_u,G_t_u,F1t_u,mset_u,iout_u,auc_roct_u,auc_prt_u=0,0,0,0,0,0,0,0,0,0
    acc,sent,spet,pret,G_t,F1t,mset,iout,auc_roct,auc_prt=0,0,0,0,0,0,0,0,0,0

    num = 0
    
    seg_results_small_path = dataset + '/Final_pre/small_pre/'
    seg_results_raw_path = dataset + '/Final_pre/raw_pre/'
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)
        
    mask_pred_tensor_small_all = 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, label, mask = batch['image'], batch['label'], batch['mask']
            img_name = batch['name'][0]
            ori_w, ori_h = mask.shape[2], mask.shape[3]
            image_transform=torch.zeros((imgs.shape[0],3,imgs.shape[2],imgs.shape[3]))
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=mask_type)

            with torch.no_grad():

                num +=1
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_1(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_1(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_1(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_2(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_2(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_2(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_3(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_3(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_3(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)                
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_4(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_4(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_4(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)    
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_5(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_5(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_5(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)    
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_6(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_6(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_6(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)   
                
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_7(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_7(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_7(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)   
                
                
                
                masks_pred_G_A, masks_pred_G_fusion_A = net_G_A_8(imgs)
                masks_pred_G_V, masks_pred_G_fusion_V = net_G_V_8(imgs)
                masks_pred_G_sigmoid_A_part = masks_pred_G_fusion_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_fusion_V.detach()

                mask_pred,_,_,_ = net_G_8(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)
                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)
                mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)   
                
                mask_pred_tensor_small_all = mask_pred_tensor_small_all/8
                _,prediction_decode = torch.max(mask_pred_tensor_small_all, 0)
                prediction_decode=prediction_decode.type(torch.FloatTensor)
                
                
                img_r = np.zeros((prediction_decode.shape[0],prediction_decode.shape[1]))
                img_g = np.zeros((prediction_decode.shape[0],prediction_decode.shape[1]))
                img_b = np.zeros((prediction_decode.shape[0],prediction_decode.shape[1]))
                
                img_r[prediction_decode==1]=255
                img_b[prediction_decode==2]=255
                img_g[prediction_decode==3]=255
                
                #img_b = remove_small_objects(img_b>0, 15, connectivity=3)
                #img_r = remove_small_objects(img_r>0, 15, connectivity=3)
                
                img_ = np.concatenate((img_b[...,np.newaxis], img_g[...,np.newaxis], img_r[...,np.newaxis]), axis=2)

                cv2.imwrite(seg_results_small_path+ img_name+ '.png', np.float32(img_)*255)
                img_ww = cv2.resize(np.float32(img_)*255, (ori_h,ori_w), interpolation = cv2.INTER_NEAREST)
                cv2.imwrite(seg_results_raw_path+ img_name+ '.png', img_ww)
            

            if mode== 'whole':
                ########################################

                # based on the whole images

                ########################################
                prediction_decode = F.interpolate(torch.unsqueeze(torch.unsqueeze(prediction_decode,axis=0),axis=0),(ori_w,ori_h), mode='nearest')
                prediction_decode = torch.squeeze(prediction_decode)
                mask_pred_softmax_cpu_decode = prediction_decode.detach().cpu()
                print(np.shape(prediction_decode))
                mask_pred_softmax_cpu_decode=mask_pred_softmax_cpu_decode.numpy()
                mask_pred_softmax_cpu_decode = np.squeeze(mask_pred_softmax_cpu_decode)

                label_cpu = label.detach().cpu().numpy()
                label_cpu = np.squeeze(label_cpu)

                mask_cpu = mask.detach().cpu().numpy()
                mask_cpu = np.squeeze(mask_cpu)
                
                count_artery = np.sum(label_cpu==1)
                count_vein = np.sum(label_cpu==2)
                count_uncertainty = np.sum(label_cpu==3)
                count_total = count_artery + count_vein + count_uncertainty

                ##########################################
                #artery
                #######################################
                #print('#######',np.unique(mask_pred_softmax_cpu_decode))
                label_cpu_flatten, mask_pred_softmax_cpu_decode_flatten = pixel_values_in_mask(label_cpu, mask_pred_softmax_cpu_decode, mask_cpu, train_or, dataset)

                #auc_roc_a=AUC_ROC(encoded_gt_vessel_point_a,encoded_pred_vessel_point_a,average=None)
                #auc_pr_a=AUC_PR(encoded_gt_vessel_point_a, encoded_pred_vessel_point_a,average=None)
                auc_roc_a=0
                auc_pr_a=0
                auc_roc_v=0
                auc_pr_v=0
                auc_roc_u=0
                auc_pr_u=0
                
                label_cpu_a,label_cpu_v,label_cpu_u=np.zeros((label_cpu_flatten.shape)),np.zeros((label_cpu_flatten.shape)),np.zeros((label_cpu_flatten.shape))
                pre_a,pre_v,pre_u=np.zeros((label_cpu_flatten.shape)),np.zeros((label_cpu_flatten.shape)),np.zeros((label_cpu_flatten.shape))
                
                label_cpu_a[label_cpu_flatten==1]=1
                label_cpu_v[label_cpu_flatten==2]=1
                label_cpu_u[label_cpu_flatten==3]=1
                
                pre_a[mask_pred_softmax_cpu_decode_flatten==1]=1
                pre_v[mask_pred_softmax_cpu_decode_flatten==2]=1
                pre_u[mask_pred_softmax_cpu_decode_flatten==3]=1
                

                acc_ve_a, sensitivity_ve_a, specificity_ve_a, precision_ve_a, G_ve_a, F1_score_ve_a, mse_a, iou_a = misc_measures(label_cpu_a, pre_a)
                
                acc_ve_v, sensitivity_ve_v, specificity_ve_v, precision_ve_v, G_ve_v, F1_score_ve_v, mse_v, iou_v = misc_measures(label_cpu_v, pre_v)
                    
                    
                acc_ve_u, sensitivity_ve_u, specificity_ve_u, precision_ve_u, G_ve_u, F1_score_ve_u, mse_u, iou_u = misc_measures(label_cpu_u, pre_u)
        
                acc_a+=acc_ve_a
                sent_a+=sensitivity_ve_a
                spet_a+=specificity_ve_a
                pret_a+=precision_ve_a
                G_t_a+=G_ve_a
                F1t_a+=F1_score_ve_a
                mset_a+=mse_a
                iout_a+=iou_a
                auc_roct_a+=auc_roc_a
                auc_prt_a+=auc_pr_a

                acc_v+=acc_ve_v
                sent_v+=sensitivity_ve_v
                spet_v+=specificity_ve_v
                pret_v+=precision_ve_v
                G_t_v+=G_ve_v
                F1t_v+=F1_score_ve_v
                mset_v+=mse_v
                iout_v+=iou_v
                auc_roct_v+=auc_roc_v
                auc_prt_v+=auc_pr_v
                
                
                if np.isnan(F1_score_ve_u):
                    acc_ve_u = 0
                    sensitivity_ve_u = 0
                    specificity_ve_u = 0
                    precision_ve_u = 0
                    G_ve_u = 0
                    F1_score_ve_u = 0
                    mse_u = 0
                    iou_u = 0
                    auc_roc_u = 0
                    auc_pr_u = 0
                    
                acc_u+=acc_ve_u
                sent_u+=sensitivity_ve_u
                spet_u+=specificity_ve_u
                pret_u+=precision_ve_u
                G_t_u+=G_ve_u
                F1t_u+=F1_score_ve_u
                mset_u+=mse_u
                iout_u+=iou_u
                auc_roct_u+=auc_roc_u
                auc_prt_u+=auc_pr_u

                
                acc+=(count_artery*acc_ve_a + count_vein*acc_ve_v + count_uncertainty*acc_ve_u)/count_total
                sent+=(count_artery*sensitivity_ve_a + count_vein*sensitivity_ve_v + count_uncertainty*sensitivity_ve_u)/count_total
                spet+=(count_artery*specificity_ve_a + count_vein*specificity_ve_v + count_uncertainty*specificity_ve_u)/count_total
                pret+=(count_artery*precision_ve_a + count_vein*precision_ve_v + count_uncertainty*precision_ve_u)/count_total
                G_t+=(count_artery*G_ve_a + count_vein*G_ve_v + count_uncertainty*G_ve_u)/count_total
                F1t+=(count_artery*F1_score_ve_a + count_vein*F1_score_ve_v + count_uncertainty*F1_score_ve_u)/count_total
                mset+=(count_artery*mse_a + count_vein*mse_v + count_uncertainty*mse_u)/count_total
                iout+=(count_artery*iou_a + count_vein*iou_v + count_uncertainty*iou_u)/count_total
                auc_roct+=(count_artery*auc_roc_a + count_vein*auc_roc_v + count_uncertainty*auc_roc_u)/count_total
                auc_prt+=(count_artery*auc_pr_a + count_vein*auc_pr_v + count_uncertainty*auc_pr_u)/count_total


    
    return  acc/ n_val, sent/ n_val, spet/ n_val, pret/ n_val, G_t/ n_val, F1t/ n_val, auc_roct/ n_val, auc_prt/ n_val, mset/ n_val, iout/ n_val, \
        acc_a/ n_val, sent_a/ n_val, spet_a/ n_val, pret_a/ n_val, G_t_a/ n_val, F1t_a/ n_val, auc_roct_a/ n_val, auc_prt_a/ n_val, mset_a/ n_val, iout_a/ n_val, \
            acc_v/ n_val, sent_v/ n_val, spet_v/ n_val, pret_v/ n_val, G_t_v/ n_val, F1t_v/ n_val, auc_roct_v/ n_val, auc_prt_v/ n_val, mset_v/ n_val, iout_v/ n_val, \
                acc_u/ n_val, sent_u/ n_val, spet_u/ n_val, pret_u/ n_val, G_t_u/ n_val, F1t_u/ n_val, auc_roct_u/ n_val, auc_prt_u/ n_val, mset_u/ n_val, iout_u/ n_val


