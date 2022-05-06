import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import os
import cv2
from torch.autograd import Function
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import filters



def pixel_values_in_mask(true_vessels, pred_vessels, mask_pad):

    true_vessels = np.squeeze(true_vessels)
    pred_vessels = np.squeeze(pred_vessels)
    true_vessels = (true_vessels[mask_pad != 0])
    pred_vessels = (pred_vessels[mask_pad != 0])
    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0
    return true_vessels.flatten(), pred_vessels.flatten()
    #return true_vessels, pred_vessels

def AUC_ROC(true_vessel_arr, pred_vessel_arr):

    AUC_ROC=roc_auc_score(true_vessel_arr, pred_vessel_arr)
    return AUC_ROC

def threshold_by_otsu(pred_vessels):

    threshold=filters.threshold_otsu(pred_vessels)
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1

    return pred_vessels_bin

def AUC_PR(true_vessel_img, pred_vessel_img):
 
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
        F1_score = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return acc, sensitivity, specificity, precision, G, F1_score, mse, iou
    except:
        return 0,0,0,0,0,0,0,0


def eval_net(net, loader, device, dataset_name,dataset_test, job_name, mask_or, train_or):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()

    mask_type = torch.float32
    n_val = len(loader) 
    tot=0
    sent=0
    spet=0
    pret=0
    G_t=0
    F1t=0
    mset=0
    iout=0
    auc_roct=0
    auc_prt=0

    num = 0
    
    seg_results_small_path = './Final_pre/' + dataset_name + '/' + dataset_test + '/'+ job_name +'/Resized_segmentation/'
    seg_results_raw_path = './Final_pre/' + dataset_name + '/' + dataset_test + '/' + job_name +'/Ori_size_segmentation/'
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)
        

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks, mask = batch['image'], batch['label'], batch['mask']
            img_name = batch['name'][0]
            target_w, target_h = imgs.shape[2], imgs.shape[3]
            ori_w, ori_h = true_masks.shape[2], true_masks.shape[3]
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                
            if mask_or:

                mask_pred_sigmoid = torch.sigmoid(mask_pred)
                
                save_image(mask_pred_sigmoid, seg_results_small_path+img_name+'.png')
                mask_pred_img = Image.open(seg_results_small_path+img_name+'.png').resize((ori_h,ori_w)).convert('L') 
                mask_pred_tensor = torchvision.transforms.ToTensor()(mask_pred_img)
                mask_pred_tensor[mask == 0]=0
                save_image(mask_pred_tensor, seg_results_raw_path+img_name+'.png')
                
                mask_pred_sigmoid_cpu = mask_pred_tensor.detach().cpu().numpy()
                mask_cpu = mask.detach().cpu().numpy()
                mask_cpu = np.squeeze(mask_cpu)
                true_masks_cpu = true_masks.detach().cpu().numpy()
                vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_masks_cpu, mask_pred_sigmoid_cpu, mask_cpu)
                            
                true_masks_cpu = true_masks.detach().cpu().numpy()
                vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_masks_cpu, mask_pred_sigmoid_cpu, mask_cpu)
                auc_roc=AUC_ROC(vessels_in_mask,generated_in_mask)
                auc_pr=AUC_PR(vessels_in_mask, generated_in_mask)
                binarys_in_mask=((generated_in_mask)>0.5).astype('float')
                acc, sensitivity, specificity, precision, G, F1_score, mse, iou = misc_measures(vessels_in_mask, binarys_in_mask)

                tot+= acc 
                sent+= sensitivity
                spet+=specificity
                pret+=precision
                G_t+=G
                F1t+=F1_score
                mset+=mse
                iout+=iou
                auc_roct+=auc_roc
                auc_prt+=auc_pr

            pbar.update()

    net.train()
    return tot / n_val, sent / n_val, spet / n_val, pret / n_val, G_t / n_val, F1t / n_val, auc_roct / n_val, auc_prt / n_val, iout/n_val, mset/n_val

