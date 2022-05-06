import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Function
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import filters
import numpy as np
from PIL import Image


def pad_imgs( imgs, img_size):
    img_h,img_w=imgs.shape[0], imgs.shape[1]
    target_h,target_w=img_size[0],img_size[1] 
    if len(imgs.shape)==3:
        d=imgs.shape[2]
        padded=np.zeros((target_h, target_w,d))
    elif len(imgs.shape)==2:
        padded=np.zeros((target_h, target_w))
    padded[(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=imgs
    #print(np.shape(padded))
    return padded



def pixel_values_in_mask(true_vessels, pred_vessels, module_pad):


    true_vessels = np.squeeze(true_vessels)
    pred_vessels = np.squeeze(pred_vessels)
    
    if 0 in module_pad:
        true_vessels = (true_vessels[module_pad != 0])
        pred_vessels = (pred_vessels[module_pad != 0])

    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0


    return true_vessels.flatten(), pred_vessels.flatten()

def AUC_ROC(true_vessel_arr, pred_vessel_arr):
    """
    Area under the ROC curve with x axis flipped
    """
    AUC_ROC=roc_auc_score(true_vessel_arr, pred_vessel_arr)
    return AUC_ROC

def threshold_by_otsu(pred_vessels):
    
    # cut by otsu threshold
    threshold=filters.threshold_otsu(pred_vessels)
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1

    return pred_vessels_bin

def AUC_PR(true_vessel_img, pred_vessel_img):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
    sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
    specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
    precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
    G = np.sqrt(sensitivity*specificity)
    F1_score_2 = 2*precision*sensitivity/(precision+sensitivity)
    return acc, sensitivity, specificity, precision, G, F1_score_2

def print_metrics(itr, **kargs):
    print ("*** Round {}  ====> ".format(itr),)
    for name, value in kargs.items():
        print ( "{} : {}, ".format(name, value),)
    print ("")
    sys.stdout.flush()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001

        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def eval_net(epoch, net, loader, device, mask):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    ##################sigmoid or softmax
    #mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type = torch.float32 if net.n_classes == 1 else torch.float32
    n_val = len(loader)  # the number of batch
    tot = 0
    img_size = (592,592)

    tot= 0 
    sent= 0
    spet=0
    pret=0
    G_t=0
    F1t=0

    auc_roct=0
    auc_prt=0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            module = batch['module']

            if len(imgs.shape)==5:
                bs_img, ncrops_img, c_img, h_img, w_img = imgs.size()
                bs_label, ncrops_label, c_label, _,_ = true_masks.size()

                imgs = imgs.view(-1, c_img, h_img, w_img)
                true_masks = true_masks.view(-1, c_label, h_img, w_img)
                module = module.view(-1, c_label, h_img, w_img)

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            module_pad_cpu = module.detach().cpu().numpy()
            module_pad_cpu = np.squeeze(module_pad_cpu)
            with torch.no_grad():
                mask_pred = net(imgs)
            
            ##################sigmoid or softmax

            if mask:

                mask_pred_sigmoid = torch.sigmoid(mask_pred)
                mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                true_masks_cpu = true_masks.detach().cpu().numpy()

                vessels_in_mask, generated_in_mask = pixel_values_in_mask(true_masks_cpu, mask_pred_sigmoid_cpu, module_pad_cpu )


                auc_roc=AUC_ROC(vessels_in_mask,generated_in_mask)
                auc_pr=AUC_PR(vessels_in_mask, generated_in_mask)
                
                binarys_in_mask=threshold_by_otsu(generated_in_mask)
                #binarys_in_mask=((generated_in_mask)>0.5).astype('float')
                ######################################## 
                acc, sensitivity, specificity, precision, G, F1_score_2 = misc_measures(vessels_in_mask, binarys_in_mask)


                tot+= acc 
                sent+= sensitivity
                spet+=specificity
                pret+=precision
                G_t+=G
                F1t+=F1_score_2

                auc_roct+=auc_roc
                auc_prt+=auc_pr
                ###################################### 
            pbar.update()

    net.train()
    return tot / n_val, tot / n_val, sent / n_val, spet / n_val, pret / n_val, G_t / n_val, F1t / n_val, auc_roct / n_val, auc_prt / n_val

