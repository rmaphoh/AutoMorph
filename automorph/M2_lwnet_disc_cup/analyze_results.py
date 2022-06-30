import argparse
from PIL import Image
import os, sys
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
from utils.evaluation import dice_score
from utils.model_saving_loading import str2bool

# future-self: dice and f1 are the same thing, but if you use f1_score from sklearn it will be much slower, the reason
# being that dice here expects bools and it won't work in multi-class scenarios. Same goes for accuracy_score.
# (see https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/)

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', type=str, default='DRIVE', help='which dataset was the model trained on')
parser.add_argument('--test_dataset', type=str, default='DRIVE', help='which dataset to test')
parser.add_argument('--path_train_preds', type=str, default=None, help='path to training predictions')
parser.add_argument('--path_test_preds', type=str, default=None, help='path to test predictions')
parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')
parser.add_argument('--public', type=str2bool, nargs='?', const=True, default=True, help='public or private dataset')

def get_labels_preds(path_to_preds, csv_path):
    df = pd.read_csv(csv_path)
    im_list, mask_list, gt_list = df.im_paths, df.mask_paths, df.gt_paths
    all_bin_preds = []
    all_preds = []
    all_gts = []
    for i in range(len(gt_list)):
        im_path = im_list[i].rsplit('/', 1)[-1]
        pred_path = osp.join(path_to_preds, im_path[:-4] + '.png')
        gt_path = gt_list[i]
        mask_path = mask_list[i]

        gt = np.array(Image.open(gt_path)).astype(bool)
        mask = np.array(Image.open(mask_path).convert('L')).astype(bool)
        from skimage import img_as_float
        try: pred = img_as_float(np.array(Image.open(pred_path)))
        except FileNotFoundError:
            sys.exit('---- no predictions found at {} (maybe run first generate_results.py?) ---- '.format(path_to_preds))
        gt_flat = gt.ravel()
        mask_flat = mask.ravel()
        pred_flat = pred.ravel()
        # do not consider pixels out of the FOV
        noFOV_gt = gt_flat[mask_flat == True]
        noFOV_pred = pred_flat[mask_flat == True]

        # accumulate gt pixels and prediction pixels
        all_preds.append(noFOV_pred)
        all_gts.append(noFOV_gt)

    return np.hstack(all_preds), np.hstack(all_gts)

def cutoff_youden(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def cutoff_dice(preds, gts):
    dice_scores = []
    thresholds = np.linspace(0, 1, 256)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds>thresh
        dice_scores.append(dice_score(gts, hard_preds))
    dices = np.array(dice_scores)
    optimal_threshold = thresholds[dices.argmax()]
    return optimal_threshold

def cutoff_accuracy(preds, gts):
    accuracy_scores = []
    thresholds = np.linspace(0, 1, 256)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds > thresh
        accuracy_scores.append(accuracy_score(gts.astype(np.bool), hard_preds.astype(np.bool)))
    accuracies = np.array(accuracy_scores)
    optimal_threshold = thresholds[accuracies.argmax()]
    return optimal_threshold

def compute_performance(preds, gts, save_path=None, opt_threshold=None, cut_off='dice', mode='train'):

    fpr, tpr, thresholds = roc_curve(gts, preds)
    global_auc = auc(fpr, tpr)

    if save_path is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve')
        ll = 'AUC = {:4f}'.format(global_auc)
        plt.legend([ll], loc='lower right')
        fig.tight_layout()
        if opt_threshold is None:
            if mode=='train':
                plt.savefig(osp.join(save_path,'ROC_train.png'))
            elif mode=='val':
                plt.savefig(osp.join(save_path, 'ROC_val.png'))
        else:
            plt.savefig(osp.join(save_path, 'ROC_test.png'))

    if opt_threshold is None:
        if cut_off == 'acc':
            # this would be to get accuracy-maximizing threshold
            opt_threshold = cutoff_accuracy(preds, gts)
        elif cut_off == 'dice':
            # this would be to get dice-maximizing threshold
            opt_threshold = cutoff_dice(preds, gts)
        else:
            opt_threshold = cutoff_youden(fpr, tpr, thresholds)

    bin_preds = preds > opt_threshold

    acc = accuracy_score(gts, bin_preds)

    dice = dice_score(gts, bin_preds)

    mcc = matthews_corrcoef(gts.astype(int), bin_preds.astype(int))

    tn, fp, fn, tp = confusion_matrix(gts, preds > opt_threshold).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return global_auc, acc, dice, mcc, specificity, sensitivity, opt_threshold

if __name__ == '__main__':
    '''
    Note 1: in order to use analyze_results, one must first have corresponding predictions in path_to_preds/

    Note 2: below, experimentA/experimentB/experimentC contain models trained on DRIVE/CHASEDB/HRF respectively
    - evaluating on test set for the training set of the same database:
    python analyze_results.py --path_train_preds results/DRIVE/experiments/unet_drive
                              --path_test_preds results/DRIVE/experiments/unet_drive
                              --train_dataset DRIVE --test_dataset DRIVE
    - different test set
    python analyze_results.py --path_train_preds results/DRIVE/experiments/unet_drive
                              --path_test_preds results/CHASEDB/experiments/unet_drive
                              --train_dataset DRIVE --test_dataset CHASEDB
    '''
    exp_path = 'experiments/'
    results_path = 'results/'

    # gather parser parameters
    args = parser.parse_args()
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    path_train_preds = args.path_train_preds
    path_test_preds = args.path_test_preds
    cut_off = args.cut_off
    public = str2bool(args.public)

    if test_dataset in ['HRF', 'DR_HAGIS']:
        print(100*'-')
        print(8 * '-', ' For ' + test_dataset + ' , performance evaluation is **quite** heavy ', 8 * '-')
        print(100 * '-')
    if test_dataset in ['AUCKLAND_0', 'AUCKLAND_50', 'AUCKLAND_100', 'AUCKLAND_150']:
        print(100 * '-')
        print(8 * '-', ' For AUCKLAND , performance evaluation is **quite** heavy ', 8 * '-')
        print(100 * '-')

    print('* Analyzing performance in {} training set -- Obtaining optimal threshold maximizing {}'.format(train_dataset, cut_off))
    print('* Reading predictions from {}'.format(path_train_preds))
    save_path = osp.join(path_train_preds, 'perf')

    perf_csv_path = osp.join(save_path, 'training_performance.csv')
    csv_path = osp.join('data', train_dataset, 'train.csv')
    if 'HRF' in train_dataset: csv_path = osp.join('data', train_dataset, 'train_full_res.csv')
    preds, gts = get_labels_preds(path_train_preds, csv_path = csv_path)
    os.makedirs(save_path, exist_ok=True)
    global_auc_tr, acc_tr, dice_tr, mcc_tr, spec_tr, sens_tr, opt_thresh_tr =\
        compute_performance(preds, gts, save_path=save_path, opt_threshold=None, cut_off=cut_off, mode='train')

    perf_df_train = pd.DataFrame({'auc': global_auc_tr,
                                'acc': acc_tr,
                                'dice/F1': dice_tr,
                                'MCC': mcc_tr,
                                'spec': spec_tr,
                                'sens': sens_tr,
                                'opt_t': opt_thresh_tr}, index=[0])
    perf_df_train.to_csv(perf_csv_path, index=False)

    print('* Analyzing performance in {} validation set'.format(train_dataset))
    perf_csv_path = osp.join(save_path, 'validation_performance.csv')
    csv_path = osp.join('data', train_dataset, 'val.csv')
    if 'HRF' in train_dataset: csv_path = osp.join('data', train_dataset, 'val_full_res.csv')
    preds, gts = get_labels_preds(path_train_preds, csv_path = csv_path)
    global_auc_vl, acc_vl, dice_vl, mcc_vl, spec_vl, sens_vl, _ =\
        compute_performance(preds, gts, save_path=save_path, opt_threshold=opt_thresh_tr, cut_off=cut_off, mode='train')
    perf_df_train = pd.DataFrame({'auc': global_auc_vl,
                                  'acc': acc_vl,
                                  'dice/F1': dice_vl,
                                  'MCC': mcc_vl,
                                  'spec': spec_vl,
                                  'sens': sens_vl}, index=[0])
    perf_df_train.to_csv(perf_csv_path, index=False)

    print('*Analyzing performance in {} test set'.format(test_dataset))
    print('* Reading predictions from {}'.format(path_test_preds))
    save_path = osp.join(path_test_preds, 'perf')
    os.makedirs(save_path, exist_ok=True)
    perf_csv_path = osp.join(save_path, 'test_performance.csv')

    csv_name = 'test.csv' if train_dataset==test_dataset else 'test_all.csv'
    if train_dataset!=test_dataset:
        print('-- Testing on a different dataset (from training)')
    else:
        print('-- Testing on same data source as training')

    if public: path_test_csv = osp.join('data', test_dataset, csv_name)
    else: path_test_csv = osp.join('private_data', test_dataset, csv_name)

    preds, gts = get_labels_preds(path_test_preds, csv_path = path_test_csv)
    global_auc_test, acc_test, dice_test, mcc_test, spec_test, sens_test, _ = \
        compute_performance(preds, gts, save_path=save_path, opt_threshold=opt_thresh_tr)
    perf_df_test = pd.DataFrame({'auc': global_auc_test,
                                 'acc': acc_test,
                                 'dice/F1': dice_test,
                                 'MCC': mcc_test,
                                 'spec': spec_test,
                                 'sens': sens_test}, index=[0])
    perf_df_test.to_csv(perf_csv_path, index=False)
    print('* Done')
    print('AUC in Train/Val/Test set is {:.4f}/{:.4f}/{:.4f}'.format(global_auc_tr, global_auc_vl, global_auc_test))
    print('Accuracy in Train/Val/Test set is {:.4f}/{:.4f}/{:.4f}'.format(acc_tr, acc_vl, acc_test))
    print('Dice/F1 score in Train/Val/Test set is {:.4f}/{:.4f}/{:.4f}'.format(dice_tr, dice_vl, dice_test))
    print('MCC score in Train/Val/Test set is {:.4f}/{:.4f}/{:.4f}'.format(mcc_tr, mcc_vl, mcc_test))
    print('Specificity in Train/Val/Test set is {:.4f}/{:.4f}/{:.4f}'.format(spec_tr, spec_vl, spec_test))
    print('Sensitivity in Train/Val/Test set is {:.4f}/{:.4f}/{:.4f}'.format(sens_tr, sens_vl, sens_test))
    print('ROC curve plots saved to ', save_path)
    print('Perf csv saved at ', perf_csv_path)
