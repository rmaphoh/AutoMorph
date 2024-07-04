
import argparse
import logging
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from dataset import BasicDataset_OUT
from torch.utils.data import DataLoader
from model import Resnet101_fl, InceptionV3_fl, Densenet161_fl, Resnext101_32x8d_fl, MobilenetV2_fl, Vgg16_bn_fl, Efficientnet_fl

AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','..')

def test_net(model_fl_1,
            model_fl_2,
            model_fl_3,
            model_fl_4,
            model_fl_5,
            model_fl_6,
            model_fl_7,
            model_fl_8,
              test_dir,
              device,
              epochs=5,
              batch_size=20,
              image_size=(512,512),
              ):

    storage_path ="Ensemble_exp_{}/{}/train_on_{}/test_on_{}/".format(args.task, args.load, args.model, args.dataset)
    n_classes = args.n_class
    # create files

    if not os.path.isdir(storage_path):
        os.makedirs(storage_path)
    
    dataset = BasicDataset_OUT(test_dir, image_size, n_classes, train_or=False)
        
    n_test = len(dataset)
    val_loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    
    prediction_decode_list = []
    filename_list = []
    prediction_list_mean = []
    prediction_list_std = []
    mask_pred_tensor_small_all = 0
    for epoch in range(epochs):

        model_fl_1.eval()
        model_fl_2.eval()
        model_fl_3.eval()
        model_fl_4.eval()
        model_fl_5.eval()
        model_fl_6.eval()
        model_fl_7.eval()
        model_fl_8.eval()


        with tqdm(total=n_test, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in val_loader:
                imgs = batch['image']
                filename = batch['img_file'][0]
                mask_pred_tensor_small_all = 0
                imgs = imgs.to(device=device, dtype=torch.float32)
                ##################sigmoid or softmax

                prediction_list = []
                with torch.no_grad():
                    #print(real_patch.size())
                    prediction = model_fl_1(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_2(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_3(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_4(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_5(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_6(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_7(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    prediction = model_fl_8(imgs)
                    prediction_softmax = nn.Softmax(dim=1)(prediction)
                    mask_pred_tensor_small = prediction_softmax.clone().detach()
                    mask_pred_tensor_small_all+=mask_pred_tensor_small.type(torch.FloatTensor)
                    prediction_list.append(mask_pred_tensor_small.type(torch.FloatTensor).cpu().detach().numpy())

                    mask_pred_tensor_small_all = mask_pred_tensor_small_all/8

                    _,prediction_decode = torch.max(mask_pred_tensor_small_all, 1)
                    
                    prediction_list = np.array(prediction_list)
                    prediction_list_mean.extend(np.mean(prediction_list, axis=0))
                    prediction_list_std.extend(np.std(prediction_list, axis=0))

                    prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
                    filename_list.extend(filename)
                    pbar.update(imgs.shape[0])
        
    Data4stage2 = pd.DataFrame({'Name':filename_list, 'softmax_good':np.array(prediction_list_mean)[:,0],'softmax_usable':np.array(prediction_list_mean)[:,1],'softmax_bad':np.array(prediction_list_mean)[:,2], 'good_sd':np.array(prediction_list_std)[:,0],'usable_sd':np.array(prediction_list_std)[:,1],'bad_sd':np.array(prediction_list_std)[:,2], 'Prediction': prediction_decode_list})

    if not os.path.exists(f'{AUTOMORPH_DATA}/Results/M1'):
        os.makedirs(f'{AUTOMORPH_DATA}/Results/M1')
    Data4stage2.to_csv(f'{AUTOMORPH_DATA}/Results/M1/results_ensemble.csv', index = None, encoding='utf8')




def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=240,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=6,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--train_on_dataset', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    parser.add_argument( '-dir', '--test_csv_dir', metavar='tcd', type=str,
                        help='path to the csv', dest='test_dir')
    parser.add_argument( '-n_class', '--n_classes', dest='n_class', type=int, default=False,
                        help='number of class')
    parser.add_argument( '-d','--test_on_dataset', dest='dataset', type=str, 
                        help='dataset name')
    parser.add_argument( '-t', '--task_name', dest='task', type=str,
                        help='The task name')
    parser.add_argument( '-r', '--round', dest='round', type=int, 
                        help='Number of round') 
    parser.add_argument( '-m', '--model', dest='model', type=str, 
                        help='Backbone of the model')     
    parser.add_argument('--seed_num', type=int, default=42, help='Validation split seed', dest='seed')
    parser.add_argument('--local_rank', default=0, type=int) 

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    

    # Check if CUDA is available
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using CUDA...")
        device = torch.device("cuda",args.local_rank)
    elif torch.backends.mps.is_available():  # Check if MPS is available (for macOS)
        logging.info("MPS is available. Using MPS...")
        device = torch.device("mps")
    else:
        logging.info("Neither CUDA nor MPS is available. Using CPU...")
        device = torch.device("cpu")

    logging.info(f'Using device {device}')

    test_dir = args.test_dir
    dataset=args.dataset
    img_size= (512,512)

    if args.model=='inceptionv3':
        model_fl = InceptionV3_fl(pretrained=True)
    if args.model=='densenet161':
        model_fl = Densenet161_fl(pretrained=True)
    if args.model == 'resnet101':   
        model_fl = Resnet101_fl(pretrained=True)
    if args.model == 'resnext101':   
        model_fl_1 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_2 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_3 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_4 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_5 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_6 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_7 = Resnext101_32x8d_fl(pretrained=True)
        model_fl_8 = Resnext101_32x8d_fl(pretrained=True)
    if args.model == 'efficientnet':   
        model_fl_1 = Efficientnet_fl(pretrained=True)
        model_fl_2 = Efficientnet_fl(pretrained=True)
        model_fl_3 = Efficientnet_fl(pretrained=True)
        model_fl_4 = Efficientnet_fl(pretrained=True)
        model_fl_5 = Efficientnet_fl(pretrained=True)
        model_fl_6 = Efficientnet_fl(pretrained=True)
        model_fl_7 = Efficientnet_fl(pretrained=True)
        model_fl_8 = Efficientnet_fl(pretrained=True)
    if args.model == 'mobilenetv2':   
        model_fl = MobilenetV2_fl(pretrained=True)
    if args.model == 'vgg16bn':   
        model_fl = Vgg16_bn_fl(pretrained=True)

    checkpoint_path_1 = './{}/{}/{}/7_seed_28/best_loss_checkpoint.pth'.format(args.task, args.load, args.model )
    checkpoint_path_2 = './{}/{}/{}/6_seed_30/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    checkpoint_path_3 = './{}/{}/{}/5_seed_32/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    checkpoint_path_4 = './{}/{}/{}/4_seed_34/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    checkpoint_path_5 = './{}/{}/{}/3_seed_36/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    checkpoint_path_6 = './{}/{}/{}/2_seed_38/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    checkpoint_path_7 = './{}/{}/{}/1_seed_40/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    checkpoint_path_8 = './{}/{}/{}/0_seed_42/best_loss_checkpoint.pth'.format(args.task, args.load, args.model  )
    
    model_fl_1.to(device=device)
    model_fl_2.to(device=device)
    model_fl_3.to(device=device)
    model_fl_4.to(device=device)
    model_fl_5.to(device=device)
    model_fl_6.to(device=device)
    model_fl_7.to(device=device)
    model_fl_8.to(device=device)

    #map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    if args.load:
        model_fl_1.load_state_dict(
            torch.load(checkpoint_path_1, map_location=device)
        )
        model_fl_2.load_state_dict(
            torch.load(checkpoint_path_2, map_location=device)
        )
        model_fl_3.load_state_dict(
            torch.load(checkpoint_path_3, map_location=device)
        )
        model_fl_4.load_state_dict(
            torch.load(checkpoint_path_4, map_location=device)
        )
        model_fl_5.load_state_dict(
            torch.load(checkpoint_path_5, map_location=device)
        )
        model_fl_6.load_state_dict(
            torch.load(checkpoint_path_6, map_location=device)
        )
        model_fl_7.load_state_dict(
            torch.load(checkpoint_path_7, map_location=device)
        )
        model_fl_8.load_state_dict(
            torch.load(checkpoint_path_8, map_location=device)
        )

    try:
        test_net(model_fl_1,
                 model_fl_2,
                 model_fl_3,
                 model_fl_4,
                 model_fl_5,
                 model_fl_6,
                 model_fl_7,
                 model_fl_8,
                  test_dir,
                  device=device,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  image_size=img_size)
    except KeyboardInterrupt:
        torch.save(model_fl.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)







