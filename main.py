import os
import numpy as np
import random
import pandas as pd
import argparse
from sklearn.utils import shuffle
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import Network,classifer,featureNet
from data_loaders import get_training,get_testing,get_validation
from training import train_baseline_epoch,train_L2RW_epoch,train_L2RW_TS_epoch,testing
##########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/ISIC2018/training_data/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/data/ISIC2018/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/data/ISIC2018/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/data/ISIC2018/skin/testing.csv', help='testing set csv file')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size per gpu')
parser.add_argument('--lr', type=float,  default=1e-3, help='learning rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### model
parser.add_argument('--mode',   type=str,  default="twostage", help='model select')
parser.add_argument('--val_sample', type=str,  default="balance_random", help='validation sample')
##########################################################################################
if __name__ == "__main__":
    args = parser.parse_args()
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] =  str(args.gpu)
   #######################################################
   #create model      
    #################################################################################################  
    if args.mode == 'twostage':
        fmodel = featureNet().cuda()
        cmodel = classifer(fmodel.f_num,7).cuda()
        model = [fmodel,cmodel]
        opts = [optim.SGD(fmodel.parameters(), lr = args.lr),
              optim.SGD(cmodel.parameters(), lr = args.lr)]
        scheduler = optim.lr_scheduler.MultiStepLR(opts[0],milestones=[80,100], gamma=0.1)  
    else:
        model = Network().cuda()  
        opts = optim.SGD(model.features.parameters(), lr = args.lr) 
        scheduler = optim.lr_scheduler.MultiStepLR(opts,milestones=[80,100], gamma=0.1) 
    #################################################################################################             
    #data loader
    train_loader,test_loader = get_training(batch_size = args.batch_size),get_testing(batch_size = args.batch_size*2)  
    #######################################################################
    #traning baseline model
    for epoch in range(args.epochs): 
        model  = train_baseline_epoch(model,opts,train_loader = train_loader)  
        Metris = testing(model,test_loader)  
        scheduler.step()
        print('epoch',epoch,Metris)          
        save_mode_path = os.path.join('./checkpoint/baseline/', 'epoch_' + str(epoch+1) + '.pth')
        if epoch%10 == 0:
            torch.save(model.state_dict(), save_mode_path)
            
    #traning L2RW model
    val_loader = get_validation(mode = 'balance_random')    
    for epoch in range(args.epochs): 
        # val_loader = get_validation(mode = 'balance_random') 
        model  = train_L2RW_epoch(model,opts,train_loader = train_loader,val_loader = val_loader)  
        Metris = testing(model,test_loader)  
        scheduler.step()
        print('epoch',epoch,Metris)          
        save_mode_path = os.path.join('./checkpoint/baseline/', 'epoch_' + str(epoch+1) + '.pth')
        if epoch%10 == 0:
            torch.save(model.state_dict(), save_mode_path)  
              
    #traning L2RW  and twostage model    
    val_loader = get_validation(mode = 'balance_random')        
    for epoch in range(args.epochs): 
        # val_loader = get_validation(mode = 'balance_random') 
        model  = train_L2RW_epoch(model,opts,train_loader = train_loader,val_loader = val_loader)  
        Metris = testing(model,test_loader)  
        scheduler.step()
        print('epoch',epoch,Metris)          
        save_mode_path = os.path.join('./checkpoint/baseline/', 'epoch_' + str(epoch+1) + '.pth')
        if epoch%10 == 0:
            torch.save(model.state_dict(), save_mode_path) 
            
    select_loader = get_training(batch_size = args.batch_size,shuffle=False) if args.val_sample == 'balance_loss' else None    
    val_loader = get_validation(mode = args.val_sample,select_loader= select_loader, model = None)        
    for epoch in range(args.epochs): 
        # model  = train_baseline_epoch(model,opts,train_loader = train_loader)       
        # val_loader = get_validation(mode = args.val_sample,select_loader=select_loader, model = model)
        # model    = train_L2RW_epoch(model,opts,train_loader = train_loader,val_loader = val_loader)  
        model    = train_L2RW_TS_epoch(model,opts,train_loader = train_loader,val_loader = val_loader) 
         
        Metris = testing(model,test_loader)     
        
        scheduler.step()
        print('epoch',epoch, 
              'auc',round(np.mean(Metris[0]),4), 
              'sen',round(np.mean(Metris[1]),4), 
              'spe',round(np.mean(Metris[2]),4), 
              'acc',round(np.mean(Metris[3]),4),               
              'f1',round(np.mean(Metris[4]),4))  
        
        save_mode_path = os.path.join('./checkpoint/baseline/', 'epoch_' + str(epoch+1) + '.pth')
        if epoch%10 == 0:
            torch.save(model.state_dict(), save_mode_path)
   