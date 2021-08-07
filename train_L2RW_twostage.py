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
from models import classifer,featureNet
from data_loaders import get_training,get_testing,get_validation_sample,get_validation_model
from training import train_L2RW_TS_epoch,testing_twostage
##########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/ISIC2018/training_data/', help='dataset root dir')
parser.add_argument('--csv_file_train', type=str, default='/data/ISIC2018/skin/training.csv', help='training set csv file')
parser.add_argument('--csv_file_val', type=str, default='/data/ISIC2018/skin/validation.csv', help='validation set csv file')
parser.add_argument('--csv_file_test', type=str, default='/data/ISIC2018/skin/testing.csv', help='testing set csv file')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size per gpu')
parser.add_argument('--lr', type=float,  default=1e-3, help='learning rate')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--mode', type=str,  default='balance_all', help='GPU to use')
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
    modelf = featureNet().cuda()
    modelc = classifer(modelf.f_num,7).cuda()
    
    
    optsf  = optim.SGD(modelf.parameters(), lr = args.lr)
    optsc  = optim.SGD(modelc.parameters(), lr = args.lr)
    
    schedulerf = optim.lr_scheduler.MultiStepLR(optsf,milestones=[80,90], gamma=0.1)  
    schedulerc = optim.lr_scheduler.MultiStepLR(optsc,milestones=[80,90], gamma=0.1) 
    #################################################################################################             
    #data loader
    train_loader = get_training(batch_size = args.batch_size)
    test_loader = get_testing(batch_size = args.batch_size*2)   
    #######################################################################
    #traning baseline model    
    save_dir = './checkpoint/L2RW_twostage/' + 'each_' + args.mode
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    ####################################################################### 
    Metris = [] 
    # select_loader = get_training(batch_size = args.batch_size,shuffle=False)
    val_loader = get_validation_sample(mode = args.mode,batch_size = args.batch_size)       
    for epoch in range(args.epochs): 
        # val_loader = get_validation_model(model = modelf,data_loader = select_loader)  
        modelf,modelc  = train_L2RW_TS_epoch(modelf,optsf,modelc,optsc,train_loader = train_loader,val_loader = val_loader)  
        metris = testing_twostage((modelf,modelc),test_loader)  
        schedulerf.step()
        schedulerc.step()
        print('epoch',epoch,metris)   
        Metris.append(metris)       
        if epoch%20 == 0 or epoch > 90:
            torch.save(modelf.state_dict(), save_dir+ '/Feature_epoch_' + str(epoch+1) + '.pth')
            torch.save(modelc.state_dict(), save_dir+ '/Classifer_epoch_' + str(epoch+1) + '.pth')
            
    pd.DataFrame(Metris).to_csv(save_dir + '/results.csv')
    
    
   