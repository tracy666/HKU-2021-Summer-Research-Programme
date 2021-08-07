from numpy.lib.function_base import select
import torch
import pandas as pd
import numpy as np
from torch.functional import cartesian_prod
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random

criterion = nn.CrossEntropyLoss().cuda()
########################################################
data_dir   = '/data/ISIC2018/'
image_path =  'training_data/'

classses    = np.zeros(7)
###########################################################################################
transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) 
########################################################
transform_test =   transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])   
###########################################################################      
class CreateDatasetFromImages(Dataset):
    def __init__(self, csv_path, file_path, transform):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """       
        self.file_path = file_path
        self.transform = transform
        # 读取 csv 文件
        #利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path) 
        #第1列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])  #self.data_info.iloc[1:,0表示读取第一列，从第二行开始一直读取到最后一行
        # 第8列是图像的 label
        self.label_arr = np.argmax(np.asarray(self.data_info.iloc[:, 1:8]),axis=1)
        # self.label_arr = np.asarray(self.data_info.iloc[:, 1:8])
        # 数据总长度
        self.data_len = len(self.data_info.index)
        
    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]
        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name + ".jpg")
        
        #如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        # if img_as_img.mode != 'L':
        #     img_as_img = img_as_img.convert('L')
         
        img_as_img = self.transform(img_as_img)

        # 得到图像的 label
        label = self.label_arr[index]

        return img_as_img, label  #返回每一个index对应的图片数据和对应的label
    
    def __len__(self):
        return self.data_len    
##########################################################################
def get_loader(csv_path,image_path,batch_size=64,transform = transform_train,shuffle=True): 
    
    dataloader = DataLoader(
                            dataset    =  CreateDatasetFromImages(
                                                            csv_path  = csv_path,
                                                            file_path = image_path,
                                                            transform = transform),
                            batch_size = batch_size, 
                            shuffle    = shuffle)
    
    return dataloader
##########################################################################
def get_training(csv_path = '/data/ISIC2018/skin/training.csv',
                 image_path = '/data/ISIC2018/training_data/',
                 batch_size = 64,transform = transform_train,
                 shuffle=True):
        
    return get_loader(csv_path = csv_path, image_path = image_path,
                      batch_size = batch_size,
                      transform = transform,
                      shuffle   = shuffle)

##########################################################################    
def get_testing(csv_path = '/data/ISIC2018/skin/testing_all.csv',
                 image_path = '/data/ISIC2018/training_data/',
                 batch_size = 64,
                 transform = transform_test,
                 shuffle=False):
        
    return get_loader(csv_path = csv_path, image_path = image_path,
                      batch_size = batch_size,
                      transform = transform,
                      shuffle   = shuffle)

##########################################################################  
def get_Blance(data_list):
    #load all training data      
    Index    = []
    Max_num = len(data_list.index) // 7 
    classses = np.zeros(7)
    labels = np.argmax(np.asarray(data_list.iloc[:, 1:8]),axis=1) 
    while np.sum(classses) < Max_num * 7:        
        for i in np.random.permutation(len(data_list.index)):
            if classses[labels[i]] < Max_num:
                Index.append(i)
                classses[labels[i]] += 1  
    Index = np.asarray(Index)    
    return Index  
##################################################################
def get_Blance_random(data_list,num = 32):
    #load all training data    
    Index    = []
    for i in np.random.permutation(len(data_list.index)):
        label = np.argmax(np.asarray(data_list.iloc[i, 1:8]),axis=0)
        if classses[label] < num:
            Index.append(i)
            classses[label] += 1   
    Index = np.asarray(Index)   
    return Index  
##########################################################################
def get_validation_sample(mode = 'balance',
                   csv_path = '/data/ISIC2018/skin/training.csv',
                   image_path = '/data/ISIC2018/training_data/',
                   batch_size = 28,
                   transform = transform_train,
                   shuffle=True,
                   num = 8):
    ##############################################
    data_list = pd.read_csv(csv_path)      
    if mode == 'random':
        Index = np.random.permutation(len(data_list.index))[:num*7]        
    if mode == 'balance':
        Index = get_Blance_random(data_list)
    if mode == 'balance_all':
        Index = get_Blance(data_list)
    #############################################
    file_name = 'Blance' + str(random.randint(0,10000)) + '.csv'             
    data_list.loc[Index].to_csv(file_name,index=0,header=None)     
    ##############################################
    dataloader     = get_loader(csv_path = file_name, 
                                image_path = image_path,
                                batch_size = batch_size,
                                transform = transform,
                                shuffle   = True) 
    
    os.remove(file_name)    
    print(mode, 'validation selected!')
    
    return dataloader

#############################################################
def get_Blance_loss(model,data_loader,num = 8):
    #load all training data  
    criterion.reduction = 'none'    
    labels,losses = [],[]     
    for _,(inputs,label) in enumerate(data_loader):        
        inputs,label  = inputs.cuda(),label.cuda()      
        loss = criterion(model[1](model[0](inputs)),label)
        losses.append(loss.detach())
        labels.append(label)  
    losses   = torch.cat(losses).reshape(-1)
    labels   = torch.cat(labels).reshape(-1)
    #############################################################
    Index = []
    sort_loss = losses.sort()[1]
    
    classses = torch.zeros(7).type_as(labels)
    for idx in sort_loss:        
        if classses[labels[idx]] < num:
            Index.append(idx.cpu().numpy())
            classses[labels[idx]] += 1   
            
    # all_index = torch.arange(0,features.size(0))
    # for c in range(7):
    #     idx        = labels == c
    #     c_features = features[idx] 
    #     dist       = torch.abs(c_features - c_features.mean(dim=0,keepdim=True)).sum(dim=0)
    #     sort_dist  = dist.sort()[1][:num] 
    #     Index.append(all_index[sort_dist])
     
    Index = np.asarray(Index)                     
    return Index   

#######################################################################
def get_Blance_kemeans(model,data_loader,num = 8):
    #load all training data      
    labels,features = [],[] 
    model.eval()     
    for _,(inputs,label) in enumerate(data_loader):        
        inputs,label  = inputs.cuda(),label.cuda()      
        feature = model(inputs)
        features.append(feature.detach())
        labels.append(label)  
    features = torch.cat(features,dim=0).cpu().numpy()  
    labels   = torch.cat(labels,dim=0).cpu().numpy() 
    ############################################################# 
    centroids = []
    for i in range(7):
        features_one = features[labels==i]
        centroid = KMeans(n_clusters=num).fit(features_one).cluster_centers_ 
        centroids.append(centroid)
        
    centroids = np.vstack(centroids) 
    print(centroids.shape) 
    dist  = cosine_similarity(centroids,features)
    Index =  np.argmax(dist,axis=1) 
                  
    return Index   
#####################################################################
def get_validation_model(csv_path = '/data/ISIC2018/skin/training.csv',
                   image_path = '/data/ISIC2018/training_data/',
                   batch_size = 28,
                   transform = transform_test,
                   shuffle=True,                   
                   num = 8,
                   data_loader = None,
                   model = None):
    ##############################################
    data_list = pd.read_csv(csv_path)       
    # Index = get_Blance_loss(model,data_loader,num) 
    Index = get_Blance_kemeans(model,data_loader,num)
    file_name = 'Blance' + str(random.randint(0,10000)) + '.csv'
    data_list.loc[Index].to_csv(file_name,index=0,header=None)     
    ##############################################
    dataloader     = get_loader(csv_path = file_name, 
                                image_path = image_path,
                                batch_size = batch_size,
                                transform = transform,
                                shuffle   = shuffle) 
    os.remove(file_name)    
    print('validation selected!')
    return dataloader