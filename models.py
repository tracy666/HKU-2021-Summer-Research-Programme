import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
####################################################################################
class Network(nn.Module): 
    def __init__(self, net_name = 'resnet18', class_num = 7):
        super(Network, self).__init__()
        model      = models.__dict__[net_name](pretrained=False)  
        self.f_num = model.fc.in_features
        self.features    = nn.Sequential(*list(model.children())[:-1])
        self.classifer       = nn.Linear(self.f_num, class_num)
    def forward(self, x): 
        out_features   = self.features(x).reshape(-1,self.f_num)  
        out_classifer  = self.classifer(out_features)            
        return out_classifer

class featureNet(nn.Module): 
    def __init__(self, net_name = 'resnet18', class_num = 7):
        super(featureNet, self).__init__()
        model      = models.__dict__[net_name](pretrained=False)  
        self.f_num = model.fc.in_features
        self.features    = nn.Sequential(*list(model.children())[:-1]) 
    def forward(self, x): 
        out_features   = self.features(x).reshape(-1,self.f_num)              
        return out_features    
#####################################################
class classifer(nn.Module): 
    def __init__(self,f_num, class_num = 7):
        super(classifer, self).__init__()
        self.linear       = nn.Linear(f_num, class_num)
        
    def forward(self, x):          
        return self.linear(x)      

##############################################    
if __name__ == "__main__":
    inputs = torch.zeros(1,3,244,244)
    model = Network()
    outputs = model.model(inputs)
    print(outputs.size())
    print(model.features)
    