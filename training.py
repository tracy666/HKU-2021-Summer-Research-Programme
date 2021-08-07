import torch
import torch.nn as nn 
import torch.nn.functional as F 
import higher
import itertools
from utils.metrics import compute_metrics_test
criterion = nn.CrossEntropyLoss().cuda()
################################################################
def train_baseline_epoch(model,opt,train_loader):
    model.train()  
    for _, (inputs, labels) in enumerate(train_loader):   
        inputs, labels = inputs.cuda(), labels.long().cuda()
        opt.zero_grad()
        outputs        = model(inputs)
        minibatch_loss = criterion(outputs, labels)
        minibatch_loss.backward()
        opt.step()   
        
    return model 
##########################################################
#训练 L2RW算法
def train_L2RW_epoch(model,opt,train_loader,val_loader): 
    model.train()  
    val_loader = itertools.cycle(val_loader)   
    for _, (inputs, labels) in enumerate(train_loader):
        inputs,labels  = inputs.cuda(),labels.long().cuda()
        opt.zero_grad()
        with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
            # 1. Update meta model on training data
            meta_train_outputs = meta_model(inputs)
            meta_opt.reduction = 'none'
            meta_train_loss = criterion(meta_train_outputs,labels)
            eps = torch.zeros(meta_train_loss.size(), requires_grad=True).cuda()
            meta_train_loss = torch.sum(eps * meta_train_loss)
            meta_opt.step(meta_train_loss) 

            # 2. Compute grads of eps on meta validation data
            meta_inputs, meta_labels = next(val_loader)
            meta_inputs, meta_labels  = meta_inputs.cuda(),meta_labels.long().cuda() 

            criterion.reduction = 'mean'
            meta_val_outputs = meta_model(meta_inputs)
            meta_val_loss = criterion(meta_val_outputs, meta_labels)
            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

        # 3. Compute weights for current training batch
        w_tilde = torch.clamp(-eps_grads, min=0)
        l1_norm = torch.sum(w_tilde)
        if l1_norm != 0:
            w = w_tilde / l1_norm
        else:
            w = w_tilde 
        # 4. Train model on weighted batch
        outputs = model(inputs)
        criterion.reduction = 'none'
        minibatch_loss = criterion(outputs, labels)
        minibatch_loss = torch.sum(w * minibatch_loss)
        minibatch_loss.backward()
        opt.step()   
    return model 

###########################################################################
def train_TS_epoch(modelf,optf,modelc,optc,train_loader): 
    modelf.train()  
    modelc.train()  
    for idx, (inputs, labels) in enumerate(train_loader):    
        ############################################################################
        inputs,labels  = inputs.cuda(),labels.long().cuda()  
        optf.zero_grad()
        features = modelf(inputs) 
        # #只更新分类取器
        # 取样中心
        outputs = modelc(features)        
        c_features,c_labels = [],[]
        for i in range(7):
            indx = labels==i
            if torch.sum(indx)>0:
                c_features.append(features[indx].mean(dim=0,keepdim=True))
                c_labels.append(torch.zeros(1).type_as(labels) + i)         
        c_features = torch.cat(c_features,dim=0)
        c_labels   = torch.cat(c_labels)  
              
        # #只更新分类取器
        optc.zero_grad()
        meta_train_outputs    = modelc(c_features) 
        class_train_loss      = criterion(meta_train_outputs,c_labels)
        class_train_loss.backward()
        optc.step()        
        
        # #只更新特征提取器
        criterion.reduction = 'mean'
        outputs = modelc(modelf(inputs))
        minibatch_loss = criterion(outputs, labels)
        minibatch_loss.backward()
        optf.step()  
        
        
        
    return modelf,modelc

###########################################################################
def train_L2RW_TS_epoch(modelf,optf,modelc,optc,train_loader,val_loader): 
    modelf.train()  
    val_loader = itertools.cycle(val_loader)   
    for idx, (inputs, labels) in enumerate(train_loader):    
        ############################################################################
        inputs,labels  = inputs.cuda(),labels.long().cuda()  
        features = modelf(inputs)   
        with higher.innerloop_ctx(modelc, optc) as (meta_model, meta_opt):
            # 1. Update meta model on training data
            meta_train_outputs = meta_model(features)        
            criterion.reduction = 'none'
            meta_train_loss = criterion(meta_train_outputs,labels)
            eps = torch.zeros(meta_train_loss.size(), requires_grad=True).cuda()
            meta_train_loss = torch.sum(eps * meta_train_loss)
            meta_opt.step(meta_train_loss) 

            # 2. Compute grads of eps on meta validation data
            meta_inputs, meta_labels = next(val_loader)
            meta_inputs, meta_labels  = meta_inputs.cuda(),meta_labels.long().cuda() 
                        
            criterion.reduction = 'mean'  
            # meta_features = modelf(meta_inputs)      
            # c_features,c_labels = [],[]
            # for i in range(7):
            #     indx = meta_labels==i
            #     if torch.sum(indx)>0:
            #         c_features.append(meta_features[indx].mean(dim=0,keepdim=True))
            #         c_labels.append(torch.zeros(1).type_as(labels) + i)         
            # c_features = torch.cat(c_features,dim=0)
            # c_labels   = torch.cat(c_labels)  
                                   
            meta_val_outputs = meta_model(modelf(meta_inputs)) 
            meta_val_loss = criterion(meta_val_outputs, meta_labels)
            
            # meta_val_outputs = meta_model(c_features)
            # meta_val_loss = criterion(meta_val_outputs, c_labels)
            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

        # 3. Compute weights for current training batch
        w_tilde = torch.clamp(-eps_grads, min=0)
        l1_norm = torch.sum(w_tilde)
        if l1_norm != 0:
            w = w_tilde / l1_norm
        else:
            w = w_tilde 
        # 4. Train model on weighted batch
        optc.zero_grad()
        optf.zero_grad()   
        meta_train_outputs    = modelc(features)
        criterion.reduction   = 'none'
        class_train_loss      = criterion(meta_train_outputs,labels)
        minibatch_classloss   = torch.sum(w * class_train_loss)
        minibatch_classloss.backward()
        optc.step()
        optf.step()   
                        
        # #只更新特征提取器
        optc.zero_grad()
        outputs = modelc(modelf(inputs))
        criterion.reduction = 'mean'
        minibatch_loss = criterion(outputs, labels) 
        minibatch_loss.backward()
        optf.step()   
        
    return modelf,modelc

###############################################################
def testing(model,test_loader):
    model.eval()  
    gt,pred = [],[]
    for _, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(),labels.long().cuda()            
        outputs        = F.softmax(model(inputs),dim=1)  
        gt.append(torch.zeros_like(outputs).scatter_(1,labels.reshape(-1,1),1))
        pred.append(outputs.detach())  
        
    gt   = torch.cat(gt,dim=0)
    pred = torch.cat(pred,dim=0)
    Metrics = compute_metrics_test(gt,pred, competition=True) 
    return Metrics 

#####################################################################################################
def testing_twostage(model,test_loader):
    model[0].eval()  
    model[1].eval()  
    gt,pred = [],[]
    for _, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(),labels.long().cuda()            
        outputs        = F.softmax(model[1](model[0](inputs)),dim=1)  
        gt.append(torch.zeros_like(outputs).scatter_(1,labels.reshape(-1,1),1))
        pred.append(outputs.detach()) 
    # Metrics = metrics_test(torch.cat(gt,dim=0),torch.cat(pred,dim=0))   
    gt   = torch.cat(gt,dim=0)
    pred = torch.cat(pred,dim=0)
    Metrics = compute_metrics_test(gt,pred, competition=True) 
    
    return Metrics 

    