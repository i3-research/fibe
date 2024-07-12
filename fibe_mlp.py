
# Copyright (c) Sheng He/Yangming Ou, Boston Children's Hospital/Harvard Medical School, 2024
# email: sheng.he@childrens.harvard.edu, yangming.ou@childrens.harvard.edu

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import datetime
import numpy as np
import os

class data_preprocessing:
    def __init__(self,
                 task_type='Classification'):
        '''
        This class is used for precprocess the data
        '''
        self.task_type = task_type
        self.labeldict = None
        self.feat_mean = None
        self.feat_std = None
        self.outdim = None
        
        # to-do
        
        # save the labeldict, feat_mean, feat_std on somewhere
    
    def rescale_feature(self,feature_df):
        
        if self.feat_mean is None:
            self.feat_mean = np.mean(feature_df,axis=0,keepdims=True)
            
        if self.feat_std is None:
            self.feat_std = np.std(feature_df,axis=0,keepdims=True)
        
        #print('debug:')
        #print(self.feat_mean.shape)
        #print(feature_df.shape)
        re_feature_df = (feature_df-self.feat_mean)/self.feat_std
        
        return re_feature_df
    
    def normalize_label(self,score_df):
        if self.task_type.lower() == 'classification':
            if isinstance(score_df,list):
                
                if self.labeldict is None:
                    labels = set(score_df)
                    self.labeldict = {}
                    for idx,lab in enumerate(labels):
                        self.labeldict[lab]=idx
                
                
                
            elif isinstance(score_df,np.ndarray):
                if self.labeldict is None:
                    labels = np.unique(score_df)
                    self.labeldict = {}
                    for idx,lab in enumerate(labels):
                        self.labeldict[lab]=idx
                
            else:
                raise ValueError('The type of score_df %s is not list or numpy!'%type(score_df))
            
            self.outdim = len(self.labeldict)
            
            labelarray = []
            for score in score_df:
                labelarray.append(self.labeldict[score])
            
            labelarray = np.array(labelarray)
            
            
        else:
            labelarray = np.array(score_df)
            self.outdim = 1
        
        #print(labelarray)
        return labelarray
    
class MLP(nn.Module):
    def __init__(self,layers=[],
                 task_type='Classification',
                 loss_function=None,
                 feature_rescale=True,
                 learning_rate=0.0001,
                 lr_step_size=None,
                 batch_size = 8,
                 device = 'cuda',
                 isBatchNorm=False,
                 path_save_model=None,
                 name_save_model=None,
                 logfile=None):
        
        '''
        parameters
        @layers:                a list to indicate the number of neurons on each layers
                                note: include the dimension of the input (data dimension) and output (output dimension)
        @feature_rescale:       True for feature rescale
        @task_type:             classification or regression
        @loss_function:         MAE for regression or cross-entropy for classification 
        @learning_rate:         learning rate (default: 0.0001)
        @lr_step_size:          stpe size to reduce learning rate
        @batch_size:            batch size of each epoch
        @device:                'cuda' for GPU, otherwise CPU
        @isBatchNorm:           whether use the Batch Normalization after each layer
        @path_save_model:       folder for save the model (default: saved_model)
        @name_save_model:        name for the saved model
        @logfile:               the name of the logfile, is none, print to screen
        '''
        
        super().__init__()
        self.logfile = logfile
        
        if layers is None or len(layers) < 2:
            msg = 'The input parameter: layers is not valided\n'
            msg += 'The model must have at least two layers'
            raise ValueError(msg)
        
        self.task_type = task_type
        self.loss_function = loss_function
        self.feature_rescale = feature_rescale
        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.batch_size = batch_size
        self.device = device
        self.path_save_model = path_save_model
        self.name_save_model = name_save_model
        self.in_dims = layers[0]
        self.out_dims = layers[-1]
        
        # -- path for the saved model
        if self.path_save_model is None:
            self.path_save_model = 'saved_model'
        
        if not os.path.exists(self.path_save_model):
            os.makedirs(self.path_save_model)
            
        if self.name_save_model is None:
            ids = 'layers-'
            for la in layers:
                ids += '%d-'%la
            self.name_save_model = 'MLPmodels-'+ids[:-1]
        
        
        # -- data pre-process handler
        self.mlp_datapreprocess = None
        
        # -- creating model ---
        self.mlp_model = torch.nn.Sequential()
        nlen = len(layers)
        for n in range(1,nlen-1):
            self.mlp_model.add_module('fc_%d'%n,nn.Linear(layers[n-1],layers[n]))
            self.mlp_model.add_module('relu_%d'%n,nn.ReLU())
            if isBatchNorm:
                self.mlp_model.add_module('batch_%d'%n,nn.BatchNorm1d(layers[n]))
        self.mlp_model.add_module('classifier',nn.Linear(layers[-2],layers[-1]))
        self.mlp_model = self.mlp_model.to(self.device)
        
        self._printlog('model has been created')
        
        # --  loss function define
        self.criteria = None
        if self.loss_function is None and self.task_type.lower() == 'classification':
            self.criteria = nn.CrossEntropyLoss()
        
        elif isinstance(self.loss_function,str):
            if self.loss_function.lower() in ['cross-entropy','crossentropy']: 
                self.criteria = nn.CrossEntropyLoss()
        
        # -- optimizer define
        
        self.optimizer = optim.Adam(self.mlp_model.parameters(),lr=self.learning_rate)
        self.scheduler = None
        
        if lr_step_size is not None:
            self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=self.lr_step_size,gamma=0.5)
    
    def cross_validation(self,feature_df,score_df,nfold=5,
                         train_epoches=1000):
        
        nSamples,nDims = self._input_data_validate(feature_df)
        
        nfoldist = self._random_split(nSamples,nfold)
        
        for n in range(nfold):
            self._printlog('Starting the %d-fold cross validation'%n)
            
            trn_feat = np.empty([0,nDims])
            trn_label = []
            
            for t in range(nfold):
                
                if isinstance(score_df,list):
                    txt_label_pre = [score_df[i] for i in nfoldist[t]]
                    nabel = len(txt_label_pre)
                    
                elif isinstance(score_df,np.ndarray):
                    txt_label_pre = score_df[nfoldist[t]]
                    nabel=txt_label_pre.shape[0]
                else:
                    raise ValueError('Score type: %s does not supported!'%type(score_df))
            
                if t==n:
                    txt_feat = feature_df[nfoldist[t]]
                    txt_label = txt_label_pre
                    #print('nfold:',t,txt_feat.shape,nabel)
                else:
                    feat = feature_df[nfoldist[t]]
                    trn_feat = np.concatenate((trn_feat,feat),axis=0)
                    trn_label += list(txt_label_pre)
            
            trn_label = np.array(trn_label)
            self.train(trn_feat,trn_label,train_epoches)
            
            #print('nfold:',t,txt_feat.shape,nabel)
            
            acc = self.test(txt_feat,txt_label)
            
            msg = 'nfold:%d has the test accuracy:%.2f'%(n,acc*100)
            print(msg)
            self._printlog(msg)
            
                    
            
    # this the main function to fit a model
    def train(self,feature_df, score_df,train_epoches=1000):
        
        self.mlp_datapreprocess = data_preprocessing(
                task_type=self.task_type)
        
        
        score_label = self.mlp_datapreprocess.normalize_label(score_df)
        
        if self.out_dims != self.mlp_datapreprocess.outdim:
            raise ValueError('The data output dimenision %s does not match with the output model %d!'%(self.mlp_datapreprocess.outdim,self.out_dims))
        
        
        if self.feature_rescale:
            feature_df = self.mlp_datapreprocess.rescale_feature(feature_df)
        
        nSamples,nDims = self._input_data_validate(feature_df)
        
        nSteps = int(nSamples / self.batch_size) 
        nfoldist = self._random_split(nSamples,nSteps,
                                      withResidule=False)
        
        for epoch in range(train_epoches):
            
            if self.scheduler is not None:
                self.scheduler()
            
            lostlist = []
            
            for fod in nfoldist:
                input_feat = feature_df[fod]
                input_score = score_label[fod]
                
                input_feat = torch.from_numpy(input_feat).to(self.device)
                input_feat = input_feat.float()
                #print('input feature:',input_feat.shape)
                
                input_score = torch.from_numpy(input_score).to(self.device)
                input_score = input_score.float()
                #print('input label:',input_score.shape)
                
                self.optimizer.zero_grad()
                out = self.mlp_model(input_feat)
                #print('output:',out.shape)
                
                #loss = 0
                #print(self.criteria)
                
                if self.criteria is None:
                    if self.task_type.lower() == 'regression':
                        # default MAE
                        if out.ndim != input_score.ndim:
                            out = out.squeeze(-1)
                            
                        loss = torch.mean(torch.abs(out-input_score))
                else:
                    input_score = input_score.long()
                    loss = self.criteria(out,input_score)
                
                loss.backward()
            
                self.optimizer.step()
                #print('Epoch:',epoch,'loss is:',loss)
                lostlist.append(loss.detach().cpu().numpy())
            
            msg = 'Train epoch %d (Total:%d) has average loss: %.4f'%(epoch,train_epoches,np.mean(lostlist))
            self._printlog(msg)
            #print(msg)
            
        self.checkpoint('final')
                
    
    def test(self,feature_df,score_df=None):
        
        score_label = self.mlp_datapreprocess.normalize_label(score_df)
        
        if self.out_dims != self.mlp_datapreprocess.outdim:
            raise ValueError('The data output dimenision %s does not match with the output model %d!'%(self.mlp_datapreprocess.outdim,self.out_dims))
        
        
        if self.feature_rescale:
            feature_df = self.mlp_datapreprocess.rescale_feature(feature_df)
        
        nSamples,nDims = self._input_data_validate(feature_df)
        

        if feature_df.shape[0] != score_label.shape[0]:
            raise ValueError('feature sample: %d and label sample: %d does not matched!'%(feature_df.shape[0],score_label.shape[0]))
        
        nSteps = int(nSamples / self.batch_size) 
        nfoldist = self._random_split(nSamples,nSteps,
                                      withResidule=False)
        
        
  
        accuracylist = []
        
        for fod in nfoldist:
            input_feat = feature_df[fod]
            input_score = score_label[fod]
            
            input_feat = torch.from_numpy(input_feat).to(self.device)
            input_feat = input_feat.float()
            #print('input feature:',input_feat.shape)
            
            input_score = torch.from_numpy(input_score).to(self.device)
            input_score = input_score.float()
            #print('input label:',input_score.shape)
            
            
            with torch.no_grad():
                out = self.mlp_model(input_feat)
                
                #print(out.shape,input_score.shape)
                if self.task_type.lower()=='classification':
                    value,pred = torch.topk(out,1,dim=1,
                                            largest=True,
                                            sorted=True)
                    pred = pred.squeeze(-1)
                    accuracy = (pred==input_score.long()).detach().cpu().numpy()
                    accuracylist += list(accuracy)
                elif self.task_type.lower()=='regression':
                    """ To-do """
                    pass
            
            
        accuracy = np.mean(accuracylist)
        
        return accuracy
                    
            
            
    
    
    def checkpoint(self,epoch):
        save_name = self.name_save_model+'-model-epoch-{}.path'.format(epoch)
        model_out_path = os.path.join(self.path_save_model, save_name)
        torch.save(self.mlp_model.state_dict(),model_out_path)
        self._printlog('Model %s has been saved!'%model_out_path)
    
    def loadmodel(self,epoch=None,model_path=None):
        if model_path is None:
            epoch = 'final' if epoch is None else epoch
            
            save_name = self.name_save_model+'-model-epoch-{}.path'.format(epoch)
            model_path = os.path.join(self.path_save_model, save_name)
        
        try:
            self.mlp_model.load_state_dict(torch.load(model_path,map_location=self.device))
            self._printlog('Model %s has been load successfully!'%model_path)
        except Exception as e:
            self._printlog('Failed loading Model %s'%model_path)
            raise ValueError(e)
            
            
        
    def _input_data_validate(self,feature_df):
        if isinstance(feature_df,np.ndarray):
            if feature_df.ndim != 2:
                raise ValueError('The dimesion of the input %d is not 2!'%feature_df.ndim)
        else:
            raise ValueError('The input type is not np.array')
        
        # check the dimensions
        nSamples,nDims = feature_df.shape
        
        if nSamples < 1:
            raise ValueError('The number of samples %d is too small!'%nSamples)
        
        if nDims < 1:
            raise ValueError('The dimensions of samples %d is too small!'%nDims)
            
        if nDims != self.in_dims:
            msg = ' The input data dimension %d does not match the input model layer: %d'%(nDims,self.in_dims)
            raise ValueError(msg)
            
        return nSamples,nDims
        
    def _random_split(self,nSamples,nFold,withResidule = True):
        
        if nSamples < nFold:
            msg='_random_split error: nSamples %d less than nFold: %d'%(nSamples,nFold)
            raise ValueError(msg)
            
        np.random.seed(int(datetime.datetime.now().timestamp()))
        ridx = np.random.permutation(nSamples)
        nsteps = int(nSamples/nFold)
        nres = nSamples - nsteps * nFold
        
        nfoldist = []
        for n in range(nFold):
            nfoldist.append(list(ridx[n*nsteps:n*nsteps+nsteps]))
        
        # deal with the residual samples
        if withResidule and nres > 0 :
            nStart = nsteps * nFold
            pts = np.random.permutation(nFold)
            for n in range(nres):
                nfoldist[pts[n]].append(ridx[nStart+n])
            
        #print(nfoldist)
        return nfoldist
        
    def _printlog(self,msg):
        # get time string
        curtime = datetime.datetime.now()
        day = '%d-%d-%d'%(curtime.year,curtime.month,curtime.day)
        time = ' %d:%d:%d '%(curtime.hour,curtime.minute,curtime.second)
        dtime = day+time
        
        
        if self.logfile is None:
            print(dtime,msg)
        else:
            
            with open(self.logfile,'a') as fp:
                fp.write('%s\n'%(dtime+msg))
                

    
