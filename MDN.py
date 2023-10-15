#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from joblib import Parallel, delayed
from torch.autograd import Variable
import sys
import statsmodels.api as sm
import scipy.stats as st
import torch.distributions as dist

path = os.getcwd()

torch.manual_seed(1)
np.random.seed(1)
def data_to_tensor(trainx, trainy, testx, testy):
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    testx = np.array(testx)
    testy = np.array(testy)

    # whether input_size=1 or not
    size = np.size(trainx)
    size = np.array(size)
    shape = np.shape(trainx)

    trainX = torch.tensor(trainx, dtype=torch.float)
    trainY = torch.tensor(trainy, dtype=torch.float)

    testX = torch.tensor(testx, dtype=torch.float)
    testY = torch.tensor(testy, dtype=torch.float)

    return trainX, trainY, testX, testY


def data_load(trainX, trainY, testX, testY, train_size, test_size):
    torch_train = Data.TensorDataset(trainX, trainY)
    torch_test = Data.TensorDataset(testX, testY)

    loader_training = Data.DataLoader(
        dataset=torch_train,
        batch_size=train_size,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=0  # subprocesses for loading data
    )

    loader_test = Data.DataLoader(
        dataset=torch_test,
        batch_size=test_size,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=0  # subprocesses for loading data
    )
    return loader_training, loader_test

def normalization(trainx, trainy, testx, testy):
    # x
    x_mean = np.mean(trainx)
    x_max = np.max(trainx)
    x_min = np.min(trainx)
    x_std = np.std(trainx)

    ##
    trainx_trans = (trainx - x_mean) / x_std
    testx_trans = (testx - x_mean) / x_std

    # train_x1 = (trainx-x_min)/ (x_max-x_min)
    # test_x1 = (testx-x_min)/(x_max-x_min)

    ##y
    y_mean = np.mean(trainy)
    y_max = np.max(trainy)
    y_min = np.min(trainy)
    y_std = np.std(trainy)

    trainy_trans = (trainy - y_mean) / y_std
    testy_trans = (testy - y_mean) / y_std
    return trainx_trans, trainy_trans, testx_trans, testy_trans, x_mean, x_std, y_mean, y_std


def result_reverse_Normalization(y, y_std, y_mean, y_trainx, index, residual_index):
    data_len = len(y_trainx)
    if index == 1:
        pred = y * y_std + y_mean

    else:
        pred = y
    if residual_index == 1:

        for num in range(data_len):
            pred[num, :] = pred[num, :] + y_trainx[num]  # pred[len,samples]
    else:
        pred = pred

    return pred

class LSTM(nn.Module):
    #Here, LSTM->ANN
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        # super(LSTM, self).__init__()
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = dropout
        self.l1=nn.Linear(input_size,hidden_size)

        self.relu1=nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        m=x.shape[0]
        x=x.reshape([m,1])
        #########################
        out=self.l1(x)
        r_out=self.relu1(out)

        r_out=self.l2(r_out)

        n_component = int(self.output_size / 3)

        mean = r_out[:, 0:n_component].squeeze(dim=1)
        v = r_out[:, n_component:2 * n_component].squeeze(dim=1)
        w = r_out[:, 2 * n_component:3 * n_component].squeeze(dim=1)
        return (mean, v, w)

def cov(a, b):
    a_mean = torch.mean(a)
    b_mean = torch.mean(b)
    N = a.shape[0]
    sum1 = 0

    for i in range(N):
        sum1 = sum1 + (a[i] - a_mean) * (b[i] - b_mean)
    sum1 = sum1 / N
    return sum1

def auto_r(out, obs, lag):
    res = obs - out
    a = res[:-1]
    b = res[lag:]
    std_a = torch.std(a)
    std_b = torch.std(b)
    cov_auto = cov(a, b)
    return cov_auto / (std_a * std_b)

def auto_all_r(out, obs, lag):
    out_np = out.detach().numpy()
    obs_np = obs.detach().numpy()
    res = obs_np - out_np
    res_r = sm.tsa.stattools.acf(res, nlags=lag, fft=True)
    res_r_lag = res_r[1:lag]
    
    mean_r = np.mean(res_r_lag)
    print('mean—_r', mean_r)
    mean_r_torch = torch.tensor(mean_r, dtype=torch.float, requires_grad=True)
    return mean_r_torch

def norm_pdf(mu, v, x):
    pdf = torch.tensor(1.) / torch.sqrt(2. * 3.141592 * v) * torch.exp(-torch.pow(x - mu, 2) / (2. * v))

    return pdf

def loglike_GMM(y, mu, var, w):  # var[T,n_component]
    # y=obs[T,]
    len_time, n_compoent = mu.shape
    mix_pdf = torch.zeros((n_compoent, len_time), requires_grad=False, dtype=torch.float)

    for i in range(n_compoent):
        mix_pdf_i = w[:, i] * norm_pdf(mu[:, i], var[:, i], y)

        mix_pdf[i, :] = mix_pdf_i

    mix_pdf_sum = torch.sum(mix_pdf, dim=0)

    mix_log = torch.log(mix_pdf_sum)

    log_sum = torch.sum(mix_log)

    return -log_sum

def normalize_w(w):
    if type(w) is np.ndarray:
        w=torch.tensor(w)

    w = torch.sigmoid(w)  # [len_time,n_component]
    w_n_sum = torch.sum(w, dim=1)  # [len_time,]

    w = torch.transpose(w, 0, 1)  # [n_component, len_time]

    w = w / w_n_sum  # normlize the weight
    w = torch.transpose(w, 1, 0)

    return w
def normalize_w_softmax(w):
    if type(w) is np.ndarray:
        w=torch.tensor(w)   #[len_time,n_component]
    m=torch.nn.Softmax(dim=1)  #[len_time,n_component]
    w=m(w)

    return w

def NLLloss(y, mu, var, w, index, index_auto):
    # y=observation

    """ Negative log-likelihood loss function. """
    N = y.shape[0]
    index=1

    if index == 1:
        # y,mean,var,w[T,n_componment]
        var = F.softplus(var) + 1e-6
        w=normalize_w_softmax(w)

        loss1 = loglike_GMM(y, mu, var, w)
        loss1 = loss1 / N

    elif index == 2:
        pi = np.pi
        var = torch.exp(var)
        loss1 = (torch.log(var) + ((y - mu).pow(2)) / var).sum() / 2
        loss1 = loss1 / N
    elif index == 0:
        mse = nn.MSELoss()
        loss1 = torch.sqrt(mse(mu, y))/N

    lm = 0.
    index_auto = 0

    if index_auto == 1:
        loss2 = torch.pow(auto_all_r(mu, y, 2), 2)

        loss = lm * loss2 + loss1
    else:
        loss = loss1
        loss2 = loss1
    return loss, loss1, loss2

def train_model(lstm, optimizer, criterion_index, loader_training, testX, testY, n_epochs, scaling,ix):
    eps = 0.01 * scaling
    loss_training = []
    loss_testing = []
    loss_1 = []
    loss_2 = []
    loss_testing_mini = 1e8
    epoch_best=0
    
    for epoch in range(n_epochs):
        loss_epoch = 0.0
        loss_1_epoch, loss_2_epoch = 0.0, 0.0

        lstm.train()
        for step, (batch_x, batch_y) in enumerate(loader_training):  # for each training step

            batch_x = batch_x.clone().detach().requires_grad_(True)
            optimizer.zero_grad()
            mu, v, w = lstm(batch_x.float())
            loss, loss1, loss2 = NLLloss(batch_y, mu, v, w, criterion_index, index_auto)

            loss_epoch += loss.item()
            loss_1_epoch += loss1.item()
            loss_2_epoch += loss2.item()

            loss.backward()
            optimizer.step()

        loss_training.append(loss_epoch)
        loss_1.append(loss_1_epoch)
        loss_2.append(loss_2_epoch)

        if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss_epoch))

            lstm.eval()
            mu_test, v_test, w_test = lstm(testX.float())
            loss_testing_i, _, _ = NLLloss(testY, mu_test, v_test, w_test, criterion_index, index_auto)
            loss_testing.append(loss_testing_i)

            ##########################
            if loss_testing_i<loss_testing_mini:
                loss_testing_mini=loss_testing_i
                epoch_best=epoch
                print('best epoch: ',epoch_best)
                torch.save({
                    'epoch_best': epoch_best,
                    'model_state_dict': lstm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_training': loss_training,
                    'loss_testing':loss_testing,

                }, path + '/para_save/checkpoint' + str(ix) + '.pkl')
                        
            if abs(epoch_best-epoch)>50:
                break

def evaludate_model(n_workers, trainX, testX):
    mu_train = []
    v_train = []
    w_train = []
    mu_test = []
    v_test = []
    w_test = []

    for i in range(n_workers):
        net = LSTM(input_size, output_size, hidden_size, num_layers, dropout)
        checkpoint = torch.load(path + '/para_save/checkpoint' + str(i) + '.pkl')
        state_dict = checkpoint['model_state_dict']
        
        net.load_state_dict(state_dict)
        net.eval()
        
        # training
        mu_train_i, v_train_i, w_train_i = net(trainX.float())  # [n_component, T]

        mu_train_i = mu_train_i.detach().numpy()
        v_train_i = v_train_i.detach().numpy()
        w_train_i = w_train_i.detach().numpy()

        mu_train.append(mu_train_i)
        v_train.append(v_train_i)
        w_train.append(w_train_i)

        # testing
        mu_test_i, v_test_i, w_test_i = net(testX.float())

        mu_test_i = mu_test_i.detach().numpy()
        v_test_i = v_test_i.detach().numpy()
        w_test_i = w_test_i.detach().numpy()

        mu_test.append(mu_test_i)
        v_test.append(v_test_i)
        w_test.append(w_test_i)

    # [n_workers,n_component, T]
    mu_train = np.array(mu_train)
    mu_test = np.array(mu_test)

    v_train = np.array(v_train)
    v_test = np.array(v_test)

    w_train = np.array(w_train)
    w_test = np.array(w_test)

    return mu_train, v_train, w_train, mu_test, v_test, w_test

def softplus(x):
    y = np.log(1. + np.exp(x))
    return y

def sigmoid(x):
    y = 1. / (1 + np.exp(-x))
    return y

def ensemble_model(criterion_index, mu_train, v_train, mu_test, v_test):
    if criterion_index == 0:
        # predict variance then F.softplus() is needed
        v_train = softplus(v_train) + 1e-6

        v_test = softplus(v_test) + 1e-6
    else:
        # s=logvar
        v_train = np.exp(v_train) + 1e-6
        v_test = np.exp(v_test) + 1e-6

    # mean
    en_mu_train = np.mean(mu_train, axis=0)
    en_mu_test = np.mean(mu_test, axis=0)

    # variance
    en_v_train = np.mean(mu_train ** 2 + v_train, axis=0) - en_mu_train ** 2
    # en_v_train=np.mean(np.power(mu_train, 2) + v_train, axis=0) - np.power(en_mu_train, 2)
    en_v_test = np.mean(mu_test ** 2 + v_test, axis=0) - en_mu_test ** 2

    en_sigma_train = np.sqrt(en_v_train)
    en_sigma_test = np.sqrt(en_v_test)

    len_time_train = np.size(en_mu_train)
    len_time_test = np.size(en_mu_test)
    pred_train = []
    pred_test = []

    for i in range(len_time_train):
        sample = np.random.normal(loc=en_mu_train[i], scale=en_sigma_train[i], size=1000)
        pred_train.append(sample)  # [time_len,size]

    for i in range(len_time_test):
        sample = np.random.normal(loc=en_mu_test[i], scale=en_sigma_test[i], size=1000)
        pred_test.append(sample)

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)
    return pred_train, pred_test, en_mu_train, en_mu_test, en_v_train, en_v_test

def i_mixture_normal(w_t, v_t, mu_t):
    n_member = np.size(mu_t)
    
    w_t_size=w_t.size
    if w_t_size == 1:
        # equivalent weights for all members
        w_t = 1. / n_member * np.ones_like(mu_t,dtype=float)
    # find boudaries from mixutre distribution with 3 sigma distance
    sig_t = np.sqrt(v_t)
    x1 = mu_t - 3 * sig_t
    x2 = mu_t + 3 * sig_t
    xmin = np.min(x1)
    xmax = np.max(x2)
    x = np.linspace(xmin, xmax, num=500)

    # generate mixture of density
    mean_t_pdf = np.zeros(500, )
    for i in range(n_member):
        mean_t_pdf = mean_t_pdf + w_t[i] * st.norm.pdf(x, mu_t[i], sig_t[i])

    p = mean_t_pdf / np.sum(mean_t_pdf)
    xsample = np.random.choice(a=x, size=500, replace=True, p=p)

    return xsample

def mix_dist(mu,v,w):  # var,y is [member,T]

    n_member, len_time = mu.shape
    fpred = []
    w_size=w.size
    for i in range(len_time):
        if w_size == 1:
            fpred_i = i_mixture_normal(w, v[:, i], mu[:, i])
        else:
            fpred_i = i_mixture_normal(w[:, i], v[:, i], mu[:, i])
        fpred.append(fpred_i)
    fpred = np.array(fpred)             #[len_time, n_sample=500]
    return fpred

###

def mix_worker(mu,v,w):

    n_workers, len_time, n_component = mu.shape
    n_all_component=n_workers*n_component

    mu_arr=np.zeros((n_all_component, len_time))
    v_arr=np.zeros((n_all_component, len_time))
    w_arr=np.zeros((n_all_component, len_time))

    index=0
    for i in range(n_component):
        for j in range(n_workers):
            mu_arr[index, :] = mu[j,:, i]
            v_arr[index, :] =v[j,:, i]
            w_arr[index, :] = w[j,:, i]
            index = index+1

    en_pred=mix_dist(mu_arr,v_arr,w_arr)              #[len_time, n_sample=500]

    en_mu=np.mean(en_pred,axis=1)
    en_v=np.var(en_pred,axis=1)

    return en_pred,en_mu,en_v
#get each worker(cpu)'s mean and variance of prediction

def mu_v_worker(mu,v,w):
    # mu,v,w[n_worker,T,n_component]
    n_workers, len_time, n_component = mu.shape

    mu_worker=np.sum(mu*w, axis=2) #[n_workers, T]
    v_worker=[]

    for i in range(n_workers):
        v_worker_i= np.sum(v[i,:,:]+np.power(mu[i,:,:],2),axis=1) - np.power(mu_worker[i,:],2)
        v_worker.append(v_worker_i)

    v_worker=np.array(v_worker)

    #prediction
    mu_i=np.zeros((n_component,len_time))
    v_i = np.zeros((n_component, len_time))
    w_i = np.zeros((n_component, len_time))
    pred_worker = []

    for i in range(n_workers):
        for j in range(n_component):
            mu_i[j,:]=mu[i,:,j]
            v_i[j,:]=v[i,:,j]
            w_i[j,:]=w[i,:,j]
        pred_worker_i=mix_dist(mu_i,v_i,w_i)
        pred_worker.append(pred_worker_i)

    pred_worker=np.array(pred_worker)       #[n_worker, len_time, n_sample]

    return pred_worker, mu_worker,v_worker

def normalize_w_softmax_output(w):
    if type(w) is np.ndarray:
        w=torch.tensor(w)   #[n_worker, len_time,n_component]
    m=torch.nn.Softmax(dim=2)  #[n_worker, len_time,n_component]
    w=m(w).detach().numpy()
    
    return w

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def lstm_para(input_size,hidden_size,output_size):
    l1=torch.nn.Linear(input_size,hidden_size,)
    l2=torch.nn.Linear(hidden_size,output_size)
    num_para_l1=count_parameters(l1)
    num_para_l2=count_parameters(l2)
    num_para=num_para_l1+num_para_l2

    return num_para

def ensemble_model_GMM(criterion_index, mu_train, v_train, w_train, mu_test, v_test, w_test):
    # mu,v,w[n_worker,T,n_component]
    criterion_index=0

    if criterion_index == 0:
        # predict variance then F.softplus() is needed
        v_train = softplus(v_train) + 1e-6
        v_test = softplus(v_test) + 1e-6

        w_train = normalize_w_softmax_output(w_train)
        w_test = normalize_w_softmax_output(w_test)

        print('soft_output')
    else:

        v_train = np.exp(v_train) + 1e-6
        v_test = np.exp(v_test) + 1e-6

    ###########
    pred_train, en_mu_train, en_v_train = mix_worker(mu_train, v_train, w_train)
    pred_test, en_mu_test, en_v_test = mix_worker(mu_test, v_test, w_test)


    return pred_train, pred_test, en_mu_train, en_mu_test, en_v_train, en_v_test



def my_worker(ix, criterion_index, loader_training, testX, testY, scaling):
    torch.set_num_threads(1)
    print("start training for worker ", ix)
    net = LSTM(input_size, output_size, hidden_size, num_layers, dropout)
    optimizer = optim.Adam(net.parameters(), lr=learningr)
    train_model(net, optimizer, criterion_index, loader_training,
                testX,
                testY,
                n_epochs,
                scaling,
                ix)
    
    print("Finishing training for worker ", ix)

def python_parallel(n_jobs, n_worker, criterion_index, loader_training, testX, testY, scaling):
    out = Parallel(n_jobs=n_jobs)(
        delayed(my_worker)(i, criterion_index, loader_training, testX, testY, scaling) for i in range(n_worker))


def random_sample_vec(N,num_samples):
    xx=np.linspace(0,N,N,endpoint=False)
    # print('xx',xx)
    x=np.random.choice(N,size=num_samples,replace=False,)

    x=np.sort(x)

    x1=[]
    x2=[]
    ss=0
    max_x=np.max(x)

    for i in range(N):

        if i<=max_x:

            if x[ss]==i:

                x1_i=xx[i]

                ss=ss+1
                x1.append(x1_i)
            else:
                x2_i=xx[i]
                # print('x2',x2_i)
                x2.append(x2_i)
        else:
            x2_i=xx[i]
            x2.append(x2_i)

    x1=np.array(x1)
    x2=np.array(x2)

    x1=x1.astype(int)
    x2=x2.astype(int)

    return x1,x2

def random_sample(x,y,p):
    N=x.shape[0]
    size=x.size()[0]

    N_1=int(np.round(N*(1-p),0))

    print(str(p*100)+'% of  '+str(N) +'  data are trained:'+str(N_1)  )

    x_vec1,x_vec2=random_sample_vec(N,N_1)

    if size==N:
        x1=x[x_vec1]
        x2=x[x_vec2]
    else:
        x1=x[x_vec1,:]
        x2=x[x_vec2,:]
    y1=y[x_vec1]
    y2=y[x_vec2]

    return x1,y1,x2,y2



def plot_area(pred,x):

    m,n=pred.shape #[len,500]

    for i in range(m):
        xi=np.linspace(x[i],x[i],n)
        plt.plot(xi, pred[i,:], marker='s', 
                     linewidth=1, linestyle='None', 
                     color='b', 
                     alpha=0.05, markeredgewidth=0.0)
    
def read_npz(filename):

    data=np.load(filename)

    x1=data['x1']
    x2=data['x2']

    y1=data['y1']
    y2=data['y2']

    x=np.concatenate((x1,x2))
    y=np.concatenate((y1,y2))

    return x1,y1,x2,y2,x,y

if __name__ == '__main__':
    # Define the hyperparamters
    n_compoent = 2  
    input_size = 1
    hidden_size = 100
    num_layers = 1
    output_size = n_compoent * 3
    batch_size = 1
    dropout = 0.1
    n_epochs = 1000
    learningr = 0.001
    index_auto = 1
    ##########################################
    start = timeit.default_timer()
    norm_index = 0  # whether use normalization or not
    criterion_index = 1  # default
    residual_index = 0  # residual model or not
    
    n_jobs, n_workers = 1, 1
    ####read data
    percent=0.3    

    filename = 'data.npz'
    
    cali_sim_flow, cali_obs_flow, vali_sim_flow, vali_obs_flow, datax, datay = read_npz(filename,

                                                                                           )  # datax=simulated, datay=ob
                       
    trainx = cali_sim_flow
    trainy = cali_obs_flow

    testx = vali_sim_flow
    testy = vali_obs_flow
    train_size=np.size(trainx)
    test_size=np.size(testx)

    #############
    # Normal
    if norm_index == 1:

        trainx_trans, trainy_trans, testx_trans, testy_trans, x_mean, x_std, y_mean, y_std = normalization(trainx,
                                                                                                           trainy,
                                                                                                           testx,
                                                                                                           testy,
                                                                                                           )
        np.savez(path+'/result/data_trans.npz',trainx_trans,
        trainy_trans,
        testx_trans,
        testy_trans,
        x_mean,
        x_std,
        y_mean,
        y_std)

    else:
        x_mean, x_std, y_mean, y_std = 100., 1000., 1000., 1000.
        trainx_trans, trainy_trans, testx_trans, testy_trans = trainx, trainy, testx, testy
    ###############

    trainX, trainY, testX, testY = data_to_tensor(trainx_trans, trainy_trans, testx_trans, testy_trans)
    max, min = torch.max(trainX), torch.min(trainX)

    scaling = (max - min)

    trainX1,trainY1,trainX2,trainY2=random_sample(trainX,trainY,percent)   

    loader_training, loader_testing = data_load(trainX1, trainY1, testX, testY, train_size, test_size)

    python_parallel(n_jobs, n_workers, criterion_index, loader_training, trainX2, trainY2, scaling)

    mu_train, v_train, w_train, mu_test, v_test, w_test = evaludate_model(n_workers, trainX, testX)


    pred_train, pred_test, en_mu_train, en_mu_test, en_v_train, en_v_test = ensemble_model_GMM(criterion_index,
                                                                                               mu_train,
                                                                                               v_train,
                                                                                               w_train,
                                                                                               mu_test,
                                                                                               v_test,
                                                                                               w_test)
    ###############

    for i in range(n_workers):
        checkpoint = torch.load(path + '/para_save/checkpoint' + str(i) + '.pkl')
        loss_training_i = checkpoint['loss_training']

        loss_training_i = torch.FloatTensor(loss_training_i)

        loss_testing_i = checkpoint['loss_testing']

        loss_testing_i = torch.FloatTensor(loss_testing_i)

        plt.plot(loss_training_i, color='black', label='Traing ' + str(i), linewidth=2)
        plt.plot(loss_testing_i, color='r', label='Testing' + str(i))


    plt.legend()
    plt.savefig(path + '/result/train__iteration.png')
    plt.show()

    ############################

    pred = result_reverse_Normalization(pred_train, y_std, y_mean, trainx, norm_index, residual_index)

    #####
    # plot
    X = np.linspace(1, train_size, num=train_size)
    plot_area(pred,cali_sim_flow)
    plt.scatter(cali_sim_flow, cali_obs_flow,color='r',alpha=0.2)
    
    plt.savefig(path + '/result/train_pred.png')
    plt.show()

    ##############################################
    end = timeit.default_timer()
    print(str(end - start))
