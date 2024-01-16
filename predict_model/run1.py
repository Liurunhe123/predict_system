import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
from datetime import timedelta
from utils.tools import dotdict

train = True

data = pd.read_excel("光伏2019.xlsx",  engine='openpyxl')
data.rename(columns={'时间':'date'}, inplace=True)
data['date'] = pd.to_datetime(data['date'])
split_point = data['date'].max() - timedelta(days=int((data['date'].max()-data['date'].min()).days*0.2))
train_data = data[data['date'] <= split_point]
test_data = data[data['date'] > split_point]




args = dotdict()
args.task_name='long_term_forecasting'
args.model = 'Informer'
args.data = 'Electronic'
args.target = '实际发电功率(mw)' # target feature in S or MS task(预测目标列)
args.freq = 't' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.checkpoints = './checkpoints' # location of model checkpoints
args.features = 'MS'  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate

args.seq_len = 96 # input sequence length of Informer encoder
args.label_len = 48 # start token length of Informer decoder
args.pred_len = 96 # prediction sequence length
args.gap_len = 0
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

# args.enc_in = 36 # encoder input size
# args.dec_in = 36 # decoder input size
# args.c_out = 1 # output size

args.enc_in = 1  # encoder input size
args.dec_in = 1  # decoder input size
args.c_out = 21  # output size

args.factor = 25  # probsparse attn factor
args.d_model = 256  # dimension of model
args.n_heads = 8  # num of heads
args.e_layers = 3  # num of encoder layers
args.d_layers = 1  # num of decoder layers
args.d_ff = 2048  # dimension of fcn in model
args.dropout = 0.02  # dropout
args.attn = 'prob'  # attention used in encoder, options:[prob, full]
args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu'  # activation
args.distil = True # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in ecoder
args.mix = True
args.inverse = True
args.padding = 0

args.batch_size = 32
args.learning_rate = 0.00005
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False  # whether to use automatic mixed precision training

args.num_workers = 0
args.itr = 1
args.train_epochs = 9
args.patience = 4
args.des = 'Exp'

args.anomaly_ratio=0.25




args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                                                                                                    args.data,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.attn,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.mix, args.des,
                                                                                                    ii)

    # set experiments
    Exp = Exp_Long_Term_Forecast

    # setting record of experiments
    exp = Exp(args)  # set experiments

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting, train_data)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting)
    # torch.cuda.empty_cache()
