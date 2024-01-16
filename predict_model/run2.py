import pandas as pd
from datetime import timedelta
from utils.tools import dotdict
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np





if __name__ == '__main__':

    train = True

    data = pd.read_excel("光伏2019.xlsx", engine='openpyxl')
    data.rename(columns={'时间': 'date'}, inplace=True)
    data.drop(['温度(°)'], axis=1, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    split_point = data['date'].max() - timedelta(days=int((data['date'].max() - data['date'].min()).days * 0.2))
    train_data = data[data['date'] <= split_point]  # 28128 8
    test_data = data[data['date'] > split_point]



    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = dotdict()
    args.task_name = 'long_term_forecast'
    # args.is_training = 1  # status
    args.model_id = '发电功率_96_96'       #############################
    args.model = 'Informer'  # model name, options: [Autoformer, Transformer, TimesNet]
    # data loader
    args.features = 'MS'  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = '实际发电功率(mw)'  # target feature in S or MS task(预测目标列)
    args.freq = 't'  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    # args.checkpoints = './checkpoints'  # location of model checkpoints
    args.checkpoints = './test_save'
    # forecasting task
    args.seq_len=96  # input sequence length
    args.label_len=48  # start token length
    args.pred_len=96  # prediction sequence length
    args.seasonal_patterns='Monthly'  # subset for M4
    args.inverse=True  # inverse output data
    # inputation task
    args.mask_rate=0.25  # mask ratio

    # anomaly detection task
    args.anomaly_ratio=0.25  # prior anomaly ratio (%)

    # model define
    args.top_k=5  # for TimesBlock
    args.num_kernels=6  # for Inception
    args.enc_in=7  # encoder input size
    args.dec_in=7  # decoder input size
    args.c_out=7  # output size
    args.d_model=256  # dimension of model
    args.n_heads=8  # num of heads
    args.e_layers=3  # num of encoder layers
    args.d_layers=1  # num of decoder layers
    args.d_ff=2048  # dimension of fcn
    args.moving_avg=25  # window size of moving average
    args.factor=25  # attn factor
    args.distil=True  # whether to use distilling in encoder, using this argument means not using distilling
    args.dropout=0.02  # dropout
    args.embed='timeF'  # time features encoding, options:[timeF, fixed, learned]
    args.activation='gelu'  # activation
    args.output_attention=False  # whether to output attention in ecoder

    # optimization
    args.num_workers=0  # data loader num workers
    args.train_epochs=1  # train epochs
    args.batch_size=64  # batch size of train input data
    args.patience=4  # early stopping patience
    args.learning_rate=0.00005  # optimizer learning rate
    args.des='Exp'  # exp description
    args.loss='MAE'  # loss function
    args.lradj='type1'  # adjust learning rate
    args.use_amp=False  # use automatic mixed precision training

    # GPU
    args.use_gpu=True  # use gpu
    args.gpu=0  # gpu
    args.use_multi_gpu=False  # use multiple gpus
    args.devices='0,1,2,3'  # device ids of multile gpus

    # de-stationary projector params
    args.p_hidden_dims=[128, 128]  # hidden layer dimensions of projector (List)
    args.p_hidden_layers=2  # number of hidden layers in projector

    # import json
    #
    # json_data = json.dumps(args)
    #
    # with open("../default_args.json", "w", encoding="utf-8") as file:
    #     file.write(json_data)



    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)



    Exp = Exp_Long_Term_Forecast



    # setting record of experiments
    exp = Exp(args)  # set experiments
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des)

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.train(setting, train_data)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting, test_data, test=1)

    print(train_data[:96])


    pred = exp.predict(setting, train_data[:96])
    print(pred)
    # import matplotlib.pyplot as plt
    # plt.plot([i for i in range(len(pred.flatten()))], pred.flatten())
    # plt.plot([1,2,3],[1,2,3])
    # plt.show()

    torch.cuda.empty_cache()
