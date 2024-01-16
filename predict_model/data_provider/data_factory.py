# from data_provider.data_loader import  Dataset_Custom
from predict_model.data_provider.my_data_loader import Dataset_Custom
from predict_model.data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    # 'custom': DatasetTrain
}


def data_provider(args, flag, raw_data):
    # Data = data_dict[args.data]
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    data_set = Data(
        original_data=raw_data,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
