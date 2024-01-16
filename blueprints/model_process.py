from flask import Blueprint, request
from predict_model.utils.tools import dotdict
from predict_model.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import json
from models import ModelInfo, DataSet
from exts import db
import pandas as pd
from datetime import timedelta, datetime
import torch
bp = Blueprint("model", __name__, url_prefix="/model")



@bp.route('/create', methods=['POST'])
def create_model():
    model_args = request.form.get('model_args')
    args=dotdict(json.loads(model_args))
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
    model_location = 'model_saving' + '/' + args.model_id + '/' + args.model + '/' + setting + '.pth'
    args.checkpoints = model_location


    model_info = ModelInfo(source_type=request.form.get('source_type'),
                           source_id=request.form.get('source_id'),
                           model_name=request.form.get('model_name'),
                           model_location=model_location,
                           model_args=json.dumps(args))
    db.session.add_all([model_info])
    db.session.commit()


    return "Create successfully!"


@bp.route('/update/<int:post_id>', methods=['PUT'])
def update_model(post_id):
    model_args = request.form.get('model_args')
    args = dotdict(json.loads(model_args))
    model_name = request.form.get('model_name')
    source_type = request.form.get('source_type')
    source_id = request.form.get('source_id')
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

    model_location = 'model_saving' + '/' + args.model_id + '/' + args.model + '/' + setting + '.pth'
    args.checkpoints = model_location

    model_info = ModelInfo.query.get(post_id)
    model_info.source_type = source_type
    model_info.source_id = source_id
    model_info.model_name = model_name
    model_info.model_args = json.dumps(args)
    model_info.model_location = model_location
    db.session.commit()

    return "Update successfully!"


@bp.route('/delete/<int:delete_id>', methods=['DELETE'])
def delete_model(delete_id):

    model_info = ModelInfo.query.get(delete_id)
    db.session.delete(model_info)
    db.session.commit()
    db.session.commit()

    return "Delete successfully!"

@bp.route('/', methods=['GET'])
def select_by_source():
    source_type = request.args.get("source_type")
    source_id = request.args.get("source_id")

    model_infos = ModelInfo.query.filter(ModelInfo.source_type == source_type, ModelInfo.source_id == source_id).all()
    if not model_infos:
        return False

    return [obj.serialize() for obj in model_infos]


@bp.route('/train/<int:id>', methods=['POST'])
def train(id):
    from_date = request.form.get('from_date')
    to_date = request.form.get('to_date')
    model_info = ModelInfo.query.get(id)


    datasets =  DataSet.query.filter(DataSet.date >= from_date).filter(DataSet.date <= to_date).order_by(DataSet.date.asc()).all()
    data = pd.DataFrame([obj.serialize() for obj in datasets])
    data['date'] = pd.to_datetime(data['date'])
    split_point = data['date'].max() - timedelta(days=int((data['date'].max() - data['date'].min()).days * 0.2))
    model_args = model_info.model_args
    args = dotdict(json.loads(model_args))
    args.enc_in = data.shape[1]-1  # encoder input size
    args.dec_in = data.shape[1]-1  # decoder input size
    args.c_out = data.shape[1]-1
    args.target = data.columns[-1]


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    model_info.model_args = json.dumps(args)


    model_info.status = 1
    db.session.commit()

    Exp = Exp_Long_Term_Forecast
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
    exp.train(setting, data)

    model_info = ModelInfo.query.get(id)
    model_info.status = 2
    db.session.commit()



    return "Training is complete!"

@bp.route('/predict/<int:id>', methods=['POST'])
def predict(id):
    date = request.form.get('date')
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    model_info = ModelInfo.query.get(id)
    model_args = model_info.model_args
    args = dotdict(json.loads(model_args))

    from_date = date - timedelta(minutes=int(args.seq_len*15))

    datasets = DataSet.query.filter(DataSet.date >= from_date).filter(DataSet.date < date).order_by(DataSet.date.asc()).all()
    data = pd.DataFrame([obj.serialize() for obj in datasets])
    data['date'] = pd.to_datetime(data['date'])

    Exp = Exp_Long_Term_Forecast
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
    preds = exp.predict(setting, data)
    preds = preds.flatten()

    result_dict = {}

    for pred in preds:
        result_dict[date.strftime("%Y-%m-%d %H:%M:%S")] = str(pred)
        date = date + timedelta(minutes=15)


    return result_dict

