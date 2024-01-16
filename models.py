
from exts import db





class ModelInfo(db.Model):

    __tablename__ = 'model_info'

    #  字段
    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment="主键id")
    source_type = db.Column(db.String(50), nullable=False, comment="数据源类型")
    source_id = db.Column(db.Integer, default=0, nullable=False, comment="数据源id")
    model_name = db.Column(db.String(50), nullable=False, comment="模型名称")
    model_location = db.Column(db.Text, nullable=False, comment="模型位置")
    model_args = db.Column(db.Text, nullable=False, comment="模型参数信息")
    status =  db.Column(db.Integer, default=0, nullable=False, comment="模型状态, 0: 未训练, 1: 训练中, 2: 训练完成, 3: 训练出错")


    def serialize(self):
        return {'source_type': self.source_type,
                'source_id': self.source_id,
                'model_name': self.model_name,
                'model_args': self.model_args}
        # return ("{{'source_type':{}, 'source_id':{}, 'model_name':{}, 'model_args':{}}}".
        #         format(self.source_type, self.source_id, self.model_name, self.model_args))



class ModelTrain(db.Model):

    __tablename__ = 'data_train'

    #  字段
    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment="主键id")
    from_date =  db.Column(db.DateTime, nullable=False, comment="起始时间")
    to_date =  db.Column(db.DateTime, nullable=False, comment="起始时间")

    model_id = db.Column(db.Integer, db.ForeignKey("model_info.id"), nullable=False,   comment="模型id")


    model_info = db.relationship('ModelInfo', backref='model_trains')



class DataSet(db.Model):

    __tablename__ = 'data_set'

    #  字段
    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment="主键id")
    date =  db.Column(db.DateTime, nullable=False, comment="时间")
    component_temperature = db.Column(db.Float, nullable=False, comment="组件温度(℃)")
    temperature = db.Column(db.Float, nullable=False, comment="温度(℃)")
    air_pressure = db.Column(db.Float, nullable=False, comment="气压(hPa)")
    humidity = db.Column(db.Float, nullable=False, comment="湿度(%)")
    total_radiation = db.Column(db.Float, nullable=False, comment="总辐射(W/m2)")
    direct_radiation = db.Column(db.Float, nullable=False, comment="直射辐射(W/m2)")
    scattered_radiation = db.Column(db.Float, nullable=False, comment="散射辐射(W/m2)")
    generated_power = db.Column(db.Float, nullable=False, comment="实际发电功率(mw)")



    def serialize(self):
        return {'date': self.date,
                'component_temperature': self.component_temperature,
                'temperature': self.temperature,
                'air_pressure': self.air_pressure,
                'humidity': self.humidity,
                'total_radiation': self.total_radiation,
                'direct_radiation': self.direct_radiation,
                'scattered_radiation': self.scattered_radiation,
                'generated_power': self.generated_power
                }