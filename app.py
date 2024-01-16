from flask import Flask
from exts import db
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from blueprints.model_process import bp as  model_bp
import config
app = Flask(__name__)



app.config.from_object(config)
db.init_app(app)

migrate = Migrate(app, db)

app.register_blueprint(model_bp)










if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:123456@localhost:3306/predict_system'
    # 指定数据库文件
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

    db = SQLAlchemy(app)