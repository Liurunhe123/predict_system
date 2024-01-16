from sqlalchemy import create_engine, text

HOSTNAME = 'localhost'
DATABASE = 'predict_system'
PORT = 3306
USERNAME = 'root'
PASSWORD = '123456'
DB_URL = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
engine = create_engine(DB_URL)


# with engine.connect() as conn:
#     # 执行原生SQL语句
#     results = conn.execute(text("select * from data_set"))
#     # 遍历所有数据
#     for result in results:
#         print(result)
