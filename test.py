import pandas as pd
import numpy as np

file_path = 'predict_model/光伏2019.xlsx'  # 请将此路径替换为您的Excel文件路径
data = pd.read_excel(file_path,  engine='openpyxl')
print(data.iloc[0][0])