import os
import sys
working_dir = os.getcwd()

sys.path.append(working_dir)

from daily_push.lib.make_data import 시뮬레이션_고객수요
from model_train_deploy.lib.db_io import Engine
import pandas as pd
import numpy as np
import geopandas as gpd
import pickle
from fitter import Fitter
from datetime import datetime

import os
import requests

db_info = {
    'ip': '163.152.172.163',
    'port':'5432',
    'username': 'tgsociety',
    'password': 'tgsociety',
    'dbname': 'postgres',
    'user_name': 'tgsociety',
    'host': '163.152.172.163',
    'db_name': 'postgres'
}

os.chdir('/opt/airflow/dags')
current_dir = os.path.dirname(os.path.abspath(__file__))
def make_push_data(execution_date, 시군구_ids = ['11220', '11230', '11240']):
    # load
    db_handler = Engine(db_info)
    working_dir = os.getcwd()
    print(f"Current working directory: {working_dir}")
    
    query = f"""
    SELECT 시군구_id, 집계구_id, 수요인구, 중심
    FROM datamart.관리_info_집계구_뷰
    WHERE 시군구_id IN ({', '.join(f"'{id}'" for id in 시군구_ids)})
    """
    # 쿼리 실행 및 GeoDataFrame으로 변환
    강남3구_집계구_경계_중심 = gpd.read_postgis(query, db_handler.engine, geom_col='중심')
    
    
    접속_도착_텀 = pd.read_excel(os.path.join('.', 'daily_push', 'raw_data', 'Weekly_Training_Data3.xlsx'))
    접속_도착_텀['예약날짜'] = pd.to_datetime(접속_도착_텀['예약날짜'])
    접속_도착_텀['학원도착날짜'] = pd.to_datetime(접속_도착_텀['학원도착날짜'])
    접속_도착_텀['term'] = 접속_도착_텀['학원도착날짜'] - 접속_도착_텀['예약날짜']

    url = 'https://getallorders-xupmv5q2rq-du.a.run.app'
    response = requests.get(url)
    data = response.json()
    
    samples = []
    for i in range(len(data)):
        if data[i]['appointedAt'] is not None:
            arrived_time = pd.to_datetime(data[i]['selectedTime']['firstVisit']['date'])
            if data[i]['arrivalAt'] is not None:
                arrived_time = max(pd.to_datetime(data[i]['arrivalAt']),  pd.to_datetime(data[i]['selectedTime']['firstVisit']['date']))
            tmp_term = (arrived_time - pd.to_datetime(data[i]['appointedAt'])).total_seconds()
            if tmp_term <0:
                tmp_term = 0
            samples.append(tmp_term)
        
    
    # data_term = 접속_도착_텀.sort_values('term').reset_index(drop=True)
    
    # samples = []
    # for i, tick in enumerate(data_term['term']):
    #     print(i, tick, tick.total_seconds())
    #     if i >= 51:
    #         samples.append(tick.total_seconds())
    
    
    핏터 = Fitter(samples)
    핏터.fit()
    
    print(execution_date)
    execution_date = pd.to_datetime(execution_date)
    if execution_date.time() == pd.Timestamp('00:00:00').time():
        execution_date += pd.Timedelta(value = 1, unit = 'second')
    simul = 시뮬레이션_고객수요(db_info = db_info, 시작일 = execution_date, 위치_데이터 = 강남3구_집계구_경계_중심, 총일수 = 1,
                      term_samples = samples, term_generate_method = 'invgamma')
    simul.데이터생성시뮬실행()
