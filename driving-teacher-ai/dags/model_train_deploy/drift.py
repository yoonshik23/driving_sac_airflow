import os
import sys
working_dir = os.getcwd()
sys.path.append(working_dir)

from model_train_deploy.lib.db_io import Engine

from model_train_deploy.lib.simulator_02 import Cluster
from model_train_deploy.lib.utils import 구분데이터붙이기, 군집나누기
import pandas as pd
import pickle
from scipy.stats import ks_2samp
from sqlalchemy import text

db_info = {'host': '163.152.172.163',
          'port': '5432',
          'db_name': 'postgres',
          'user_name': 'tgsociety',
          'password': 'tgsociety'}



def drift(execution_date, 시군구_ids = ['11220', '11230', '11240']):
    execution_date = pd.to_datetime(execution_date)
    db_handler = Engine(db_info)

    query = '''
        SELECT *
        FROM datamart.history_학습id
        ORDER BY 등록일시 DESC
        LIMIT 1
    '''
    train_history = pd.read_sql(query, db_handler.engine)

    new_query = '''
    SELECT *
    FROM datamart.고객수요_집계구_뷰
    WHERE 예약희망일시 >= %(st_dt)s
        and 예약희망일시 <= %(ed_dt)s
        AND 시군구_id IN %(시군구_ids)s
    '''
    # 새로운 데이터 불러오기
    ref_data = pd.read_sql(new_query, db_handler.engine, params={'st_dt': train_history.loc[0, 'train_data_start_dt'], 'ed_dt': train_history.loc[0, 'train_data_end_dt'], '시군구_ids': tuple(시군구_ids)})
    
    cur_data = pd.read_sql(new_query, db_handler.engine, params={'st_dt': train_history.loc[0, 'train_data_start_dt'], 'ed_dt': execution_date, '시군구_ids': tuple(시군구_ids)})

    query = '''
    SELECT demand_cluster_pickle
    FROM datamart.storage_demand_cluster
    WHERE train_id = %(train_id)s
    '''
    
    # 쿼리 실행 및 데이터 불러오기
    tmp_df = pd.read_sql(query, db_handler.engine, params={'train_id': int(train_history.loc[0, 'train_id'])})

    클러스터링결과 = pickle.loads(tmp_df.loc[0, 'demand_cluster_pickle'])
    # 클러스터링결과['cluster'] = self.cluster
    # 클러스터링결과['hierarchy_sample_data'] = self.hierarchy_sample_data

    ref_data = 구분데이터붙이기(ref_data.copy(), '예약희망일시')
    cur_data = 구분데이터붙이기(cur_data.copy(), '예약희망일시')
    
    군집_ref_data = 군집나누기(ref_data.copy(), 클러스터링결과['cluster'].군집s)
    군집_cur_data = 군집나누기(cur_data.copy(), 클러스터링결과['cluster'].군집s)
    
    p_values = []
    for key in list(군집_ref_data):
        if (len(군집_ref_data[key])==0) & (len(군집_cur_data[key])==0):
            p_values.append(1)
        elif (len(군집_ref_data[key])==0) | (len(군집_cur_data[key])==0):
            p_values.append(0)
        else:
            p_values.append(ks_2samp(군집_ref_data[key], 군집_cur_data[key])[1].item())
    
            
    data_start_dt = train_history.loc[0, 'train_data_start_dt']
    data_end_dt = cur_data.sort_values(by = '예약희망일시', ascending = False).reset_index(drop=True).loc[0, '예약희망일시']
    
    insert_query = text('''INSERT INTO datamart."history_드리프트id" ("drift_data_start_dt", "drift_data_end_dt", "logical_dt") VALUES (:st, :ed, :logi) RETURNING "drift_id";''')
    
    with db_handler.engine.connect() as connection:
        trans = connection.begin()
        result = connection.execute(insert_query, {'st': data_start_dt, 'ed': data_end_dt, 'logi': execution_date})
        
        # 생성된 id 값을 가져옴
        drift_id = result.scalar()
        trans.commit()
    
    tmp_df = pd.DataFrame(columns = ['drift_id', 'train_id', 'cluster_num', 'drift_method', 'drift_value', 'drift_yn'])
    cluster_nums = []
    drift_values = []
    drift_yns = []
    for i, p in enumerate(p_values):
        cluster_nums.append(i)
        drift_values.append(p)
        drift_yns.append(p <= 0.9)
    tmp_df = pd.DataFrame({
        'drift_id': [drift_id]*len(p_values),
        'train_id': [int(train_history.loc[0, 'train_id'])]*len(p_values),
        'cluster_num': cluster_nums,
        'drift_method': ['Kolmogorov_Smirnov Test']*len(p_values),
        'drift_value': drift_values,
        'drift_yn': drift_yns
    })
    tmp_df.to_sql('result_drift', db_handler.engine, schema = 'datamart', index = False, if_exists='append')
    return any(drift_yns)





    