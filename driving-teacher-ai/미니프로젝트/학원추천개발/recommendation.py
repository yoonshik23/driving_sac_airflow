'''
위경도 잘못들어오면 거의 무한루프 수준으로 돔.
lat, long 순서 주의
'''

import pandas as pd
import numpy as np
import pickle
이름정의 = {'가격': 'lesson_price', '서비스점수': 'service_rate', '강의점수': 'lecutre_rate', '시설점수': 'facility_rate', '거리': 'distance_m'}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 지구의 반지름 (미터 단위)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c  # 미터 단위 거리
    return distance

def 전처리_pred(lat, long, 학원data, scaler):
    data = 학원data.copy()
    data['user_lat'] = lat
    data['user_long'] = long
    data['distance_m'] = data.apply(lambda row: haversine(row['academy_lat'], row['academy_long'], row['user_lat'], row['user_long']), axis=1)
    X_predict = data.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m', 'name']].reset_index(drop = True)
    # print(X_predict[X_predict['distance_m']<=25000])
    dist = 25000
    tmp_X_predict = X_predict[X_predict['distance_m']<=25000]
    while len(tmp_X_predict) == 0:
        dist += 5000
        tmp_X_predict = X_predict[X_predict['distance_m']<=dist]
        if dist >= 100000:
            tmp_X_predict = X_predict.copy()
            break
    X_predict = tmp_X_predict.copy()
    X_predict.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m']] = scaler.fit_transform(X_predict.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m']])
    return X_predict

def recommend(location:tuple, selected_columns:list):
    '''
    location : (lat, long) 
        위 형태의 length 2짜리 튜플, 각각 float
        
    selected_columns : ['가격', '서비스점수', '강의점수', '시설점수', '거리'] 
        위 형태의 length 1 ~ 5 의 리스트. 무엇을 선택했는지에 대한 정보.
        string 형태 바꾸려면 같은 모듈에 있는 이름정의 dictionary를 수정.
    '''

    select_cols  = [이름정의[x] for x in selected_columns]
    select_cols = sorted(select_cols)
    학원data = pd.read_csv('./models/academy_data.csv', index_col = 0)

    # value_to_int = {value: idx for idx, value in enumerate(학원data['academy_name'].unique())}
    # int_to_value = {idx: value for value, idx in value_to_int.items()}
    # 학원data['academy_name'] = 학원data['academy_name'].map(value_to_int)
    
    file_name = '__'.join(sorted(select_cols))

    with open('./models/'+file_name + '__scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    예측data = 전처리_pred(lat = location[0], long = location[1], 학원data = 학원data, scaler = scaler)
    
    with open('./models/'+file_name + '__rf.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    예측data['prob'] = rf_model.predict_proba(예측data.loc[:, select_cols])[:, 1]
    
    if (len(예측data['prob'].unique()) == 1) & (예측data['prob'].unique()[0] == 0):
        with open('./models/'+file_name + '__logi.pkl', 'rb') as file:
            logi_model = pickle.load(file)
        예측data['prob'] = logi_model.predict_proba(예측data.loc[:, select_cols])[:, 1]

    max_a_row = 예측data[예측data['prob'] == 예측data['prob'].max()]
    b_value = max_a_row['name'].values[0]
    return b_value

