import pandas as pd
import numpy as np
import pickle
이름정의 = {'가격': 'lesson_price', '서비스점수': 'service_rate', '강의점수': 'lecutre_rate', '시설점수': 'facility_rate', '거리': 'distance_m'}

# Constants
INITIAL_DIST = 10000
DIST_INCREMENT = 5000
MAX_DIST = 100000
EARTH_RADIUS = 6371000  # Earth's radius in meters

preference_name_map = {
    'price': 'lesson_price',
    'service': 'service_rate',
    'lecture': 'lecutre_rate',
    'facility': 'facility_rate',
    'distance': 'distance_m'
}

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi, delta_lambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = EARTH_RADIUS * c  # Distance in meters
    return distance

def preprocessing_pred(lat: float, long: float, academy_data: pd.DataFrame, scaler ) -> pd.DataFrame:
    """
    Preprocess the data for prediction.
    """
    data = academy_data.copy()
    data['user_lat'] = lat
    data['user_long'] = long
    data['distance_m'] = data.apply(lambda row: haversine(row['academy_lat'], row['academy_long'], row['user_lat'], row['user_long']), axis=1)
    X_predict = data.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m', 'academy_id']].reset_index(drop = True)
    # print(X_predict[X_predict['distance_m']<=25000])
    dist = INITIAL_DIST
    tmp_X_predict = X_predict[X_predict['distance_m']<=dist]
    while len(tmp_X_predict) == 0:
        dist += DIST_INCREMENT
        if dist >= MAX_DIST:
            tmp_X_predict = X_predict.copy()
            break
        tmp_X_predict = X_predict[X_predict['distance_m']<=dist]

    
    X_predict = tmp_X_predict.copy()
    
    # Explicitly cast the relevant columns to float64
    X_predict.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m']] = X_predict.loc[:, [
        'lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m']].astype('float64')
    
    if scaler is not None:
        X_predict.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m']] = scaler.fit_transform(X_predict.loc[:, ['lesson_price', 'service_rate', 'lecutre_rate', 'facility_rate', 'distance_m']])
    return X_predict

def recommend(location: tuple, selected_columns: list) -> str:
    """
    Recommend an academy based on the user's location and preferences.

    :param location: A tuple (lat, long) representing the user's location.
    :param selected_columns: A list of selected preferences.
    :return: The name of the recommended academy.
    """

    # select_cols  = [이름정의[x] for x in selected_columns]
    select_cols = sorted([preference_name_map[x] for x in selected_columns])
    

    try:
        academy_data = pd.read_csv('./models/academy_data.csv', index_col=0)
        academy_data = academy_data[academy_data['state']==1]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return ""

    if len(select_cols) != 1:
        file_name = '__'.join(sorted(select_cols))
        
        try:
            with open(f'./models/{file_name}__scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return ""
        
        predict_data = preprocessing_pred(lat = location[0], long = location[1], academy_data = academy_data, scaler = scaler)
        
        try:
            with open(f'./models/{file_name}__rf.pkl', 'rb') as file:
                rf_model = pickle.load(file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return ""
        predict_data['prob'] = rf_model.predict_proba(predict_data.loc[:, select_cols])[:, 1]
        
        if (len(predict_data['prob'].unique()) == 1) & (predict_data['prob'].unique()[0] == 0):
            try:
                with open(f'./models/{file_name}__logi.pkl', 'rb') as file:
                    logi_model = pickle.load(file)
                predict_data['prob'] = logi_model.predict_proba(
                    predict_data.loc[:, select_cols])[:, 1]
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return ""
    
        max_a_row = predict_data.loc[predict_data['prob'].idxmax()]
        b_value = max_a_row['academy_id']
        
    elif len(select_cols) == 1:
        predict_data = preprocessing_pred(lat = location[0], long = location[1], academy_data = academy_data, scaler = None)
        # print(predict_data)
        if (select_cols[0] == 'lesson_price') | (select_cols[0] == 'distance_m'):
            m_value = predict_data[select_cols[0]].min()
        else:
            m_value = predict_data[select_cols[0]].max()
        tmp_df = predict_data[predict_data[select_cols[0]] == m_value]

        if len(tmp_df)>1:
            m_value = tmp_df['distance_m'].min()
            tmp_df = tmp_df[tmp_df['distance_m'] == m_value]


        b_value = tmp_df['academy_id'].values[0]
            
    else:
        print('선택컬럼이 없음.')
    return b_value

