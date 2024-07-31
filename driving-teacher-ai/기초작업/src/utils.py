import pandas as pd
import requests

def 구분데이터붙이기(data, time_col:str):
    data = data.sort_values(time_col).reset_index(drop = True)
    data.loc[:, ['연월일', '연', '월', '요일']] = [[x.strftime('%Y%m%d'), x.strftime('%Y'), x.strftime('%m'), str(x.weekday())] for x in data[time_col]]
    
    indexes = data.loc[:, ['연']].drop_duplicates()
    연월일 = data.loc[:, ['연월일', '연', '월']].drop_duplicates()
    공휴일_url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
    공휴일_params ={'serviceKey' : 'bGmKwFeZXvOCMt0Sng8CdDmuYV7Dk3at3SPr/AvRuBmSYANlSBNGq2JsXWAGufkONpfh4TNRc3+RHhriEeje/g==', 
             'solYear' : '2022', 'numOfRows' : '30', '_type': 'json' }

    직전연월 = (pd.to_datetime(data.loc[0, time_col].strftime('%Y-%m')) - pd.Timedelta(value = 1, unit = 'day')).strftime('%Y%m')

    공휴일s = []
    for i in range(len(indexes)):
        indexes.index[i]
        공휴일_params['solYear'] = indexes.iloc[i,0]

        공휴일_response = requests.get(공휴일_url, params = 공휴일_params)
        공휴일s += 공휴일_response.json()['response']['body']['items']['item']

    data['공휴일'] = False
    for i in range(len(공휴일s)):
        data.loc[data['연월일'] == str(공휴일s[i]['locdate']), '공휴일'] = True
    시간대s = []
    for i in range(len(data)):
        if data.loc[i, time_col].strftime('%H') in ['05', '06', '07', '08', '09']:
            시간대s.append('조조')
        elif data.loc[i, time_col].strftime('%H') in ['10', '11', '12', '13', '14', '15', '16']:
            시간대s.append('점심')
        elif data.loc[i, time_col].strftime('%H') in ['17', '18', '19', '20', '21']:
            시간대s.append('저녁')
        elif data.loc[i, time_col].strftime('%H') in ['22', '23', '24', '01', '02', '03', '04']:
            시간대s.append('심야')
    data['시간대'] = 시간대s
    return data