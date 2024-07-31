import pandas as pd
import requests
from pytimekr import pytimekr

from shapely.geometry import Point
import random
import torch

def 구분데이터붙이기(data, time_col:str):
    data[time_col] = pd.to_datetime(data[time_col])
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
        if 공휴일_response.json()['response']['body']['totalCount'] != 0:
            공휴일s += 공휴일_response.json()['response']['body']['items']['item']
        else:
            공휴일s += [{'locdate': x.strftime('%Y%m%d')} for x in pytimekr.holidays(int(indexes.iloc[i,0]))]

    data['공휴일'] = False
    for i in range(len(공휴일s)):
        data.loc[data['연월일'] == str(공휴일s[i]['locdate']), '공휴일'] = True
    시간대s = []
    for i in range(len(data)):
        시간대s.append(which_시간대(data.loc[i, time_col]))

    data['시간대'] = 시간대s
    return data
    
def which_시간대(시간):
    if 시간.strftime('%H') in ['05', '06', '07', '08', '09']:
        return '조조'
    elif 시간.strftime('%H') in ['10', '11', '12', '13', '14', '15', '16']:
        return '점심'
    elif 시간.strftime('%H') in ['17', '18', '19', '20', '21']:
        return '저녁'
    elif 시간.strftime('%H') in ['22', '23', '00', '01', '02', '03', '04']:
        return '심야'
        
def generate_polygon_random_points(poly):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []

    while len(points) == 0:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if random_point.within(poly):
            points.append(random_point)
    
    return points

def 공휴일출력(year):
    공휴일_url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'
    공휴일_params ={'serviceKey' : 'bGmKwFeZXvOCMt0Sng8CdDmuYV7Dk3at3SPr/AvRuBmSYANlSBNGq2JsXWAGufkONpfh4TNRc3+RHhriEeje/g==', 
             'solYear' : year, 'numOfRows' : '30', '_type': 'json' }

    공휴일s = []


    공휴일_response = requests.get(공휴일_url, params = 공휴일_params)
    if 공휴일_response.json()['response']['body']['totalCount'] != 0:
        공휴일s += 공휴일_response.json()['response']['body']['items']['item']
    else:
        공휴일s += [{'locdate': x.strftime('%Y%m%d')} for x in pytimekr.holidays(int(year))]


    tmp = []
    for i in range(len(공휴일s)):
        tmp.append(str(공휴일s[i]['locdate']))

    return tmp

def 예약가능시간출력(접속일, 당해년공휴일, 며칠뒤까지 = 30):
    '''
    월화수목금토일 0123456
    휴일_예약가능시각 = ['8', '10', '12', '14', '16', '18', '20']
    업무일_예약가능시각 = ['9', '12', '15', '18', '20']


    '''
    예약가능시간 = {'예약일시' : [], '월' : [], '요일' : [], '공휴일' : [], '시간대' : [], '며칠후' : []}

    for value in range(1, 며칠뒤까지+1):
        오늘 = 접속일 + pd.Timedelta(value = value, unit = 'day')
        
        # 토요일/일요일 이거나 공휴일
        if (오늘.strftime('%Y%m%d') in 당해년공휴일) | ((오늘.weekday() == '5') | (오늘.weekday() == '6')):
            for hour in [' 09:00', ' 09:30', ' 10:00', ' 10:30', ' 11:00', ' 11:30', ' 12:00', ' 12:30', ' 13:00', ' 13:30', ' 14:00', ' 14:30', ' 15:00', ' 15:30', ' 16:00', ' 16:30', ' 17:00', ' 17:30', ' 18:00', ' 18:30', ' 19:00', ' 19:30', ' 20:00', ' 20:30', ' 21:00']:
                예약가능시간['예약일시'].append(오늘.strftime('%Y-%m-%d') + hour +':00')
            for _ in range(25):
                예약가능시간['월'].append(오늘.strftime('%m'))
            for _ in range(25):
                예약가능시간['요일'].append(오늘.weekday())
                
            공휴일 = False
            if 오늘.strftime('%Y%m%d') in 당해년공휴일:
                공휴일 = True

            for _ in range(25):
                예약가능시간['공휴일'].append(공휴일)
                
            # 시간대 : '조조', '점심', '저녁', '심야'
            #     (05 ~ 09), (10 ~ 16), (17 ~ 21), (22 ~ 04)
            for _ in range(2):
                예약가능시간['시간대'].append('조조')
            for _ in range(14):
                예약가능시간['시간대'].append('점심')
            for _ in range(9):
                예약가능시간['시간대'].append('저녁')

            for _ in range(25):
                예약가능시간['며칠후'].append(value)
                
        else:
            for hour in [' 09:00', ' 09:30', ' 10:00', ' 10:30', ' 11:00', ' 11:30', ' 12:00', ' 12:30', ' 13:00', ' 13:30', ' 14:00', ' 14:30', ' 15:00', ' 15:30', ' 16:00', ' 16:30', ' 17:00', ' 17:30', ' 18:00', ' 18:30', ' 19:00', ' 19:30', ' 20:00', ' 20:30', ' 21:00']:
                예약가능시간['예약일시'].append(오늘.strftime('%Y-%m-%d') + hour +':00')
            
            for _ in range(25):
                예약가능시간['월'].append(오늘.strftime('%m'))
            for _ in range(25):
                예약가능시간['요일'].append(오늘.weekday())

            for _ in range(25):
                예약가능시간['공휴일'].append(False)

            # 시간대 : '조조', '점심', '저녁', '심야'
            #     (05 ~ 09), (10 ~ 16), (17 ~ 21), (22 ~ 04)
            for _ in range(2):
                예약가능시간['시간대'].append('조조')
            for _ in range(14):
                예약가능시간['시간대'].append('점심')
            for _ in range(9):
                예약가능시간['시간대'].append('저녁')

            for _ in range(25):
                예약가능시간['며칠후'].append(value)

    예약가능시간 = pd.DataFrame(예약가능시간)
    예약가능시간['예약일시'] = pd.to_datetime(예약가능시간['예약일시'])
    예약가능시간['예약일시'] = 예약가능시간['예약일시'].dt.tz_localize('Asia/Seoul')
    return 예약가능시간

def 이웃시간판단(일시, 요일, 공휴일):

    리턴 = []
    if (공휴일) | ((요일 == '5') | (요일 == '6')): # 휴일이라는 뜻
        시 = 일시.strftime('%H')
        if 시 != '20':
            리턴 += str(int(시) + 2)
        if 시 != '08':
            리턴 += format(int(시) - 2, '0>2')
    else: # 업무일이라는 뜻
        if 시 not in ['18', '20']:
            리턴 += str(int(시) + 3)
        elif 시 == '18':
            리턴 += str(int(시) + 2)
        if 시 not in ['09', '20']:
            리턴 += format(int(시) - 3, '0>2')
        elif 시 == '20':
            리턴 += str(int(시) - 2)
    return 리턴
    
def time_to_minutes(time_str):
    '''
    %H_%M 형태의 시간을 minute으로 변환
    '''
    hours, minutes = map(int, time_str.split('_'))
    return hours * 60 + minutes


def state_to_tensor(state, device, max_length = 12):
    '''
    [연속형변수, 가변변수]
    '''
    continuous_vars_tensor = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(device)
    variable_length_tensors = [torch.tensor(lst, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device) for lst in state[1]]
    
    # 모든 tensor의 길이가 0인 경우 처리
    if all(tensor.size(1) == 0 for tensor in variable_length_tensors):
        variable_length_tensors = [torch.zeros(1, 1, 1, dtype=torch.float32).to(device) for _ in variable_length_tensors]
    
    variable_length_tensors = pad_variable_length_tensors(variable_length_tensors, device, max_length)
    return continuous_vars_tensor, variable_length_tensors

# 패딩 함수
def pad_variable_length_tensors(tensors, device, max_length=12):

    padded_tensors = []
    for tensor in tensors:
        padding_size = max_length - tensor.size(1)
        if padding_size > 0:
            padding = torch.zeros((tensor.size(0), padding_size), dtype=torch.float32).to(device)
            padded_tensor = torch.cat([tensor.squeeze(-1), padding], dim=1)
        else:
            padded_tensor = tensor.squeeze(-1)
        padded_tensors.append(padded_tensor)
    padded_tensors = torch.stack(padded_tensors, dim=1)  # (batch_size, 25, max_length)
    return padded_tensors











