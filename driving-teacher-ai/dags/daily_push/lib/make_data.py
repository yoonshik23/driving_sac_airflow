# sys.path.append('/opt/airflow/dags/daily_push')
import os
import sys
working_dir = os.getcwd()

sys.path.append(working_dir)

import pandas as pd
import numpy as np
from datetime import datetime

from geoalchemy2 import Geometry, WKTElement
from scipy.stats import invgamma

from sqlalchemy import create_engine
import random
from daily_push.lib.utils import 구분데이터붙이기

class 시뮬레이션_고객수요():
    휴일_예약가능시각 = ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    업무일_예약가능시각 = ['9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

    def __init__(self, 
                 db_info,
                 시작일:str, 
                 위치_데이터:pd.DataFrame,
                 총일수:int=365, 
                 휴일_시간별_예약_평균:list=[6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 6],
                 비수기_업무일_시간별_예약_평균:list=[1, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 8, 6], # 50
                 성수기_업무일_시간별_예약_평균:list=[3, 3, 6, 6, 6, 6, 6, 6, 6, 10, 10, 8, 6], # 50
                 term_samples = [], term_generate_method = 'uniform'):
        
        '''
        시작일:str, 
        위치_데이터:pd.DataFrame centroid 컬럼 필수
        총일수:int=365, 
        휴일_시간별_예약_평균:list = [6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 6], 
        업무일_시간별_예약_평균:list = [1, 1, 2, 2, 2, 2, 2, 2, 2, 10, 10, 8, 6], 
        '''
        
        self.위치_데이터 = 위치_데이터.reset_index(drop=True)
        self.시작일 = pd.to_datetime(시작일)
        print('시작일: ', self.시작일)
        seed = int(self.시작일.timestamp()+datetime.now().timestamp())  # logical date를 기반으로 시드 생성
        print(seed)
        np.random.seed(seed)
        self.오늘 = self.시작일
        self.총일수 = 총일수
        self.휴일_시간별_예약_평균 = 휴일_시간별_예약_평균
        self.비수기_업무일_시간별_예약_평균 = 비수기_업무일_시간별_예약_평균
        self.성수기_업무일_시간별_예약_평균 = 성수기_업무일_시간별_예약_평균
        self.engine = create_engine(f"postgresql://{db_info['username']}:{db_info['password']}@{db_info['ip']}:{db_info['port']}/{db_info['dbname']}")

        self.고객접속기록 = pd.DataFrame(columns = [
            # '고객_코드',
                                              '고객_접속일시', '고객_접속point', '예약희망일시', '예약희망_point', '예약희망_차종'])

        self.term_samples = term_samples
        self.term_generate_method = term_generate_method

        date_range = pd.date_range(start=self.시작일, periods=self.총일수+1, freq='D')
        self.tmp_df = pd.DataFrame(date_range, columns=['Date'])
        self.tmp_df = 구분데이터붙이기(self.tmp_df, 'Date')
        # if len() 가끔 안되니까 예외문 처리 해서 될때까지 하게하기
    def 데이터생성시뮬실행(self):
        # 고객_코드0 = 0
        for i in range(self.총일수):

            self.오늘예약기록 = pd.DataFrame()
            오늘휴일 = self._01_오늘휴일판단()
            self._02_오늘예약횟수생성()
            self._04_장소생성()
            # self.오늘예약기록['고객_코드'] = np.arange(고객_코드0, 고객_코드0 + len(self.오늘예약기록))
            # 고객_코드0 = 고객_코드0 + len(self.오늘예약기록)
            self._05_디비적재()

            # self.오늘예약기록 = pd.concat([self.오늘예약기록, ])

            self.오늘 = self.오늘 + pd.Timedelta(value = 1, unit = 'day')

    def _01_오늘휴일판단(self):
        # 현재 시간을 가져온다.

        # 오늘_년월일 = self.오늘.strftime("%Y-%m-%d")
        
        # 리스트_휴일 = pytimekr.holidays(self.오늘.year)
        
        # 휴일_유무 = False
        # for 날짜 in 리스트_휴일:
        #     if(오늘_년월일 == str(날짜)): 
        #         휴일_유무 = True  # 오늘이 공휴일이면 True
        
        휴일_유무 = False
        if self.오늘.weekday() > 4: # 오늘이 주말이면
            휴일_유무 = True
        공휴일_유무 = self.tmp_df[self.tmp_df['연월일'] == self.오늘.strftime("%Y%m%d")]['공휴일'].iloc[0]
        self.오늘휴일 = 휴일_유무 | 공휴일_유무

        print(self.오늘휴일)
        return self.오늘휴일
        
    def _02_오늘예약횟수생성(self): 
        print(self.오늘휴일)
        if self.오늘휴일 == True:
            시간별_예약_평균 = self.휴일_시간별_예약_평균
            print(시간별_예약_평균)
        elif self.오늘.month in [12, 1, 2, 6, 7, 8]:
            시간별_예약_평균 = self.성수기_업무일_시간별_예약_평균
        elif self.오늘.month in [3, 4, 5, 9, 10, 11]:
            시간별_예약_평균 = self.비수기_업무일_시간별_예약_평균
        print(시간별_예약_평균)
        self.오늘_시간별_예약_횟수 = np.random.poisson(lam = 시간별_예약_평균).tolist()
        print(self.오늘_시간별_예약_횟수)
        k = 0
        for i, 횟수 in enumerate(self.오늘_시간별_예약_횟수) :
            for j in range(횟수):
                예약_일시 = self.오늘.strftime("%Y-%m-%d")+' '
                if self.오늘휴일:
                    예약_일시 += self.휴일_예약가능시각[i]
                else:
                    예약_일시 += self.업무일_예약가능시각[i]
                if (random.random() >= 0.5)|(예약_일시[-2:]=='21'):
                    예약_일시 += ':00:00'
                else:
                    예약_일시 += ':30:00'
                예약_일시 = pd.to_datetime(예약_일시)
                if self.term_generate_method == 'uniform':
                    self.오늘예약기록.loc[k, '고객_접속일시'] = 예약_일시 - pd.Timedelta(value = np.random.randint(low = 144, high = 10080), unit = 'minute')  # 하루 ~ 1주일 전에 접속하여 예약
                self.오늘예약기록.loc[k, '예약희망일시'] = pd.to_datetime(예약_일시)

                k += 1
        if self.term_generate_method == 'invgamma':
            a, loc, scale = invgamma.fit(self.term_samples)
            terms = invgamma.rvs(a = a, scale = scale, size = len(self.오늘예약기록))
            terms2 = [term_smoothing(term) for term in terms]
            for i in range(len(self.오늘예약기록)):
                self.오늘예약기록.loc[i, '고객_접속일시'] = self.오늘예약기록.loc[i, '예약희망일시'] - terms2[i]
                
    def _04_장소생성(self):
        # idx_경계구s = np.random.randint(low = 0, high = , size = len(self.오늘예약기록))
        choices = random.choices(np.arange(0, len(self.위치_데이터)), weights = self.위치_데이터['수요인구'], k=len(self.오늘예약기록)) # 인구수에 비례하게 고객 위치 생성

        points = []
        for idx_집계구 in choices:
            points.append(self.위치_데이터.loc[idx_집계구, '중심'])
        self.오늘예약기록['고객_접속point'] = points
        self.오늘예약기록['고객_접속point'] = self.오늘예약기록['고객_접속point'].apply(lambda x: WKTElement(x.wkt, srid=4326))

        self.오늘예약기록.loc[:, '예약희망_point'] = self.오늘예약기록.loc[:, '고객_접속point']

    def _05_디비적재(self):
        self.오늘예약기록.sort_values('고객_접속일시', ascending = True, inplace = True)
        self.오늘예약기록.to_sql(name = '고객수요',
                                con = self.engine,
                                schema = 'datamart',
                                if_exists = 'append',
                                index = False, dtype={'고객_접속point': Geometry('POINT', srid=4326),
                                                     '예약희망_point': Geometry('POINT', srid=4326)})
def term_smoothing(term):
    '''
    14일 이상으로 생성된 것은 13일로 고정
    '''
     # (초단위) 29일 이상이면 일을 29일로 고정하고 그 밑은 그대로
    if term >= 2505600:
        term = 2505600 + term%86400
    # (초단위) 21시간 미만이면 21시간 1초로 고정
    if term < 75600:
        term = 75601 + term % 75600
    return pd.Timedelta(value = term, unit = 'second')
