import time

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
import random
import matplotlib.pyplot as plt 

from .db_io import Engine
from .utils import 구분데이터붙이기, which_시간대#, generate_polygon_random_points, 공휴일출력, 예약가능시간출력, 이웃시간판단, time_to_minutes

import requests
import pickle

from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp

from ortools.sat.python import cp_model



# 이십사절기 = ['동지', '소한', '대한', '입춘', '우수', '경칩', '춘분', '청명', '곡우', '입하', '소만', '망종', '하지', '소서', '대서', '입추', '처서', '백로', '추분', '한로', '상강', '입동', '소설', '대설']
며칠뒤 = 30
예약가능시간s = ['09_00', '09_30', '10_00', '10_30', '11_00', '11_30', '12_00', '12_30', '13_00', '13_30', '14_00', '14_30', '15_00', '15_30', '16_00', '16_30', '17_00', '17_30', '18_00', '18_30', '19_00', '19_30', '20_00', '20_30', '21_00']
시간_인덱스_딕셔너리 = {time: index for index, time in enumerate(예약가능시간s)}

class Make_cluster():
    def __init__(self, db_info, execution_time):
        self.db_handler = Engine(db_info)
        self._01_수요데이터불러오기(end_datetime = execution_time)
        self._02_구분데이터붙이기()
        self._03_클러스터링()
        # self._클러스터링결과저장()

    def _01_수요데이터불러오기(self, start_datetime : datetime.datetime = None, end_datetime = None):
        '''
        db_info(필) : user_name, password, host, port, db_name 이 key로 포함된 dict 형식
        start_datetime(선) : 수요데이터 불러올 시작일시. None일 경우 가장 오래된 것부터 불러옴.
        end_datetime(선) : 수요데이터 불러올 끝일시. None일 경우 db의 가장 최근 데이터까지 불러옴.
        '''
        

        wheres = []
        if start_datetime is not None:
            wheres.append('"예약희망일시" >= \'{st_dt}\''.format(
                st_dt = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            ))
        if end_datetime is not None:
            wheres.append('"예약희망일시" <= \'{ed_dt}\''.format(
                ed_dt = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
            ))
        self.data = self.db_handler.select(table = '고객수요', schema='datamart', columns = ['고객_접속일시', '예약희망일시', '예약희망_point'], wheres = wheres, orderbys = ['"예약희망일시" asc'], 
                                           gis_col = '예약희망_point', 
                                          )
        self.data['예약희망일시'] = self.data['예약희망일시'].dt.tz_convert('Asia/Seoul')
        self.data['고객_접속일시'] = self.data['고객_접속일시'].dt.tz_convert('Asia/Seoul')
        
    def _02_구분데이터붙이기(self):
        '''
        수요 데이터에 요일, 24절기, 휴일여부, 조조/점심/저녁, 기후(추후 추가) 등 구분 기준 붙이기
        '''
        self.data = 구분데이터붙이기(self.data, '예약희망일시')

    def _03_클러스터링(self):
        '''
        key : [(월, 요일, 공휴일, 시간대)]
        
        월 : '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
        요일 : '0', '1', '2', '3', '4', '5', '6'
        공휴일 : True, False
        시간대 : '조조', '점심', '저녁', '심야'
            (05 ~ 09), (10 ~ 16), (17 ~ 21), (22 ~ 04)
        '''
        value_kinds = {}
        value_kinds['월'] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        value_kinds['요일'] = ['0', '1', '2', '3', '4', '5', '6']
        value_kinds['공휴일'] = [True, False]
        value_kinds['시간대'] = ['조조', '점심', '저녁', '심야']
        # 모집단에 0 추가
        모집단 = self.data.drop_duplicates(subset = ['연월일', '시간대']).reset_index(drop = True).loc[:, ['연월일', '연', '월', '요일', 
                                                                                                       # '절기', 
                                                                                                       '공휴일', '시간대']]
        카운트 = self.data.groupby(['연월일', '시간대'])[['고객_접속일시']].count()
        for i in range(len(카운트)):
            모집단.loc[(모집단['연월일']==카운트.index[i][0])&(모집단['시간대']==카운트.index[i][1]), 'count'] = 카운트.iloc[i,0]
        self.모집단 = 모집단
        self.모집단count = self.모집단.groupby(['월', '요일', '공휴일', '시간대'])[['count']].count()

        모든키 = list(self.모집단count.index)
        self.cluster = Cluster(value_kinds, 모든키 = 모든키, 모든원소 = self.모집단)
        self.cluster._01_군집화()
        self.cluster._02_cutting(method = 'threshold', values = {'threshold': 0.05})

        self.hierarchy_sample_data = {}
        
        for key in list(self.cluster.군집s):
            self.hierarchy_sample_data[key] = pd.DataFrame(columns = ['고객_접속일시', '예약희망일시', '예약희망_point', '연월일', '연', '월', '요일', '공휴일', '시간대'])
            for condition in eval(key):
                condition1 = (self.data['월'] == condition[0])
                condition2 = (self.data['요일'] == condition[1])
                condition3 = (self.data['공휴일'] == condition[2])
                condition4 = (self.data['시간대'] == condition[3])
                self.hierarchy_sample_data[key] = pd.concat([self.hierarchy_sample_data[key], 
                                                            self.data.loc[condition1&condition2&condition3&condition4, :]]).reset_index(drop = True)

    def _클러스터링결과저장(self):
        클러스터링결과 = {}
        클러스터링결과['cluster'] = self.cluster
        클러스터링결과['hierarchy_sample_data'] = self.hierarchy_sample_data

        with open('./data/simulator/클러스터링결과.pickle', 'wb') as f:
            pickle.dump(클러스터링결과, f)
    
    def _클러스터링결과불러오기(self):
        with open('./data/simulator/클러스터링결과.pickle', 'rb') as f:
            클러스터링결과 = pickle.load(f)

        self.cluster = 클러스터링결과['cluster']
        self.hierarchy_sample_data = 클러스터링결과['hierarchy_sample_data']
        

class Simulator_01():
    
    def __init__(self, 위치데이터_path, simulation_start_dt, simulation_end_dt, db_info, is_first = False, data_start_date = pd.to_datetime('2022-01-01'), weight_학생숫자 = 1, 차량숫자 = 12):
        '''
        시뮬레이터에서 사용하는 point 좌표는 모두 위경도 -> 미터 단위로 바꾸기 
        df[''] = df[''].to_crs(epsg = 3857) (위경도좌표 -> 미터좌표)
        df[''] = df[''].to_crs(epsg=4326) (미터좌표 -> 위경도좌표)
        '''
        
        if is_first == True:
            self.db_handler = Engine(db_info)
            self._01_수요데이터불러오기(start_datetime = data_start_date)
            self._02_구분데이터붙이기()
            self._03_클러스터링()
            self._클러스터링결과저장()

            with open(위치데이터_path, 'rb') as f:
                강남3구_집계구_경계_중심 = pickle.load(f)
            강남3구_집계구_경계_중심.reset_index(drop = True, inplace = True)
            self.위치데이터 = 강남3구_집계구_경계_중심.loc[:, ['count', 'geometry', 'centroid']]
            self.위치데이터['geometry'] = self.위치데이터['geometry'].to_crs(epsg = 3857)
            self.위치데이터['centroid'] = self.위치데이터['centroid'].to_crs(epsg = 3857)
            # 위치별 30분 이내거리(지도직선거리 5키로) index
            self.위치_5000이내 = {}
            for i in range(len(self.위치데이터)):
                self.위치_5000이내[i] = []
                for j in range(i+1, len(self.위치데이터)):
                    if self.위치데이터.loc[i, 'centroid'].distance(self.위치데이터.loc[j, 'centroid']) <= 5000:
                        self.위치_5000이내[i].append(j)
                        if j not in list(self.위치_5000이내):
                            self.위치_5000이내[j] = []
                        self.위치_5000이내[j].append(i)
            self._위치데이터저장()

            self.차량데이터 = self.db_handler.select(table = '차량정보기록', schema='datamart', columns = ['차량코드', '차종', '기록일시', '주차_point'],
                                               gis_col = '주차_point', 
                                              )
            self.차량데이터['주차_point'] = self.차량데이터['주차_point'].to_crs(epsg = 3857)
            self.차량데이터['기록일시'] = self.차량데이터['기록일시'].dt.tz_convert('Asia/Seoul')
            self._차량데이터저장()
                        
        elif is_first == False:
            self._클러스터링결과불러오기()
            self._위치데이터불러오기()
            self._차량데이터불러오기()

        self.차량데이터 = self.차량데이터.iloc[:차량숫자, :]
        
        self.weight_학생숫자 = weight_학생숫자
        self._04_학생_생성(simulation_start_dt, simulation_end_dt, 위치_데이터 = self.위치데이터, weight_학생숫자 = self.weight_학생숫자)
        self.종료일 = pd.to_datetime(simulation_end_dt).tz_localize('Asia/Seoul')
        # 구나 동 붙인 view에서 불러오게 수정.
        
        
        
        self.금년 = '9999'
        self.오늘 = pd.to_datetime('1999-01-01 9:00:00').tz_localize('Asia/Seoul')
        self.위치별예약가능일시 = {}
        self.일자별최적화시행수 = {}
        self.예약가능일 = []
        self.예약가능일_info_state = {}
        
        
    def environment_init(self, simulation_start_dt, simulation_end_dt):
        self._클러스터링결과불러오기()
        self._04_학생_생성(simulation_start_dt, simulation_end_dt, 위치_데이터 = self.위치데이터, weight_학생숫자 = self.weight_학생숫자)

    def run(self, 선택최적 = 0, 선택rule = 'rule1'):
        '''
        self.남은예약자s 에서 한 명씩 불러오며 진행.
        1. 차량현황, 예약현황, 예약자 위치를 기반으로 예약 가능 목록 보여주기
        2. 예약 희망일시와 비교해 예약 여부 결정 (여러 개 있으면 아예 랜덤?)
        3. 날이 바뀔 때 당일 예약 돼있는 사람과 차량 매칭, 통계치 계산
        4. 반복
        종료 후 결과 통계 데이터 뽑기

        뽑을 데이터 :
            놓친 손님 수
            잡은 손님 수
            운행한 차량 수
            각 차량이 쉰 시간 비율
            초과해서 받아 차량으로 커버 못한 고객 수
            총 같은 차량으로 당일 교육받은 연속한 수강생 간 거리

            순서대로 다 받았을 때와 알고리즘 적용했을 때의 차이 봐야함.
                -> 같은 예약자s로 두 번 돌려야 함.
        '''
        self.count_total_예약 = 0
        self.count_total_고객 = 0
        self.count_total_수업수 = 0
        self.count_total_차량수 = 0
        
        self.고객접속기록 = {'고객_접속일시': [], '고객_접속point': [], '예약_유무': [], '예약_일시': [], '예약_point': [], '예약_차종': []}
        self.남은예약 = {'예약_일시': [], '예약_point': [], '예약_point_idx': [], '예약_차종': [], '연월일': [], '임시배정차량코드': []}
        self.지난예약 = {'예약_일시': [], '예약_point': [], '예약_point_idx': [], '예약_차종': [], '연월일': [], '임시배정차량코드': []}
        self.일자별예약고객수 = {'일자': [], 'count_예약': [], 'count_고객': []}

        self.남은예약자s = self.예약자s.copy()

        self.첫접속일 = self.남은예약자s[0]['접속일시']
        self.접속일 = self.첫접속일
        self.선택최적 = 선택최적
        self.선택rule = 선택rule
        
        self.past_예약자s = []
        sim_st = time.time()
        for 오늘 in pd.date_range(start=self.첫접속일, end=self.종료일+pd.Timedelta(value = 1, unit = 'day'), freq='1D'):
            print(오늘.strftime('%Y%m%d'))
            day_st = time.time()
            self.count_오늘예약 = 0
            self.count_오늘고객 = 0
            self.오늘 = 오늘
            
            self._차량최적화()
            
            
                    
            for i, 예약자 in enumerate(self.남은예약자s.copy()):
                # print(예약자)
                if self.오늘.strftime('%Y%m%d') != 예약자['접속일시'].strftime('%Y%m%d'):
                    break
                # 차량 점검 등 반영 x (반영계획 없음. 따라서 그냥 차량정보 쭉 사용)
                # 예약가능일시 = self._예약가능일판단(예약자 = 예약자, 예약현황 = self.남은예약, 차량현황 = self.차량데이터)
                # 예약가능일 = 예약가능일
                self.count_오늘고객 += 1
                self.count_total_고객 += 1
                self._예약선택(예약자)

                self.past_예약자s.append(self.남은예약자s.pop(0))
                
            self._통계값산출()
            time_ed = time.time()
            
            print("{:<15}".format('하루 실행 시간')+': ' "{:^6}".format(round(time_ed - day_st, 2)), "{:<15}".format('총 실행 시간')+': ' "{:^6}".format(round(time_ed - sim_st, 2)))
            print('\n')

    def reset(self, 선택최적 = 0, 선택rule = 'rule1'):
        '''
        self.남은예약자s 에서 한 명씩 불러오며 진행.
        1. 차량현황, 예약현황, 예약자 위치를 기반으로 예약 가능 목록 보여주기
        2. 예약 희망일시와 비교해 예약 여부 결정 (여러 개 있으면 아예 랜덤?)
        3. 날이 바뀔 때 당일 예약 돼있는 사람과 차량 매칭, 통계치 계산
        4. 반복
        종료 후 결과 통계 데이터 뽑기

        뽑을 데이터 :
            놓친 손님 수
            잡은 손님 수
            운행한 차량 수
            각 차량이 쉰 시간 비율
            초과해서 받아 차량으로 커버 못한 고객 수
            총 같은 차량으로 당일 교육받은 연속한 수강생 간 거리

            순서대로 다 받았을 때와 알고리즘 적용했을 때의 차이 봐야함.
                -> 같은 예약자s로 두 번 돌려야 함.
        '''

        self.count_total_예약 = 0
        self.count_total_고객 = 0
        self.count_total_수업수 = 0
        self.count_total_차량수 = 0
        self.하루_강화학습_거절횟수 = 0
        self.total_강화학습_거절횟수 = 0
        self.하루_rule_거절횟수 = 0
        self.total_rule_거절횟수 = 0

        self.금년 = '9999'
        self.오늘 = pd.to_datetime('1999-01-01 9:00:00').tz_localize('Asia/Seoul')
        self.위치별예약가능일시 = {}
        self.일자별최적화시행수 = {}
        self.예약가능일 = []
        self.예약가능일_info_state = {}
        
        self.고객접속기록 = {'고객_접속일시': [], '고객_접속point': [], '예약_유무': [], '예약_일시': [], '예약_point': [], '예약_차종': []}
        self.남은예약 = {'예약_일시': [], '예약_point': [], '예약_point_idx': [], '예약_차종': [], '연월일': [], '임시배정차량코드': []}
        self.지난예약 = {'예약_일시': [], '예약_point': [], '예약_point_idx': [], '예약_차종': [], '연월일': [], '임시배정차량코드': []}
        self.일자별예약고객수 = {'일자': [], 'count_예약': [], 'count_고객': []}

        self.남은예약자s = self.예약자s.copy()

        self.첫접속일 = self.남은예약자s[0]['접속일시']
        self.접속일 = self.첫접속일
        self.선택최적 = 선택최적
        self.선택rule = 선택rule
        self.simulation_done = False
        
        self.past_예약자s = []
        self.sim_st = time.time()

        self.date_range = list(pd.date_range(start=self.첫접속일, end=self.종료일+pd.Timedelta(value = 1, unit = 'day'), freq='1D'))
        self.오늘 = self.date_range.pop(0)
        print(self.오늘.strftime('%Y%m%d'))

        self.day_st = time.time()
        self.count_오늘예약 = 0
        self.count_오늘고객 = 0
            
        self._차량최적화()


        self.예약자 = self.남은예약자s.pop(0)
        self.past_예약자s.append(self.예약자)
        
        state = []
        for i in range(며칠뒤):
            tmp_시 = pd.to_datetime(self.예약자['접속일시'].strftime('%Y%m%d')) + pd.Timedelta(value = i+1, unit = 'day')
            tmp_시차 = tmp_시 - pd.to_datetime(self.예약자['접속일시'].strftime('%Y-%m-%d %H:%M:%S'))
            tmp = self.학생생성기준일[self.학생생성기준일['연월일'] == tmp_시.strftime('%Y%m%d')].reset_index(drop = True)
            
            continuous_sate = [tmp_시차.total_seconds(), self.예약자['예약희망_point_idx'], self.위치데이터.loc[self.예약자['예약희망_point_idx'], 'count'], tmp.loc[0, '휴일'], tmp['mean'].mean(), tmp['std'].mean()]
            sequence_state = []
            for tmp_list in self.예약가능일_info_state[(self.예약자['접속일시']+pd.Timedelta(value = i+1, unit = 'day')).strftime('%Y%m%d')]:
                sequence_state.append([self.예약자['예약희망_point'].distance(x) for x in tmp_list])
                continuous_sate.append(len(tmp_list))
            state.append([continuous_sate, sequence_state])
        return state


    def step(self, actions, is_reinforce = True):
        reward = 0
        next_state = [0]
        dones = [0]*며칠뒤

        if self.simulation_done == False:
            reward = self._예약선택(self.예약자, actions, is_reinforce)
            
            self.count_오늘고객 += 1
            self.count_total_고객 += 1
        
            next_state = []
            if len(self.남은예약자s) == 0:
                '''
                시뮬레이션 종료 조건
                '''
                dones = [1]*며칠뒤
                self.simulation_done = True
                while len(self.date_range) > 0:
                    
                    self._하루reset()

                self._통계값산출()
                next_state = [[[0]*6, [[] for _ in range(25)]] for i in range(30)]
            else:
                self.예약자 = self.남은예약자s.pop(0)
                self.past_예약자s.append(self.예약자)

                

                while self.오늘.strftime('%Y%m%d') != self.예약자['접속일시'].strftime('%Y%m%d'):
                    '''
                    다음 예약자가 하루 넘어감
                    '''
                    reward = self._하루reset()
                    dones[0] = 1
                    
                
                '''
                state: [[continuous_sate, sequence_state]*며칠뒤]
                continuous_state: [남은 시간, 집계구_index, 집계구_인구]
                sequence_state: [[[예약한사람과 이사람과의 거리]*25]*최대10]
                '''
                for i in range(며칠뒤):
                    tmp_시 = pd.to_datetime(self.예약자['접속일시'].strftime('%Y%m%d')) + pd.Timedelta(value = i+1, unit = 'day')
                    tmp_시차 = tmp_시 - pd.to_datetime(self.예약자['접속일시'].strftime('%Y-%m-%d %H:%M:%S'))
                    tmp = self.학생생성기준일[self.학생생성기준일['연월일'] == tmp_시.strftime('%Y%m%d')].reset_index(drop = True)
                    
                    continuous_sate = [tmp_시차.total_seconds(), self.예약자['예약희망_point_idx'], self.위치데이터.loc[self.예약자['예약희망_point_idx'], 'count'], tmp.loc[0, '휴일'], tmp['mean'].mean(), tmp['std'].mean()]
                    

                    sequence_state = []
                    for tmp_list in self.예약가능일_info_state[(self.예약자['접속일시']+pd.Timedelta(value = i+1, unit = 'day')).strftime('%Y%m%d')]:
                        sequence_state.append([self.예약자['예약희망_point'].distance(x) for x in tmp_list])
                        continuous_sate.append(len(tmp_list))
                    next_state.append([continuous_sate, sequence_state])
            
            
            if dones is not True:
                pass

        else:
            dones = [1]*며칠뒤
            next_state = [[[0]*6, [[] for _ in range(25)]] for i in range(30)]
            pass
            
        return next_state, reward, dones
            
    def make_init_states(self, sample_num = 100):
        tmp_시 = pd.Timedelta(value = 29, unit = 'day').total_seconds() + random.randint(0, 86399)
        tmp = self.위치데이터.sample(n=sample_num)
        states = []
        for i in range(sample_num):
            연월일 = np.random.choice(self.학생생성기준일['연월일'].unique())
            tmp_df = self.학생생성기준일[self.학생생성기준일['연월일'] == 연월일].reset_index(drop = True)
            
            continuous_sate = [tmp_시, tmp.index[i], tmp['count'].iloc[i], tmp_df.loc[0, '휴일'], tmp_df['mean'].mean(), tmp_df['std'].mean()] + [0]*25
            sequence_state = [[] for _ in range(25)]
            states.append([continuous_sate, sequence_state])
        return states

    def _하루reset(self):

        

        
        self.오늘 = self.date_range.pop(0)
        print(self.오늘.strftime('%Y%m%d'))
        self._차량최적화()
        self._통계값산출()
        self.time_ed = time.time()
        
        print("{:<15}".format('하루 실행 시간')+': ' "{:^6}".format(round(self.time_ed - self.day_st, 2)), "{:<15}".format('총 실행 시간')+': ' "{:^6}".format(round(self.time_ed - self.sim_st, 2)))
        print('\n')
        
        try:
            reward = self.count_오늘_수업수/self.count_오늘_차량수
        except:
            reward = 0
        self.day_st = time.time()
        self.count_오늘예약 = 0
        self.count_오늘고객 = 0
        self.하루_강화학습_거절횟수 = 0
        self.하루_rule_거절횟수 = 0
        return reward
        


    def _통계값산출(self):
        '''
        뽑을 데이터 :
            놓친 손님 수
            잡은 손님 수
            운행한 차량 수
            각 차량이 쉰 시간 비율
            초과해서 받아 차량으로 커버 못한 고객 수
            총 같은 차량으로 당일 교육받은 연속한 수강생 간 거리

            순서대로 다 받았을 때와 알고리즘 적용했을 때의 차이 봐야함.
                -> 같은 예약자s로 두 번 돌려야 함.
        '''
        self.일자별예약고객수['일자'].append((self.오늘-pd.Timedelta(value = 1, unit = 'day')).strftime('%Y%m%d'))
        self.일자별예약고객수['count_예약'].append(self.count_오늘예약)
        self.일자별예약고객수['count_고객'].append(self.count_오늘고객)
        print("{:<15}".format('하루 예약자 수')+': ' "{:^6}".format(self.count_오늘예약), "{:<15}".format('total 예약자 수')+': ' "{:^6}".format(self.count_total_예약))
        print("{:<15}".format('하루 접속 고객 수')+': ' "{:^6}".format(self.count_오늘고객), "{:<15}".format('total 접속 고객 수')+': ' "{:^6}".format(self.count_total_고객))
        if self.count_오늘고객 != 0:
            print("{:<15}".format('하루 예약자 비율')+': ' "{:^6}".format(round(100*(self.count_오늘예약/self.count_오늘고객), 2)), "{:<15}".format('total 예약자 비율')+': ' "{:^6}".format(round(100*(self.count_total_예약/self.count_total_고객), 2)))
        print("{:<15}".format('하루 수업 횟수')+': ' "{:^6}".format(self.count_오늘_수업수), "{:<15}".format('total 수업 횟수')+': ' "{:^6}".format(self.count_total_수업수))
        if self.count_오늘_수업수 != 0:
            print("{:<15}".format('하루 차량당 수업횟수')+': ' "{:^6}".format(round((self.count_오늘_수업수/self.count_오늘_차량수), 2)), "{:<15}".format('total 차량당 수업횟수')+': ' "{:^6}".format(round((self.count_total_수업수/self.count_total_차량수), 2)))
        print("{:<15}".format('하루 Rule 거절 횟수')+': ' "{:^6}".format(self.하루_rule_거절횟수), "{:<15}".format('total Rule 거절 횟수')+': ' "{:^6}".format(self.total_rule_거절횟수))
        print("{:<15}".format('하루 강화학습 거절 횟수')+': ' "{:^6}".format(self.하루_강화학습_거절횟수), "{:<15}".format('total 강화학습 거절 횟수')+': ' "{:^6}".format(self.total_강화학습_거절횟수))
        
            
        
        
        
    def _예약선택(self, 예약자, actions, is_reinforce = True):
        '''
        예약 선택하면 차량 배정하고 차량별 배정 가능한 경계구 리스트, 예약 가능 시간 True/False 리스트 업데이트

        예약 성공하면 reward + 1
        '''
        reward = 0
        
        self.고객접속기록['고객_접속일시'].append(예약자['접속일시'])
        self.고객접속기록['고객_접속point'].append(예약자['예약희망_point'])

        # tmp_예약가능일 = 예약가능일[예약가능일['is_예약가능']==True]
        # tmp_예약가능일 = 예약가능일.copy()
        예약일s = []
        for 희망일시 in 예약자['예약희망일시']:
            # 랜덤으로 가능한 차 중에 고르기
            # 일단 임시배정으로, 할거 없으면 최적화, 최적화 횟수  count해서 최대 3회까지만.


            try:
                차량idx = random.choice(self.위치별예약가능일시[희망일시.strftime('%Y%m%d_%H_%M')][예약자['예약희망_point_idx']])
                if self.선택rule == 'rule1':
                    if (희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17'):
                        self.하루_rule_거절횟수 += 1
                        self.total_rule_거절횟수 += 1
                        continue
                elif self.선택rule == 'rule2':
                    tmp = self.학생생성기준일[self.학생생성기준일['연월일'] == 희망일시.strftime('%Y%m%d')].reset_index(drop = True)
                    d = (pd.to_datetime(희망일시.strftime('%Y%m%d')) - pd.to_datetime(예약자['접속일시'].strftime('%Y%m%d'))).days
                    if (d > 1)&(tmp.loc[0, '휴일'] == False)&((희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17')):
                        
                        self.하루_rule_거절횟수 += 1
                        self.total_rule_거절횟수 += 1
                        continue
                elif self.선택rule == 'rule3':
                    d = (pd.to_datetime(희망일시.strftime('%Y%m%d')) - pd.to_datetime(예약자['접속일시'].strftime('%Y%m%d'))).days
                    
                    if (d > 1)&((희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17')):
                        
                        self.하루_rule_거절횟수 += 1
                        self.total_rule_거절횟수 += 1
                        continue
                elif self.선택rule == 'rule4':
                    ts = (pd.to_datetime(희망일시.strftime('%Y%m%d')) - pd.to_datetime(예약자['접속일시'].strftime('%Y%m%d'))).total_seconds()
                    tmp = self.학생생성기준일[self.학생생성기준일['연월일'] == 희망일시.strftime('%Y%m%d')].reset_index(drop = True)
                    if (tmp.loc[0, '휴일'] == False)&(ts > 43200)&((희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17')):
                        
                        self.하루_rule_거절횟수 += 1
                        self.total_rule_거절횟수 += 1
                        continue

                # action 적용 부분
                if  is_reinforce == True:
                    if (actions[(희망일시.date() - self.오늘.date()).days - 1][0][시간_인덱스_딕셔너리[희망일시.strftime('%H_%M')]] < 0):
                        self.하루_강화학습_거절횟수 += 1
                        self.total_강화학습_거절횟수 += 1
                        continue
                예약일s.append([희망일시, 차량idx, 0])
            except:
                if self.일자별최적화시행수[희망일시.strftime('%Y%m%d_%H_%M')] == self.선택최적:
                    # 최적화 진행
                    tmp_남예 = pd.DataFrame(self.남은예약)
                    tmp_남예 = tmp_남예[tmp_남예['연월일'] == 희망일시.strftime('%Y%m%d')]
    
                    # 탐색 대상 일시는 마지막에 append
                    customer_times = [time_to_minutes(x.strftime('%H_%M')) for x in tmp_남예['예약_일시']]
                    customer_times.append(time_to_minutes(희망일시.strftime('%H_%M')))
                    
                    customer_loc_idxes = [x for x in tmp_남예['예약_point_idx']]
                    customer_loc_idxes.append(예약자['예약희망_point_idx'])
    
                    
                    num_vehicles = len(self.차량데이터)
                    num_customers = len(customer_times)
                    
                    # 모델 생성
                    model = cp_model.CpModel()
                    
                    # 변수 생성: 각 고객을 어느 차량에 배정할지 결정
                    vehicle_assignments = [model.NewIntVar(0, num_vehicles - 1, f'customer_{i}') for i in range(num_customers)]
                    vehicle_used = [model.NewIntVar(0, 1, f'vehicle_{j}') for j in range(num_vehicles)]
                    
                    for i in range(num_customers):
                        for j in range(i + 1, num_customers):
                            if i in self.위치_5000이내[j]:
                                if abs(customer_times[i] - customer_times[j]) <= 120:
                                    model.Add(vehicle_assignments[i] != vehicle_assignments[j])
                            else:
                                if abs(customer_times[i] - customer_times[j]) <= 150:
                                    model.Add(vehicle_assignments[i] != vehicle_assignments[j])
    
                    # 각 고객이 배정된 차량이 실제로 사용되었는지 여부
                    for i in range(num_customers):
                        model.AddElement(vehicle_assignments[i], vehicle_used, 1)
                    # for i in range(num_customers):
                    #     model.Add(vehicle_used[vehicle_assignments[i]] == 1)
                    
                    # 목적식: 사용된 차량의 수를 최소화
                    model.Minimize(sum(vehicle_used))
                    
                    # 최적화: 모델을 해결
                    solver = cp_model.CpSolver()
                    # solver.parameters.max_time_in_seconds = 10.0
                    status = solver.Solve(model)
                    

                    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
                        # print(희망일시.strftime('%Y%m%d'))
                        if self.선택rule == 'rule1':
                            if (희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17'):
                                self.하루_rule_거절횟수 += 1
                                self.total_rule_거절횟수 += 1
                                continue
                        elif self.선택rule == 'rule2':
                            tmp = self.학생생성기준일[self.학생생성기준일['연월일'] == 희망일시.strftime('%Y%m%d')].reset_index(drop = True)
                            if (tmp.loc[0, '휴일'] == False)&((희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17')):
                                
                                self.하루_rule_거절횟수 += 1
                                self.total_rule_거절횟수 += 1
                                continue
                        elif self.선택rule == 'rule3':
                            d = (pd.to_datetime(희망일시.strftime('%Y%m%d')) - pd.to_datetime(예약자['접속일시'].strftime('%Y%m%d'))).days
                            
                            if (d > 1)&((희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17')):
                                
                                self.하루_rule_거절횟수 += 1
                                self.total_rule_거절횟수 += 1
                                continue
                        elif self.선택rule == 'rule4':
                            ts = (pd.to_datetime(희망일시.strftime('%Y%m%d')) - pd.to_datetime(예약자['접속일시'].strftime('%Y%m%d'))).total_seconds()
                            
                            if (ts > 43200)&((희망일시.strftime('%H')=='19')|(희망일시.strftime('%H_%M')=='20_00')|(희망일시.strftime('%H')=='16')|(희망일시.strftime('%H')=='17')):
                                
                                self.하루_rule_거절횟수 += 1
                                self.total_rule_거절횟수 += 1
                                continue
            
                        # action 적용 부분
                        if  is_reinforce == True:
                            if (actions[(희망일시.date() - self.오늘.date()).days - 1][0][시간_인덱스_딕셔너리[희망일시.strftime('%H_%M')]] < 0):
                                self.하루_강화학습_거절횟수 += 1
                                self.total_강화학습_거절횟수 += 1
                                continue
                        
                        예약일s.append([희망일시, solver.Value(vehicle_assignments[-1]), 1, [solver.Value(vehicle_assignments[i]) for i in range(num_customers-1)], customer_loc_idxes[:-1], [x for x in tmp_남예['예약_일시']]])

         
                    else:
                        self.일자별최적화시행수[희망일시.strftime('%Y%m%d_%H_%M')] += 1
        

        if len(예약일s) > 0:
            '''
            현재 가능한 것중에 랜덤으로 고름. 
            (통계는 이미 가능일에 반영이 돼있음.)
            '''
            # reward += 1/1500
            tmp = random.sample(예약일s, 1)[0]
            # print(tmp)
            예약일 = tmp[0]
            차량코드s = [tmp[1]]
            최적화여부 = tmp[2]

            
            self.고객접속기록['예약_유무'].append(True)
            self.고객접속기록['예약_일시'].append(예약일)
            self.고객접속기록['예약_point'].append(예약자['예약희망_point'])
            self.고객접속기록['예약_차종'].append('2종보통')

            self.남은예약['예약_일시'].append(예약일)
            self.남은예약['예약_point'].append(예약자['예약희망_point'])
            self.남은예약['예약_point_idx'].append(예약자['예약희망_point_idx'])
            self.남은예약['예약_차종'].append('2종보통')
            self.남은예약['연월일'].append(예약일.strftime('%Y%m%d'))

            self.남은예약['임시배정차량코드'].append(차량코드s[0])
            
            self.count_오늘예약 += 1
            self.count_total_예약 += 1

            self.예약가능일_info_state[예약일.strftime('%Y%m%d')][시간_인덱스_딕셔너리[예약일.strftime('%H_%M')]].append(예약자['예약희망_point'])

            
            idxes_5000이내s = []
            예약일시s = []
            if 최적화여부 == 1:
                차량코드s = tmp[3] + 차량코드s
                고객위치s = tmp[4]
                예약일시s += tmp[5]
                num_customers = len(차량코드s)
                
                # 미래일 없는거 찾아 추가하기
                tmp_append = []
                tmp_연월일 = 예약일.strftime('%Y%m%d')
                for 예약가능시간 in 예약가능시간s:
                    tmp_append.append(tmp_연월일+'_'+예약가능시간)
                # 첫 차량 코드 (코드말고 인덱스로)
                for column in tmp_append:
                    self.위치별예약가능일시[column] = [np.arange(0, len(self.차량데이터)).tolist()]*len(self.위치데이터)
                
                condition = lambda x: x == 예약일.strftime('%Y%m%d')
                idxes = [index for index, value in enumerate(self.남은예약['연월일']) if condition(value)]
                for i in range(num_customers-1):
                    # print(차량코드s)
                    tmp_차량코드 = 차량코드s[i]
                    self.남은예약['임시배정차량코드'][idxes[i]] = tmp_차량코드
                    
                    idxes_5000이내s.append(self.위치_5000이내[고객위치s[i]]) # tmp[4]: customer_loc_idxes
                
            else:
                num_customers = 1
            idxes_5000이내s.append(self.위치_5000이내[예약자['예약희망_point_idx']])
            예약일시s.append(예약일)
                
            for i in range(num_customers):
                tmp_차량코드 = 차량코드s[i]
                idxes_5000이내 = idxes_5000이내s[i]
                
                idxes_5000이외 = list(set(np.arange(0, len(self.위치데이터)).tolist()) - set(idxes_5000이내))
    
                # 2두시간 이내는 아예 예약 불가
                앞 = max(pd.to_datetime(예약일시s[i].strftime('%Y-%m-%d') + ' 9:00:00').tz_localize('Asia/Seoul'), 예약일시s[i] -pd.Timedelta(value = 120, unit = 'minutes'))
                뒤 = min(pd.to_datetime(예약일시s[i].strftime('%Y-%m-%d') + ' 21:00:00').tz_localize('Asia/Seoul'), 예약일시s[i] + pd.Timedelta(value = 120, unit = 'minutes'))
                for tmp_datetime in pd.date_range(start=앞, end=뒤, freq='30min'):
                    for idx in np.arange(0, len(self.위치데이터)).tolist():
                        try:
                            self.위치별예약가능일시[tmp_datetime.strftime('%Y%m%d_%H_%M')][idx].remove(tmp_차량코드)
                        except:
                            pass
    
                # 2시간 30분은 5km이외에 있는 경우 예약 불가
                이전시간 = 예약일시s[i] - pd.Timedelta(value = 150, unit = 'minutes')
                if 이전시간 >= pd.to_datetime(예약일시s[i].strftime('%Y-%m-%d') + ' 9:00:00').tz_localize('Asia/Seoul'):
                    for idx in idxes_5000이외:
                        try:
                            self.위치별예약가능일시[이전시간.strftime('%Y%m%d_%H_%M')][idx].remove(tmp_차량코드)
                        except:
                            pass
                이후시간 = 예약일시s[i] + pd.Timedelta(value = 150, unit = 'minutes')
                if 이후시간 <= pd.to_datetime(예약일시s[i].strftime('%Y-%m-%d') + ' 21:00:00').tz_localize('Asia/Seoul'):
                    for idx in idxes_5000이외:
                        try:
                            self.위치별예약가능일시[이후시간.strftime('%Y%m%d_%H_%M')][idx].remove(tmp_차량코드)
                        except:
                            pass
                        
        
        else:
            self.고객접속기록['예약_유무'].append(False)
            self.고객접속기록['예약_일시'].append(None)
            self.고객접속기록['예약_point'].append(None)
            self.고객접속기록['예약_차종'].append(None)
        return reward

    def _차량최적화(self):


        # todo 최적화 진행 코드
        tmp_남예 = pd.DataFrame(self.남은예약)
        예약s = tmp_남예[tmp_남예['연월일']==self.오늘.strftime('%Y%m%d')].reset_index(drop = True)
        self.count_오늘_수업수 = len(예약s)
        self.count_total_수업수 += self.count_오늘_수업수




        
        if len(예약s) > 0 :
            # 탐색 대상 일시는 마지막에 append
            customer_times = [time_to_minutes(x.strftime('%H_%M')) for x in 예약s['예약_일시']]
            
            customer_loc_idxes = [x for x in 예약s['예약_point_idx']]
    
            num_vehicles = len(self.차량데이터)
            num_customers = len(customer_times)
            
            # 모델 생성
            model = cp_model.CpModel()
            
            # 변수 생성: 각 고객을 어느 차량에 배정할지 결정
            vehicle_assignments = [model.NewIntVar(0, num_vehicles - 1, f'customer_{i}') for i in range(num_customers)]
            vehicle_used = [model.NewIntVar(0, 1, f'vehicle_{j}') for j in range(num_vehicles)]
            
            for i in range(num_customers):
                for j in range(i + 1, num_customers):
                    if i in self.위치_5000이내[j]:
                        if abs(customer_times[i] - customer_times[j]) <= 120:
                            model.Add(vehicle_assignments[i] != vehicle_assignments[j])
                    else:
                        if abs(customer_times[i] - customer_times[j]) <= 150:
                            model.Add(vehicle_assignments[i] != vehicle_assignments[j])
    
            # 각 고객이 배정된 차량이 실제로 사용되었는지 여부
            for i in range(num_customers):
                model.AddElement(vehicle_assignments[i], vehicle_used, 1)
            
            # 목적식: 사용된 차량의 수를 최소화
            model.Minimize(sum(vehicle_used))
            
            # 최적화: 모델을 해결
            solver = cp_model.CpSolver()
            # solver.parameters.max_time_in_seconds = 10.0
            status = solver.Solve(model)
            
 
            if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
                condition = lambda x: x == self.오늘.strftime('%Y%m%d')
                idxes = [index for index, value in enumerate(self.남은예약['연월일']) if condition(value)]
                # print(idxes)
                for i in range(num_customers):
                    tmp_차량코드 = solver.Value(vehicle_assignments[i])
                    self.남은예약['임시배정차량코드'][idxes[i]] = tmp_차량코드
                    
                tmp_남예 = pd.DataFrame(self.남은예약)
                예약s = tmp_남예[tmp_남예['연월일']==self.오늘.strftime('%Y%m%d')].reset_index(drop = True)
        
            else:
                print('뭔가잘못됨. 되는 것만 받았는데 FEASIBLE SOLUTION이 없는 말이 안되는 상황.')


            
            self.count_오늘_차량수 = len(예약s['임시배정차량코드'].unique())
            self.count_total_차량수 += self.count_오늘_차량수
            indices_to_remove = np.where(tmp_남예['연월일'] == self.오늘.strftime('%Y%m%d'))[0].tolist()
            # 역순으로 인덱스를 정렬하여 삭제
            for index in sorted(indices_to_remove, reverse=True):
                self.지난예약['예약_일시'].append(self.남은예약['예약_일시'].pop(index))
                self.지난예약['예약_point'].append(self.남은예약['예약_point'].pop(index))
                self.지난예약['예약_point_idx'].append(self.남은예약['예약_point_idx'].pop(index))
                self.지난예약['예약_차종'].append(self.남은예약['예약_차종'].pop(index))
                self.지난예약['연월일'].append(self.남은예약['연월일'].pop(index))
                self.지난예약['임시배정차량코드'].append(self.남은예약['임시배정차량코드'].pop(index))


        # 연월일이 바꼈으면 self.예약가능일, self.위치별예약가능일시 두 개 에서 과거 예약일을 다 없애고 30일 이내 예약가능일시 컬럼을 추가한다.
        # self.위치별예약가능일시 는 딕셔너리 형태로 관리하는게 훨배 나을 듯. (각 시간대별로 예약가능 차량 코드를 리스트형태로 관리)
        # 근데 이러면 예약 가능일시 탐색이 어려워지니까 True/False dataframe도 따로 또 관리 -> 오키도키요

        
        # 과거일 다 없애기
        지난날s = [s for s in self.예약가능일 if int(s) <= int(self.오늘.strftime('%Y%m%d'))]
        tmp_drop = []
        for 지난날 in 지난날s:
            self.예약가능일.remove(지난날)
            del self.예약가능일_info_state[지난날]
            for 예약가능시간 in 예약가능시간s:
                tmp_drop.append(지난날+'_'+예약가능시간)
                
        for key in tmp_drop:
            del self.위치별예약가능일시[key]
            del self.일자별최적화시행수[key]

        # 미래일 없는거 찾아 추가하기
        tmp_append = []
        for value in range(1, 며칠뒤+1):
            tmp_연월일 = (self.오늘 + pd.Timedelta(value = value, unit = 'day')).strftime('%Y%m%d')
            
            if tmp_연월일 not in self.예약가능일:
                self.예약가능일.append(tmp_연월일)
                self.예약가능일_info_state[tmp_연월일] = [[] for _ in range(25)] # 좌표로 저장해놓고 obseve할 때 거리 계산
                
                for 예약가능시간 in 예약가능시간s:
                    tmp_append.append(tmp_연월일+'_'+예약가능시간)
                    self.일자별최적화시행수[tmp_연월일+'_'+예약가능시간] = 0

        # 첫 차량 코드 (코드말고 인덱스로)
        for column in tmp_append:
            self.위치별예약가능일시[column] = [np.arange(0, len(self.차량데이터)).tolist()]*len(self.위치데이터)

    
    def _01_수요데이터불러오기(self, start_datetime : datetime.datetime = None, end_datetime = None):
        '''
        db_info(필) : user_name, password, host, port, db_name 이 key로 포함된 dict 형식
        start_datetime(선) : 수요데이터 불러올 시작일시. None일 경우 가장 오래된 것부터 불러옴.
        end_datetime(선) : 수요데이터 불러올 끝일시. None일 경우 db의 가장 최근 데이터까지 불러옴.
        '''
        

        wheres = []
        if start_datetime is not None:
            wheres.append('"예약희망일시" >= \'{st_dt}\''.format(
                st_dt = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
            ))
        if end_datetime is not None:
            wheres.append('"예약희망일시" <= \'{ed_dt}\''.format(
                ed_dt = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
            ))
        self.data = self.db_handler.select(table = '고객수요', schema='datamart', columns = ['고객_접속일시', '예약희망일시', '예약희망_point'], wheres = wheres, orderbys = ['"예약희망일시" asc'], 
                                           gis_col = '예약희망_point', 
                                          )
        self.data['예약희망일시'] = self.data['예약희망일시'].dt.tz_convert('Asia/Seoul')
        self.data['고객_접속일시'] = self.data['고객_접속일시'].dt.tz_convert('Asia/Seoul')
        
        
    def _02_구분데이터붙이기(self):
        '''
        수요 데이터에 요일, 24절기, 휴일여부, 조조/점심/저녁, 기후(추후 추가) 등 구분 기준 붙이기
        '''
        self.data = 구분데이터붙이기(self.data, '예약희망일시')


    
    def _03_클러스터링(self):
        '''
        key : [(월, 요일, 공휴일, 시간대)]
        
        월 : '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
        요일 : '0', '1', '2', '3', '4', '5', '6'
        공휴일 : True, False
        시간대 : '조조', '점심', '저녁', '심야'
            (05 ~ 09), (10 ~ 16), (17 ~ 21), (22 ~ 04)
        '''
        value_kinds = {}
        value_kinds['월'] = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        value_kinds['요일'] = ['0', '1', '2', '3', '4', '5', '6']
        value_kinds['공휴일'] = [True, False]
        value_kinds['시간대'] = ['조조', '점심', '저녁', '심야']
        # 모집단에 0 추가
        모집단 = self.data.drop_duplicates(subset = ['연월일', '시간대']).reset_index(drop = True).loc[:, ['연월일', '연', '월', '요일', 
                                                                                                       # '절기', 
                                                                                                       '공휴일', '시간대']]
        카운트 = self.data.groupby(['연월일', '시간대'])[['고객_접속일시']].count()
        for i in range(len(카운트)):
            모집단.loc[(모집단['연월일']==카운트.index[i][0])&(모집단['시간대']==카운트.index[i][1]), 'count'] = 카운트.iloc[i,0]
        self.모집단 = 모집단
        self.모집단count = self.모집단.groupby(['월', '요일', '공휴일', '시간대'])[['count']].count()

        모든키 = list(self.모집단count.index)
        self.cluster = Cluster(value_kinds, 모든키 = 모든키, 모든원소 = self.모집단)
        self.cluster._01_군집화()
        self.cluster._02_cutting(method = 'threshold', values = {'threshold': 0.05})

        self.hierarchy_sample_data = {}
        
        for key in list(self.cluster.군집s):
            self.hierarchy_sample_data[key] = pd.DataFrame(columns = ['고객_접속일시', '예약희망일시', '예약희망_point', '연월일', '연', '월', '요일', '공휴일', '시간대'])
            for condition in eval(key):
                condition1 = (self.data['월'] == condition[0])
                condition2 = (self.data['요일'] == condition[1])
                condition3 = (self.data['공휴일'] == condition[2])
                condition4 = (self.data['시간대'] == condition[3])
                self.hierarchy_sample_data[key] = pd.concat([self.hierarchy_sample_data[key], 
                                                            self.data.loc[condition1&condition2&condition3&condition4, :]]).reset_index(drop = True)
                        
    def _04_학생_생성(self, dt_start, dt_end, 위치_데이터, weight_학생숫자 = 1):
        '''
        key : (월, 요일, 공휴일, 시간대)
        '''

        freq = '1d'
        # 시작일과 종료일 사이에 주어진 간격으로 일시 생성
        date_range = pd.date_range(start=dt_start, end=dt_end, freq=freq)
        tmp = []
        for x in date_range:
            tmp.append(x.strftime('%Y-%m-%d') + ' 08:00:00')
            tmp.append(x.strftime('%Y-%m-%d') + ' 12:00:00')
            tmp.append(x.strftime('%Y-%m-%d') + ' 18:00:00')
            tmp.append(x.strftime('%Y-%m-%d') + ' 02:00:00')
        self.학생생성기준일시 = pd.DataFrame()
        self.학생생성기준일시['일시'] = tmp
        self.학생생성기준일시 = 구분데이터붙이기(self.학생생성기준일시, '일시')

        date_range = pd.date_range(start = pd.to_datetime(dt_start) - pd.Timedelta(value = 30, unit = 'day'), end = pd.to_datetime(dt_end) + pd.Timedelta(value = 30, unit = 'day'), freq=freq)
        tmp = []
        for x in date_range:
            tmp.append(x.strftime('%Y-%m-%d') + ' 08:00:00')
            tmp.append(x.strftime('%Y-%m-%d') + ' 12:00:00')
            tmp.append(x.strftime('%Y-%m-%d') + ' 18:00:00')
            tmp.append(x.strftime('%Y-%m-%d') + ' 02:00:00')
        self.학생생성기준일 = pd.DataFrame()
        self.학생생성기준일['일시'] = tmp
        self.학생생성기준일 = 구분데이터붙이기(self.학생생성기준일, '일시')
        self.학생생성기준일['휴일'] = False
        for i in range(len(self.학생생성기준일)):
            key = (self.학생생성기준일.loc[i, '월'], self.학생생성기준일.loc[i, '요일'], self.학생생성기준일.loc[i, '공휴일'], self.학생생성기준일.loc[i, '시간대'])
            if (key[1] == '5')|(key[1] == '6')|(key[2] == True):
                 self.학생생성기준일.loc[i, '휴일'] = True
            for keys in list(self.cluster.군집s):
                if key in eval(keys):
                    m = np.mean(self.cluster.군집s[keys])
                    s = np.std(self.cluster.군집s[keys])
                    break
            self.학생생성기준일.loc[i, 'mean'] = m
            self.학생생성기준일.loc[i, 'std'] = s

        # 하루씩 진행하며 예약 생성
        tmp_고객_접속일시 = []
        tmp_예약희망일시 = []
        tmp_예약희망_point = []
        tmp_월 = []
        tmp_요일 = []
        tmp_공휴일 = []
        tmp_시간대 = []
        for i in range(len(self.학생생성기준일시)):
            self_key = (self.학생생성기준일시.loc[i, '월'], self.학생생성기준일시.loc[i, '요일'], self.학생생성기준일시.loc[i, '공휴일'], self.학생생성기준일시.loc[i, '시간대'])
            cluster_keys = list(self.hierarchy_sample_data)
            for k in cluster_keys:
                if self_key in eval(k):
                    cluster_key = k
                    continue
            # 학생 발생 숫자 샘플링
            num = random.sample(self.cluster.군집s[cluster_key], 1)[0] * weight_학생숫자
            tmp = self.hierarchy_sample_data[cluster_key].sample(int(num), replace = True, ignore_index = True)
            for j in range(len(tmp)):
                tmp_예약희망일시.append(pd.to_datetime(self.학생생성기준일시.loc[i, '일시'].strftime('%Y-%m-%d') + ' ' + tmp.loc[j, '예약희망일시'].strftime('%H:%M:%S')))
                tmp_고객_접속일시.append(tmp_예약희망일시[-1] - (tmp.loc[j, '예약희망일시'] - tmp.loc[j, '고객_접속일시']))
                tmp_예약희망_point.append(None)
                tmp_월.append(self.학생생성기준일시.loc[i, '월'])
                tmp_요일.append(self.학생생성기준일시.loc[i, '요일'])
                tmp_공휴일.append(self.학생생성기준일시.loc[i, '공휴일'])
                tmp_시간대.append(self.학생생성기준일시.loc[i, '시간대'])
        self.가예약정보 = pd.DataFrame(data = {'고객_접속일시': tmp_고객_접속일시, '예약희망일시': tmp_예약희망일시, 
                                         '예약희망_point': tmp_예약희망_point, '월': tmp_월, 
                                         '요일': tmp_요일, '공휴일': tmp_공휴일, '시간대': tmp_시간대})
        self.가예약정보['예약희망일시'] = self.가예약정보['예약희망일시'].dt.tz_localize('Asia/Seoul')
        self.가예약정보['고객_접속일시'] = self.가예약정보['고객_접속일시'].dt.tz_localize('Asia/Seoul')
        self.가예약정보 = self.가예약정보.sort_values('고객_접속일시').reset_index(drop=True)
        
        '''
        ## 군체 만들기
        # 고객_접속일시가 같은 사람 다섯 개씩 묶고 안나눠 떨어지는 부분은 5개 미만인 상태로 묶기
        # 사람마다 딕셔너리 형태로 {'접속일시': dt, '예약희망일시': [dt, dt, ...], '예약희망_point': 포인트 생성}
        '''
        self.가예약정보['접속연월일'] = [x.strftime('%Y%m%d') for x in self.가예약정보['고객_접속일시']]
        self.예약자s = []
        연월일s = self.가예약정보['접속연월일'].drop_duplicates().to_list()
        for 연월일 in 연월일s:
            index_list = self.가예약정보[self.가예약정보['접속연월일']== 연월일].index.to_list()
            while len(index_list) > 0 :
                if len(index_list) <= 3:
                    tmp_indexes = index_list.copy()
                else:
                    tmp_indexes = random.sample(index_list, 3)
                for i in tmp_indexes:
                    index_list.remove(i)
                for j in range(len(tmp_indexes)):
                    self.예약자s.append({'예약희망일시': []})
                    for i in tmp_indexes:
                        self.예약자s[-1]['예약희망일시'].append(self.가예약정보.loc[i, '예약희망일시'])
                    self.예약자s[-1]['접속일시'] = self.가예약정보.loc[tmp_indexes[j], '고객_접속일시']
                # self.예약자s[-1]['예약희망_point'] = 


        choices = random.choices(np.arange(0, len(위치_데이터)), weights = 위치_데이터['count'], k=len(self.예약자s)) # 인구수에 비례하게 고객 위치 생성

        points = []
        points_idx = []
        for idx_집계구 in choices:
            # 집계구 중심으로 수정하기
            # points.append(generate_polygon_random_points(위치_데이터.loc[idx_집계구, 'geometry'])[0])
            points.append(위치_데이터.loc[idx_집계구, 'centroid'])
            points_idx.append(idx_집계구)

        for i in range(len(self.예약자s)):
            self.예약자s[i]['예약희망_point'] = points[i]
            self.예약자s[i]['예약희망_point_idx'] = points_idx[i]

        self.예약자s = sorted(self.예약자s, key = lambda x: x['접속일시'])





        
        
    def _클러스터링결과저장(self):
        클러스터링결과 = {}
        클러스터링결과['cluster'] = self.cluster
        클러스터링결과['hierarchy_sample_data'] = self.hierarchy_sample_data

        with open('./data/simulator/클러스터링결과.pickle', 'wb') as f:
            pickle.dump(클러스터링결과, f)
    
    def _클러스터링결과불러오기(self):
        with open('./data/simulator/클러스터링결과.pickle', 'rb') as f:
            클러스터링결과 = pickle.load(f)

        self.cluster = 클러스터링결과['cluster']
        self.hierarchy_sample_data = 클러스터링결과['hierarchy_sample_data']

    def _위치데이터저장(self):
        위치데이터 = {}
        위치데이터['위치데이터'] = self.위치데이터
        위치데이터['위치_5000이내'] = self.위치_5000이내

        with open('./data/simulator/위치데이터.pickle', 'wb') as f:
            pickle.dump(위치데이터, f)
    
    def _위치데이터불러오기(self):
        with open('./data/simulator/위치데이터.pickle', 'rb') as f:
            위치데이터 = pickle.load(f)

        self.위치데이터 = 위치데이터['위치데이터']
        self.위치_5000이내 = 위치데이터['위치_5000이내']

    def _차량데이터저장(self):

        with open('./data/simulator/차량데이터.pickle', 'wb') as f:
            pickle.dump(self.차량데이터, f)
    
    def _차량데이터불러오기(self):
        with open('./data/simulator/차량데이터.pickle', 'rb') as f:
            self.차량데이터 = pickle.load(f)


    def _예약가능일판단(self, 예약자, 예약현황, 차량현황, 
                 # model
                ):
        '''
        접속일 하루 뒤부터 가능 (당일 불가능),
        구 단위로 같은 예약 가능일이 존재하도록하기.

        구 단위로 거리는 신경쓰지 말고 2시간 30분 이상이면 가능하게하기
        '''

        if self.금년 != 예약자['접속일시'].strftime('%Y'):
            self.금년 = 예약자['접속일시'].strftime('%Y')

            self.당해년공휴일 = 공휴일출력(self.금년)
            
        예약가능일 = 예약가능시간출력(접속일 = 예약자['접속일시'], 당해년공휴일 = self.당해년공휴일, 며칠뒤까지 = 며칠뒤)
        '''
        5키로 이내 30분
        
        Todo: 예약현황과 차량현황 바탕으로 feasible solution 판단해서 예약 가능한지 넣기
        
        '''
        # 예약가능일에 만약 여기 예약하면 어떤 차 임시배정할지까지 계산해서 넣기
        # 임시배정차량 중에 같은날 이웃시간 1.5km 이내 있으면 우선 배정, (2시간 간격일 때)
        # 3시간 간격이면 거리제한 없이 순서대로
        # 없으면 이웃시간 아닌 것 중에 1.5km 이내 있으면 배정, (가운데 띄워놓고 받으면 그 가운데 채우는거 판단 하기가..)
        # 없으면 주차_point에서 제일 가까운 거 새로배정, 
        # 새로배정할 차량 없으면 solver 이용해서 feasible 솔루션 탐색후 임시배정 새로.
        # feasible 솔루션 없으면 배정불가판단.
        
        for i in range(len(예약가능일)):
            '''
            업무일 일 때: 18시인지 20시인지 판단하고 서로 1.5키로 이내인거 있으면 해당걸로 배차, 없으면 새로 배차.
            18혹은 20이 아니면 그냥 순서대로 다 배차

            휴일 일 때: 
            1. 이웃시간 1.5km 이내 있는지 확인하고 우선적으로 같은 차량 배정
            2. 한 칸 건너 뛰어 2km 이내 있는지 확인하고 우선적으로 같은 차량 배정
            3. 없으면 배정 아예 안된 차 중에 주차_point가 가장 가까운거 배정
            4. 새로 배정할 차 없으면 solver 이용해서 feasible 솔루션 탐색후 임시배정 새로하기
            5. 시간제한 매우 짧게 해서 feasible 솔루션 없으면 배정불가 판단.
            '''
            # 남은 예약 중 연월일 같은 것.(같은 날)
            tmp_남예 = pd.DataFrame(self.남은예약)
            tmp_남예 = tmp_남예[tmp_남예['연월일'] == 예약가능일.loc[i, '예약일시'].strftime('%Y%m%d')]
            tmp_남예.loc[:, ['시']] = [x.strftime('%H') for x in tmp_남예['예약_일시']]
            self.tmp_남예 = tmp_남예
            
            tmp_공휴일 = 예약가능일.loc[i, '공휴일']
            tmp_요일 = 예약가능일.loc[i, '요일']
            tmp_일시 = 예약가능일.loc[i, '예약일시']
            tmp_시 = tmp_일시.strftime('%H')
            if (tmp_공휴일) | ((tmp_요일 == '5') | (tmp_요일 == '6')):
                '''
                휴일일 때
                '''
                예약가능일.drop(index = i, inplace = True)
            else: # 업무일 일 때

                tmp_차량코드s = list(set(차량현황['차량코드'].to_list()) - set(tmp_남예[tmp_남예['시'] == tmp_시]['임시배정차량코드'].to_list()))
                if tmp_시 not in ['18', '20']:
                    '''
                    18시, 20시가 아니면 아무거나 배치
                    '''

                    if len(tmp_차량코드s)==0:
                        예약가능일.drop(index = i, inplace = True)
                    else:
                        # print('평일 18, 20 이외', tmp_차량코드s)
                        예약가능일.loc[i, '임시배정차량코드'] = tmp_차량코드s[0]
                else:
                    '''
                    18시, 20시면 
                    1. 같은날 다른시 있으면 1.5키로 이내인 거 있으면 같은거 우선 매치
                    2. 없으면 새로운거 매치
                    3. 새로운거 없으면 배정불가
                    '''
                    if tmp_시 == '18':
                        tmp_반대시 = '20'
                    else:
                        tmp_반대시 = '18'

                    tmp_반대시_남예 = tmp_남예[tmp_남예['시'] == tmp_반대시].reset_index(drop = True)
                    
                    tmp_최소dist = 100000 # 100키로미터
                    tmp_최소차량코드 = '999'
                    for j in range(len(tmp_반대시_남예)):
                        if tmp_반대시_남예.loc[j, '임시배정차량코드'] in tmp_차량코드s: # 반대시에 있어도 같은 시간 이미 매칭된 차량이면 비교 안하고 패스
                            continue
                        tmp_dist = 예약자['예약희망_point'].distance(tmp_반대시_남예.loc[j, '예약_point'])
                        if tmp_dist <= tmp_최소dist:
                            tmp_최소차량코드 = tmp_반대시_남예.loc[j, '임시배정차량코드']
                            tmp_최소dist = tmp_dist
                    if tmp_최소dist <= 1500: # 반대시에 1.5km 이내 있으면 최소거리인거 매칭
                        예약가능일.loc[i, '임시배정차량코드'] = tmp_최소차량코드
                    else: # 없으면 아무거나 매칭
                        tmp_반대시_차량코드s = list(set(차량현황['차량코드'].to_list()) - set(tmp_반대시_남예['임시배정차량코드'].to_list()))
                        tmp = list(set(tmp_차량코드s) - set(tmp_반대시_차량코드s))
                        if len(tmp)==0:
                            예약가능일.drop(index = i, inplace = True)
                        else:
                            # print('평일 18, 20', tmp)
                            예약가능일.loc[i, '임시배정차량코드'] = tmp[0]
                        
            # 이웃시간s = 이웃시간판단(일시 = 예약가능일.loc[i, '예약일시'], 요일 = 예약가능일.loc[i, '요일'], 공휴일 = 예약가능일.loc[i, '공휴일'])
            
            # tmp = pd.DataFrame()
            # for 시간 in 이웃시간s:
            #     tmp = pd.concat([tmp, tmp_남예[tmp_남예['시']==시간]])
            # tmp_남예 = tmp
            # if len(tmp_남예) == 0 : # 같은날 이웃한 시간대에 이미 받은 예약이 없음.
            #     pass
            # self.남은예약[self.남은예약['연월일']==]
        
        # 예약가능일['is_예약가능'] = True

        return 예약가능일


class Cluster():
    def __init__(self, value_kinds, 모든키, 모든원소):
        self.모든키 = 모든키
        self.모든원소 = 모든원소
        self.군집keys = []
        for 공휴일 in value_kinds['공휴일']:
            for 월 in value_kinds['월']:
                for 시간대 in value_kinds['시간대']:
                    군집 = []
                    for 요일 in value_kinds['요일']:
                        if 공휴일 == False: # 공휴일이 아니면 각각 따로 보기
                            # if (월, 요일, 공휴일, 시간대) in self.모든키:
                            self.군집keys.append([(월, 요일, 공휴일, 시간대)])

                        else: # 공휴일이면 요일 상관없이 다 합치기
                            # if (월, 요일, 공휴일, 시간대) in self.모든키:
                            군집.append((월, 요일, 공휴일, 시간대))
                    if len(군집) > 0:
                        self.군집keys.append(군집)

        self.군집s = {}
        for 군집key in self.군집keys:
            self.군집s[str(sorted(군집key))] = []
            for key in 군집key:
                tmp_list = self.모든원소.loc[(self.모든원소['월']==key[0])&(self.모든원소['요일']==key[1])&(self.모든원소['공휴일']==key[2])&(self.모든원소['시간대']==key[3]), 'count'].to_list()
                if len(tmp_list) == 0:
                    if  key[2] == False:
                        tmp_list = [0]
                    elif key[3] == '심야':
                        tmp_list = [0]
                self.군집s[str(sorted(군집key))] += tmp_list

        
        ### 집단 2개 이하인 것 평균 기준으로 합치기 (0개 인 것도 고려)
        terminal = False
        while terminal == False:
            terminal = True
            for key in list(self.군집s):

                if len(self.군집s[key]) <= 3:
                    terminal = False
                    keys = list(self.군집s)
                    keys.remove(key)
                    tmp = []
                    for k in keys:
                        tmp.append(np.abs(np.mean(self.군집s[key]) - np.mean(self.군집s[k])))
                    self.군집s[str(sorted(eval(keys[np.argmin(tmp)]) + eval(key)))] = self.군집s[keys[np.argmin(tmp)]] + self.군집s[key]
                    self.군집s.pop(keys[np.argmin(tmp)])
                    self.군집s.pop(key)
                    break
                    
        ### 집단 30개 이하인 것 맨-휘트니 U 검정 기준으로 합치기
        #  https://blog.naver.com/PostView.naver?blogId=istech7&logNo=50152096673
        terminal = False
        while terminal == False:
            terminal = True
            for key in list(self.군집s):
                if len(self.군집s[key]) <= 30:
                    terminal = False
                    keys = list(self.군집s)
                    keys.remove(key)
                    tmp = []
                    for k in keys:
                        u_stat, p_value = mannwhitneyu(self.군집s[key], self.군집s[k], alternative='two-sided')
                        
                        tmp.append(p_value)
                    # print(key, k, p_value)
                    self.군집s[str(sorted(eval(keys[np.argmax(tmp)]) + eval(key)))] = self.군집s[keys[np.argmax(tmp)]] + self.군집s[key]
                    self.군집s.pop(keys[np.argmax(tmp)])
                    self.군집s.pop(key)
                    break
        self.시작_군집s = self.군집s.copy()
        self.시작_군집개수 = len(self.시작_군집s)

        
        '''
        거리척도 우선 from scipy.stats import ks_2samp 기준.
        다른거 더 조사해서 옵션으로 넣어보기.
        '''
        self.군집간거리 = {}
        tmp_keys = list(self.군집s)
        for i in range(len(tmp_keys)):
            for j in range(i+1, len(tmp_keys)):
                stat, p_value = ks_2samp(self.군집s[tmp_keys[i]], self.군집s[tmp_keys[j]])
                self.군집간거리[str(sorted([tmp_keys[i], tmp_keys[j]]))] = (stat, p_value)
        self.history = []
    def _01_군집화(self):
        '''
        거리척도 우선 from scipy.stats import ks_2samp 기준.
        다른거 더 조사해서 옵션으로 넣어보기.
        '''
        while len(self.군집s) > 1:
            # p value가 가장 큰 key
            max_key = max(self.군집간거리, key=lambda k: self.군집간거리[k][1])
            # print(max_key)
            max_value = self.군집간거리[max_key]
            key_a = eval(max_key)[0]
            key_b = eval(max_key)[1]
            

            
            new_key = str(sorted(eval(key_a) + eval(key_b)))

            self.군집s[new_key] = self.군집s[key_a] + self.군집s[key_b]
            a = self.군집s.pop(key_a)
            b = self.군집s.pop(key_b)
            
            tmp_keys = list(self.군집s)
            tmp_keys.remove(new_key)
            for key in tmp_keys:
                stat, p_value = ks_2samp(self.군집s[new_key], self.군집s[key])
                self.군집간거리[str(sorted([new_key, key]))] = (stat, p_value)

                self.군집간거리.pop(str(sorted([key_a, key])))
                self.군집간거리.pop(str(sorted([key_b, key])))
            self.군집간거리.pop(str(sorted([key_b, key_a])))
            # history : [[key, [statistic, p-value], 왼원소, 오른원소]]
            self.history.append([max_key, max_value, a, b])

        self.last_군집s = self.군집s.copy()

        self.p_values = []
        self.stats = []
        for i, value in enumerate(self.history):
            self.p_values.append(value[1][1])
            self.stats.append(value[1][0])

        plt.plot(self.p_values)
        plt.show()
    def _02_cutting(self, method = 'threshold', values = {}):
        if method == 'threshold':
            self._cutby_thres(**values)
        elif method == 'number':
            self._cutby_num(**values)
        else:
            self._cutby_thres(**values)
        
    def _cutby_num(self, num_cluster = 5):
        '''
        num_cluster : 개수로 자르기


        
        '''
        plt.plot(self.p_values)
        plt.hlines(y = self.p_values[-(num_cluster-1)], 
                   xmin = 0, xmax = len(self.p_values)-1,
                   colors = 'red', linestyles = 'dashed')
        plt.show()
        
        self.군집s = self.last_군집s.copy()
        for i in range(1, num_cluster):
            # history : [[key, [statistic, p-value], 왼원소, 오른원소]]
            self.history[-i]
            
            tmp_key = self.history[-i][0]
            key_a = eval(tmp_key)[0]
            key_b = eval(tmp_key)[1]
            a = self.history[-i][2]
            b = self.history[-i][3]
            self.군집s.pop(str(sorted(eval(key_a) + eval(key_b))))
            self.군집s[key_a] = a
            self.군집s[key_b] = b
    
        
    def _cutby_thres(self, threshold = 0.05):
        '''
        threshold : 값 기준으로 자르기
        '''
        num_cluster = len(self.p_values) + 1
        for i in range(len(self.p_values)):
            
            if self.p_values[i] <= threshold:
                break
            num_cluster -= 1
        
        plt.plot(self.p_values)
        plt.hlines(y = self.p_values[-(num_cluster-1)], 
                   xmin = 0, xmax = len(self.p_values)-1,
                   colors = 'red', linestyles = 'dashed')
        plt.show()
        
        self.군집s = self.last_군집s.copy()
        for i in range(1, num_cluster):
            # history : [[key, [statistic, p-value], 왼원소, 오른원소]]
            self.history[-i]
            
            tmp_key = self.history[-i][0]
            key_a = eval(tmp_key)[0]
            key_b = eval(tmp_key)[1]
            a = self.history[-i][2]
            b = self.history[-i][3]
            self.군집s.pop(str(sorted(eval(key_a) + eval(key_b))))
            self.군집s[key_a] = a
            self.군집s[key_b] = b
        



