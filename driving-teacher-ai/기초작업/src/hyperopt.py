
import ray
from hyperopt import hp, Trials, tpe, fmin

class hyperopt_병렬처리():
    def __init__(self, 주차장개수, data_고객위치, data_주차장위치, dict_주차장_고객_거리, num_cpus = 8):
        self.num_cpus = num_cpus
        ray.init(num_cpus = self.num_cpus)
        self.주차장개수 = 주차장개수
        self.data_고객위치 = data_고객위치
        self.data_주차장위치 = data_주차장위치
        self.dict_주차장_고객_거리 = dict_주차장_고객_거리

    def 최적화실행(self, timeout = 3600):

        def hyper_opt(주차장개수, data_고객위치, data_주차장위치, dict_주차장_고객_거리, timeout):
            import pandas as pd
            import requests
            import json
            import pickle
            import datetime
            import numpy as np
            
            
            # from ortools.linear_solver import pywraplp
            from ortools.sat.python import cp_model
            from ortools.linear_solver import pywraplp
            from hyperopt import STATUS_OK
            
            class 주차장거리_해계산기():
                '''
                for Genetic algorithmn
                '''
                # url = "https://apis-navi.kakaomobility.com/v1/future/directions" # 미래길찾기 url. 하루 만건정도 제한이 있음.
                url = "https://apis-navi.kakaomobility.com/v1/destinations/directions" # 다중목적지 길찾기 url. 출발지를 30개까지 지정 가능해 하루 최대 30만건 호출 가능. 하지만 현재시간 기준으로만 알려줌.
                api_key = '03eb0935432befc99575d4cf5cd16641'
                # api_key = 'f8df45aec27d46580551f1fe6b0bcadd'
                headers = {
                "Authorization": f"KakaoAK {api_key}",
                "Content-Type": "application/json"
                }
                
                def __init__(self, data_고객위치, data_주차장위치, dict_주차장_고객_거리 = {}, 주차장개수 = 10
                            ):
                    '''
                    dict_지점간거리: 
                        key : str(set([지점1, 지점2]))
                        value : 지점 간 이동시간
            
            
                    '''
                    self.data_고객위치 = data_고객위치
                    self.data_고객위치['count'] = self.data_고객위치['count'].astype(int)
                    self.data_주차장위치 = data_주차장위치
                    
                    self.dict_주차장_고객_거리 = dict_주차장_고객_거리
                    self.주차장개수 = 주차장개수
                    self.주차장capa = int(1.1*(self.data_고객위치['count'].sum()/self.주차장개수))
            
                def hopt_func(self, space):
                    
                    self._01_해입력(space['결정변수'])
                    self._02_가까운주차장탐색()
                    self._03_지점간거리채우기()
                    return {'loss': self._04_해계산(), 'status': STATUS_OK}         
            
                def opt_func(self, 결정변수):
                    
                    self._01_해입력(결정변수)
                    self._02_가까운주차장탐색()
                    self._03_지점간거리채우기()
                    return self._04_해계산()
                    
            
            
                def _01_해입력(self, 결정변수):
                    '''
                    해 : index가 원소인 리스트
                    '''
                    self.해 = []
                    for i in range(len(결정변수)):
                        self.해.append(self.data_주차장위치.index[int(결정변수[i])])
                    # self.해 = 결정변수
                    if len(self.해) == self.주차장개수:
                        # print('입력해: ', self.해)
                        pass
                    else:
                        # print('입력주차장개수: ', len(self.해))
                        pass
            
                def _02_가까운주차장탐색(self, verbose = 0):
                    '''
                    주차장마다의 캐파를 고려해서 집계구와 주차장을 매칭.
                    
                    '''
                    dict_주차장까지거리 = {}
                    for idx in self.해:
                        dict_주차장까지거리[idx] = self.data_주차장위치.loc[idx,'centroid'].distance(self.data_고객위치['centroid'])
                    self.주차장까지거리 = pd.DataFrame(dict_주차장까지거리).T
                    
                    solver = pywraplp.Solver.CreateSolver('SCIP')
                    if verbose == 1:
                        solver.EnableOutput()
                
                    주차장마다_차_몇개_할당 = {}
                    for idx in self.data_주차장위치.index:
                        주차장마다_차_몇개_할당[idx] = 0
                    for idx in self.해:
                        주차장마다_차_몇개_할당[idx] += 1
                
                    주차장마다_인구_몇명_할당 = {}
                    for idx in self.해:
                        for i in self.data_고객위치.index:
                            주차장마다_인구_몇명_할당[(idx, i)] = solver.IntVar(0, self.주차장capa * self.주차장개수, f'x[{idx},{i}]')
                
                
                    # 수요 제약
                    for i in self.data_고객위치.index:
                        solver.Add(sum(주차장마다_인구_몇명_할당[idx, i] for idx in self.해) >= self.data_고객위치.loc[i, 'count'])
                
                    # 용량 제약
                    for idx in self.해:
                        solver.Add(sum(주차장마다_인구_몇명_할당[idx, i] for i in self.data_고객위치.index) <= 주차장마다_차_몇개_할당[idx] * self.주차장capa)
                
                    # 목적함수
                    solver.Minimize(sum(주차장마다_인구_몇명_할당[idx, i] * self.주차장까지거리.loc[idx, i] for i in self.data_고객위치.index for idx in self.해))
                
                    # 문제 풀이
                    status = solver.Solve()
                
                    # 결과 출력
                    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
                        # print('최적할당 끝')
                        pass
                    else:
                        print('The problem does not have an optimal solution.')
                    # return solver, status, 주차장마다_인구_몇명_할당
            
                    self.주차장마다_인구_몇명_할당 = 주차장마다_인구_몇명_할당
                    
                    # solver_주차장할당, 결정변수 = solve_주차장할당문제(self.data_고객위치, self.해, self.주차장까지거리, capacity=self.주차장capa)
                    self.dict_할당된주차장idx = {}
                    for i in self.data_고객위치.index:
                        self.dict_할당된주차장idx[i] = []
                        for idx in self.data_주차장위치.index:
                            if idx in self.해:
                                self.주차장마다_인구_몇명_할당[idx,i] = self.주차장마다_인구_몇명_할당[idx,i].solution_value()
                            else:
                                self.주차장마다_인구_몇명_할당[idx,i] = 0
                            if self.주차장마다_인구_몇명_할당[idx,i] >= 1:
                                self.dict_할당된주차장idx[i].append(idx)
                    
                def _03_지점간거리채우기(self):
                    self.해거리s = {}
                    if len(self.해) == self.주차장개수:
                    
                        # print('    api 호출 시작')
                        k = 1
                        기본_호출인풋 = {}
                        for idx in self.해:
                            기본_호출인풋[idx] = {'origin': {'x': self.data_주차장위치.loc[idx, 'centroid'].x, 'y': self.data_주차장위치.loc[idx, 'centroid'].y}, 
                                                'destinations': [],
                                            'radius': 10000,
                                               'priority': 'TIME',
                                               'roadevent': 2}
            
                        for i in self.data_고객위치.index:
                            for idx in self.dict_할당된주차장idx[i]:
            
                                key_val = (idx, i)
                                if key_val not in self.dict_주차장_고객_거리.keys():
                                    기본_호출인풋[idx]['destinations'].append({'x': self.data_고객위치.loc[i, 'centroid'].x, 'y': self.data_고객위치.loc[i, 'centroid'].y, 'key': i})
                
                                    if len(기본_호출인풋[idx]['destinations']) == 30:
                                        self._카카오길찾기api실행(호출인풋 = 기본_호출인풋[idx], 주차장_idx = idx)
                                        기본_호출인풋[idx]['destinations'] = []
                                else:
                                    self.해거리s[idx, i] = self.dict_주차장_고객_거리[idx,i]
                            for idx in self.해:
                                if len(기본_호출인풋[idx]['destinations']) > 0:
                                    self._카카오길찾기api실행(호출인풋 = 기본_호출인풋[idx], 주차장_idx = idx)
                                    기본_호출인풋[idx]['destinations'] = []
                    
                        with open('./data/수리적최적화_지점간거리.pickle', 'wb') as f:
                            pickle.dump(self.dict_주차장_고객_거리, f, pickle.HIGHEST_PROTOCOL)
                        # print('    api 종료')
                        
            
                def _04_해계산(self):
                    self.해거리 = 0
                    for i in self.data_고객위치.index:
                        self.dict_할당된주차장idx[i] = []
                        for idx in self.해:
                            인구 = self.주차장마다_인구_몇명_할당[idx,i]
            
                            if 인구 >= 1:
                                self.해거리 += 인구 * self.해거리s[idx,i]

                    return self.해거리
                def _카카오길찾기api실행(self, 호출인풋, 주차장_idx):
                    response = requests.post(self.url, headers = self.headers, data = json.dumps(호출인풋))
                    if response.status_code == 200:
                        for j in range(len(response.json()['routes'])):
                            i = int(response.json()['routes'][j]['key'])
                            dic_key = (주차장_idx, i)
                            if response.json()['routes'][j]['result_code'] == 104:
                                self.dict_주차장_고객_거리[dic_key] = 0
                                self.해거리s[dic_key] = 0
                            elif response.json()['routes'][j]['result_code'] == 0:
                                self.dict_주차장_고객_거리[dic_key] = response.json()['routes'][j]['summary']['duration']
                                self.해거리s[dic_key] = response.json()['routes'][j]['summary']['duration']
                            elif response.json()['routes'][j]['result_code'] == 1: # 길찾기 결과를 찾을 수 없음
                                self.dict_주차장_고객_거리[dic_key] = 10000
                                self.해거리s[dic_key] = 10000
                            elif response.json()['routes'][j]['result_code'] == 304: # 목적지가 설정한 길 찾기 반경 범위를 벗어남
                                self.dict_주차장_고객_거리[dic_key] = 10000
                                self.해거리s[dic_key] = 10000
                            else:
                                print('    api 호출 에러', response.json()['routes'][j]['result_code'])
                                print(response.json()['routes'][j]['result_msg'])
                    else:
                        print('    api 호출 에러', response.status_code)
                        print(response.json())
                    return response
            
            solver = 주차장거리_해계산기(data_고객위치 = data_고객위치, data_주차장위치 = data_주차장위치,
                                dict_주차장_고객_거리 = dict_주차장_고객_거리, 
                                주차장개수 = 주차장개수)
            space = {'결정변수': [hp.randint(str(x), len(data_주차장위치)) for x in range(주차장개수)]}
            trials = Trials()
            best = fmin(fn= solver.hopt_func,
                        space= space,
                        algo= tpe.suggest,
                        max_evals = 10000,
                        trials= trials,
                           timeout = timeout)
            return best, min(trials.losses())
        remote_cal = ray.remote(hyper_opt)
        result = [remote_cal.remote(주차장개수 = self.주차장개수, data_고객위치 = ray.put(self.data_고객위치), 
                                    data_주차장위치 = ray.put(self.data_주차장위치), 
                                    dict_주차장_고객_거리 = ray.put(self.dict_주차장_고객_거리), 
                                    timeout = timeout) for _ in range(self.num_cpus)]
        
        results = ray.get(result)
        ray.shutdown()
        return results


