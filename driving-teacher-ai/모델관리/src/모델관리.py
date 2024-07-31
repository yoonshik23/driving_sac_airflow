
from .db_io import Engine
class Model_management():
    def __init__(self, execution_date, db_info):
        try:
            utc_time = pd.to_datetime(time_str).tz_localize('UTC')
        except:
            utc_time = pd.to_datetime(time_str)
        self.execution_date = utc_time.tz_convert('Asia/Seoul')
        self.db_handler = Engine(db_info)
        
    def run(self):
        self._01_make_cluster()
        self._02_train_rl()
        self._03_validation()
        self._04_save()
    def measure_drift(self):
        self._클러스터링결과불러오기()
        pass
    def make_cluster(self):
        pass
    def train_rl(self):
        pass
    def validation(self):
        pass
    def save(self):
        pass


    def _클러스터링결과불러오기(self):
        with open('./data/simulator/클러스터링결과.pickle', 'rb') as f:
            클러스터링결과 = pickle.load(f)

        self.cluster = 클러스터링결과['cluster']
        self.hierarchy_sample_data = 클러스터링결과['hierarchy_sample_data']