# driving-teacher-ai

**환경** :  
python=3.9  
pip install pandas==2.2.1 scikit-learn==1.3.0  

**사용법** :  
from recommendation import recommend  

recommend(location:tuple, selected_columns:list) : 함수

**input** : 
location, selected_columns

    location : (lat, long)   
        위 형태의 length 2짜리 튜플, 각각 float  
         
    selected_columns : ['가격', '서비스점수', '강의점수', '시설점수', '거리']   
        위 형태의 length 1 ~ 5 의 리스트. 무엇을 선택했는지에 대한 정보.  
        string 형태 바꾸려면 같은 모듈에 있는 이름정의 dictionary를 수정.  

**output** :
학원코드 string.
(models/academy_data.csv 에 있는 name 컬럼을 읽음.)
        

recommendation.py 파일의 아래 dictionary 변수를 수정하여 input 값 기준 변경. (어떤 것 선택했는지)  
이름정의 = {'가격': 'lesson_price', '서비스점수': 'service_rate', '강의점수': 'lecutre_rate', '시설점수': 'facility_rate', '거리': 'distance_m'}

**주의사항** :
models 폴더와 recommendation.py 경로는 상대경로로 지금과 동일하게 돼있어야 함.  
