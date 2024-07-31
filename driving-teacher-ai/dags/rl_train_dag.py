# dags/my_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from model_train_deploy.rl_train import rl_train

def load_data_to_db():
    # 사용 예시
    print(my_function())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 6, 28, 1, 0, 0),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
}


# rl_train DAG 정의
with DAG(
    'rl_train',
    default_args=default_args,
    description='A RL training DAG',
    schedule_interval=None,  # 수동으로 트리거될 때만 실행
    max_active_runs = 1,
) as dag:
    
    # 태스크 정의
    t1 = PythonOperator(
        task_id='train_model',
        python_callable=rl_train,
        op_kwargs={
            'execution_date': '{{ dag_run.conf["logical_date"] }}'
        },
        dag=dag,
    )
