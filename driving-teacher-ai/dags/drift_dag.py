# dags/my_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago

from datetime import datetime, timedelta
from model_train_deploy.drift import drift


# 분기 함수 정의
def branch_func(**kwargs):
    task_instance = kwargs['task_instance']
    drift_result = task_instance.xcom_pull(task_ids='check_drift')
    if drift_result:
        return 'trigger_rl_train'
    else:
        return 'do_nothing'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 6, 28, 2, 0, 0),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
}

# drift DAG 정의
with DAG(
    'drift_dag',
    default_args=default_args,
    description='A drift detection DAG',
    schedule_interval=timedelta(days=1),
    max_active_runs=1,
) as dag:
    
    # drift 체크 태스크
    check_drift = PythonOperator(
        task_id='check_drift',
        python_callable=drift,
        dag=dag,
    )
    
    # 분기 태스크
    branch = BranchPythonOperator(
        task_id='branch_task',
        python_callable=branch_func,
        provide_context=True,
        dag=dag,
    )
        
    # rl_train DAG 트리거 태스크
    trigger_rl_train = TriggerDagRunOperator(
        task_id='trigger_rl_train',
        trigger_dag_id='rl_train',
        execution_date="{{ execution_date }}",
        conf={"logical_date": "{{ ts }}"},
    )
    # rl_train DAG 완료 감지 태스크
    wait_for_rl_train = ExternalTaskSensor(
        task_id='wait_for_rl_train',
        external_dag_id='rl_train',
        external_task_id=None,  # None이면 DAG의 모든 태스크가 완료될 때까지 기다림
        mode='poke',
        timeout=60 * 60 * 23,  # 최대 대기 시간 설정 (예: 23시간)
        poke_interval=300,  # 5분 간격으로 체크
        allowed_states=['success'],  # 완료 상태 감지
        failed_states=['failed'],  # 실패 상태 감지
        execution_date_fn=lambda dt: dt,
        dag=dag,
    )
    
    # 완료 태스크
    finish = DummyOperator(
        task_id='finish',
        dag=dag,
    )
    
    # 아무것도 하지 않는 태스크
    do_nothing = PythonOperator(
        task_id='do_nothing',
        python_callable=lambda: print("No drift detected."),
        dag=dag,
    )
    
    # 태스크 순서 정의
    check_drift >> branch
    branch >> trigger_rl_train >> wait_for_rl_train >> finish
    branch >> do_nothing >> finish
